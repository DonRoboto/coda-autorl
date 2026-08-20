#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CODA scheduler with KL-only I2O and donor-relative incremental GP-uncertainty O2I.

Experimental variants
---------------------
- ``full``: CODA (KL context + donor-relative incremental uncertainty O2I)
- ``i2o``:  CODA-I2O (KL-based learner context, fixed entropy baseline)
- ``o2i``:  CODA-O2I (PB2-style context + incremental uncertainty O2I)

Communication design
--------------------
I2O uses a single PPO diagnostic: approximate policy KL divergence. The learner
callback converts KL into a bounded policy-update state and maintains a
lineage-consistent EMA. Complete CODA and CODA-I2O supply preceding reward and
policy-update state as causal context to the outer TV-GP.

O2I communicates the additional posterior uncertainty introduced by a
candidate relative to the inherited donor configuration. Both queries use the
same donor context and the donor's latest applied entropy coefficient:

    sigma_c(h) = sigma_obs([context_d, h,   a_d])
    sigma_d    = sigma_obs([context_d, h_d, a_d])

    U_delta(h) = clip(
        max(sigma_c(h) - sigma_d, 0)
        / (sigma_prior([context_d, h, a_d]) + eps_U),
        0, 1,
    ).

Thus, a candidate receives a large O2I signal only when it is more uncertain
than the state already inherited from the donor. If donor and candidate are
equally uncertain, the incremental signal is approximately zero even when both
are far from observed data. The donor entropy is used only as a known GP query
condition; it is not added to the new actuator:

    a_new = clip(a0 + min(delta_max, omega_U * U_delta(h)), 0, a_guard).

The UCB acquisition evaluates each candidate under the entropy induced by its
uncertainty. After the continuous candidate is converted to executable values,
uncertainty and entropy are recomputed for the executable configuration.

Only the completed-observation GP produces the O2I message. The auxiliary GP
containing pending batch proposals is used solely for within-event UCB
uncertainty diversification.

The outer optimizer adapts exactly four PPO coordinates:

    train_batch_size, lambda (GAE), clip_param, lr.

Entropy remains excluded from the acquisition search coordinates and is retained
only as an execution feature. Learning rate is represented in log10 space
inside the GP geometry while RLlib receives the original executable value.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import _PBTTrialState
from ray.tune.utils.util import flatten_dict, unflatten_dict

from configs.ppo_config import (
    CODA_LOG10_COORDINATES,
    MIN_VALID_TRANSITIONS,
    MAX_GP_POINTS,
    REWARD_Z_CLIP,
    QUANTILE_FRACTION,
)

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from ray.tune.schedulers.pb2_utils import TV_SquaredExp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "CODA requires scikit-learn and Ray Tune's PB2 utilities."
    ) from exc

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _trial_key(trial: Trial) -> str:
    """Return the stable Tune trial identifier used in CODA's table."""
    trial_id = getattr(trial, "trial_id", None)
    return str(trial_id) if trial_id else str(trial)


def _fill_config(
    config: Dict,
    hyperparam_bounds: Dict[str, Union[dict, list, tuple]],
) -> Dict:
    """Fill missing hyperparameters from numeric [low, high] bounds."""
    filled = {}
    for name, bounds in hyperparam_bounds.items():
        if isinstance(bounds, dict):
            if name not in config:
                config[name] = {}
            filled[name] = _fill_config(config[name], bounds)
        elif isinstance(bounds, (list, tuple)) and name not in config:
            if len(bounds) != 2:
                raise ValueError(f"Invalid bounds for {name}: {bounds}")
            low, high = bounds
            config[name] = filled[name] = np.random.uniform(low, high)
    return filled


def _normalize(data: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Affine normalization using explicit 2 x d limits (no clipping)."""
    data = np.asarray(data, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    low, high = limits[0], limits[1]
    scale = high - low
    if np.any(scale <= 0):
        raise ValueError(
            f"All normalization bounds must have positive width: {limits}"
        )
    return (data - low) / scale


def _denormalize(data: np.ndarray, limits: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    low, high = limits[0], limits[1]
    return data * (high - low) + low


_LOG_SCALE_HYPERPARAMS = frozenset(CODA_LOG10_COORDINATES)


def _transform_hyperparameters(
    data: np.ndarray,
    names: Sequence[str],
) -> np.ndarray:
    """Map executable HP values to CODA's internal model geometry."""
    arr = np.asarray(data, dtype=np.float64).copy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError("Hyperparameter matrix shape does not match names")

    for j, name in enumerate(names):
        if name in _LOG_SCALE_HYPERPARAMS:
            if np.any(arr[:, j] <= 0.0):
                raise ValueError(
                    f"Log-scaled hyperparameter {name!r} must be > 0"
                )
            arr[:, j] = np.log10(arr[:, j])
    return arr


def _inverse_transform_hyperparameters(
    data: np.ndarray,
    names: Sequence[str],
) -> np.ndarray:
    """Map CODA model coordinates back to executable HP values."""
    arr = np.asarray(data, dtype=np.float64).copy()
    one_dimensional = arr.ndim == 1
    if one_dimensional:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError("Hyperparameter matrix shape does not match names")

    for j, name in enumerate(names):
        if name in _LOG_SCALE_HYPERPARAMS:
            arr[:, j] = np.power(10.0, arr[:, j])
    return arr.reshape(-1) if one_dimensional else arr


def _transformed_bounds(
    bounds: Dict[str, Tuple[float, float]],
    names: Sequence[str],
) -> np.ndarray:
    """Return 2 x d bounds in CODA's internal model geometry."""
    raw = np.asarray([bounds[name] for name in names], dtype=np.float64).T
    low = _transform_hyperparameters(raw[0].reshape(1, -1), names)[0]
    high = _transform_hyperparameters(raw[1].reshape(1, -1), names)[0]
    limits = np.vstack((low, high))
    if np.any(limits[1] <= limits[0]):
        raise ValueError(f"Invalid transformed hyperparameter bounds: {limits}")
    return limits


def _robust_reward_to_unit_interval(
    train_rewards: np.ndarray,
    query_reward: float,
    *,
    z_clip: float = 4.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Causally normalize reward context without task-specific bounds."""
    rewards = np.asarray(train_rewards, dtype=np.float64).reshape(-1)
    rewards = rewards[np.isfinite(rewards)]

    if rewards.size == 0:
        raise ValueError(
            "Cannot normalize reward context without finite training rewards"
        )
    if not np.isfinite(query_reward):
        raise ValueError("query_reward must be finite")
    if z_clip <= 0:
        raise ValueError("z_clip must be > 0")

    center = float(np.median(rewards))
    q25, q75 = np.percentile(rewards, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)

    if not np.isfinite(scale) or scale < eps:
        mad = float(np.median(np.abs(rewards - center)))
        scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < eps:
        scale = float(np.std(rewards))

    if not np.isfinite(scale) or scale < eps:
        return np.full(rewards.shape, 0.5, dtype=np.float64), 0.5

    train_z = np.clip((rewards - center) / scale, -z_clip, z_clip)
    query_z = float(
        np.clip((float(query_reward) - center) / scale, -z_clip, z_clip)
    )

    train_unit = (train_z + z_clip) / (2.0 * z_clip)
    query_unit = (query_z + z_clip) / (2.0 * z_clip)
    return train_unit.astype(np.float64), float(query_unit)


def _normalize_context(
    X_context_raw: np.ndarray,
    new_context_raw: np.ndarray,
    context_columns: Sequence[str],
    context_bounds: Dict[str, Tuple[float, float]],
    *,
    reward_z_clip: float = REWARD_Z_CLIP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize CODA context using only pre-decision historical data."""
    X_context_raw = np.asarray(X_context_raw, dtype=np.float64)
    new_context_raw = np.asarray(new_context_raw, dtype=np.float64).reshape(-1)

    if X_context_raw.ndim != 2:
        raise ValueError("X_context_raw must be a 2-D array")
    if X_context_raw.shape[1] != len(context_columns):
        raise ValueError("Context matrix width does not match context_columns")
    if new_context_raw.size != len(context_columns):
        raise ValueError("new_context_raw width does not match context_columns")

    X_norm = np.empty_like(X_context_raw, dtype=np.float64)
    fixed_norm = np.empty(len(context_columns), dtype=np.float64)

    for j, name in enumerate(context_columns):
        if name == "R_before":
            scaled_train, scaled_query = _robust_reward_to_unit_interval(
                X_context_raw[:, j],
                new_context_raw[j],
                z_clip=reward_z_clip,
            )
            X_norm[:, j] = scaled_train
            fixed_norm[j] = scaled_query
            continue

        if name not in context_bounds:
            raise ValueError(
                f"Missing fixed normalization bounds for context {name!r}"
            )

        limits = np.asarray(context_bounds[name], dtype=np.float64).reshape(2, 1)
        X_norm[:, j] = _normalize(
            X_context_raw[:, j].reshape(-1, 1), limits
        ).reshape(-1)
        fixed_norm[j] = float(
            _normalize(
                np.asarray([[new_context_raw[j]]], dtype=np.float64), limits
            )[0, 0]
        )

    return X_norm, fixed_norm


def _standardize_response(y: np.ndarray) -> np.ndarray:
    """PB2-compatible standardization, including [-2, 2] clipping."""
    y = np.asarray(y, dtype=np.float64)
    standardized = (y - np.mean(y, axis=0)) / (np.std(y, axis=0) + 1e-8)
    return np.clip(standardized, -2.0, 2.0)


def _safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, RuntimeError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _cast_and_clip(value: float, template, bounds: Sequence[float]):
    """Cast a BO proposal back to the trial's original scalar type."""
    low, high = float(bounds[0]), float(bounds[1])
    value = float(np.clip(value, low, high))

    if isinstance(template, (int, np.integer)) and not isinstance(template, bool):
        return int(round(value))
    if isinstance(template, np.floating):
        return type(template)(value)
    if isinstance(template, float):
        return float(value)
    try:
        return type(template)(value)
    except (TypeError, ValueError):
        return value


# -----------------------------------------------------------------------------
# Transition construction
# -----------------------------------------------------------------------------

def _prepare_gp_transitions(
    data: pd.DataFrame,
    hyperparam_bounds: Dict[str, Tuple[float, float]],
    execution_feature_bounds: Dict[str, Tuple[float, float]],
    use_learner_context: bool,
    max_gp_points: int = MAX_GP_POINTS,
) -> Tuple[pd.DataFrame, list[str]]:
    """Construct temporally aligned, lineage-consistent GP transitions."""
    if data.empty:
        return pd.DataFrame(), []

    hp_names = list(hyperparam_bounds.keys())
    execution_names = list(execution_feature_bounds.keys())
    model_feature_names = hp_names + execution_names

    df = data.reset_index(drop=False).rename(columns={"index": "_insertion_order"})
    df = df.copy()

    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)

    # Causal order is insertion order, not Time. Checkpoint inheritance can
    # restore a smaller counter than one previously seen on a discarded branch.
    df = df.sort_values(
        ["Trial", "_insertion_order"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = df.groupby("Trial", sort=False, dropna=False)

    df["T_before"] = grouped["Time"].shift(1)
    df["R_before"] = grouped["Reward"].shift(1)
    df["S_before"] = grouped["policy_update_state"].shift(1)
    df["S_valid_before"] = grouped["policy_update_state_valid"].shift(1)

    df["time_change"] = df["Time"] - df["T_before"]
    df["reward_change"] = df["Reward"] - df["R_before"]
    df["y"] = df["reward_change"] / df["time_change"]

    context_columns = ["T_before"]
    if use_learner_context:
        context_columns += ["R_before", "S_before"]

    gp_input_columns = context_columns + model_feature_names
    missing = [name for name in gp_input_columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing GP execution columns: {missing}")

    finite_inputs = np.all(
        np.isfinite(df[gp_input_columns].to_numpy(dtype=np.float64)), axis=1
    )

    valid = (
        (~df["is_synthetic"])
        & (df["time_change"] > 0)
        & np.isfinite(df["y"])
        & finite_inputs
    )
    if use_learner_context:
        valid &= (
            df["S_valid_before"].fillna(0.0).astype(float) >= 0.5
        )

    transitions = (
        df.loc[valid]
        .sort_values(["_insertion_order"], kind="mergesort")
        .reset_index(drop=True)
    )

    if max_gp_points and len(transitions) > max_gp_points:
        transitions = transitions.iloc[-max_gp_points:].reset_index(drop=True)

    return transitions, context_columns


# -----------------------------------------------------------------------------
# GP, uncertainty, and acquisition
# -----------------------------------------------------------------------------

def _fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha_candidates: Sequence[float] = (1e-10, 1e-8, 1e-6, 1e-4),
) -> GaussianProcessRegressor:
    """Fit the TV-GP, increasing diagonal regularization if necessary."""
    last_error: Optional[Exception] = None

    for alpha in alpha_candidates:
        kernel = TV_SquaredExp(variance=1.0, lengthscale=1.0, epsilon=0.1)
        model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer="fmin_l_bfgs_b",
            alpha=alpha,
            normalize_y=False,
        )
        try:
            model.fit(X, y)
            # Audit attribute: scikit-learn does not otherwise expose which
            # retry-level diagonal regularization succeeded.
            model.coda_alpha_used_ = float(alpha)
            return model
        except (np.linalg.LinAlgError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "TV-GP fit failed with alpha=%g; retrying with more "
                "regularization: %s",
                alpha,
                exc,
            )

    raise RuntimeError(
        "Unable to fit CODA TV-GP after regularization retries"
    ) from last_error



def _finite_scalar(value, default=np.nan) -> float:
    """Convert scalar-like kernel values to a finite float."""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return float(default)
    return float(arr[0])


def _extract_gp_fit_diagnostics(
    model: GaussianProcessRegressor,
) -> Tuple[float, float, float, float]:
    """Return fitted kernel variance, lengthscale, epsilon, and GP alpha.

    Ray's ``TV_SquaredExp`` exposes these as direct kernel attributes. The
    ``get_params`` fallback keeps the audit robust if the kernel is wrapped by
    a compatible scikit-learn composition in a future Ray release.
    """
    kernel = getattr(model, "kernel_", None)
    if kernel is None:
        return np.nan, np.nan, np.nan, _safe_float(
            getattr(model, "coda_alpha_used_", np.nan), np.nan
        )

    def read_parameter(*aliases: str) -> float:
        for name in aliases:
            if hasattr(kernel, name):
                value = _finite_scalar(getattr(kernel, name), np.nan)
                if np.isfinite(value):
                    return value

        try:
            params = kernel.get_params(deep=True)
        except Exception:
            params = {}

        for name in aliases:
            for key, value in params.items():
                if key == name or key.endswith(f"__{name}"):
                    scalar = _finite_scalar(value, np.nan)
                    if np.isfinite(scalar):
                        return scalar
        return float("nan")

    variance = read_parameter("variance")
    lengthscale = read_parameter("lengthscale", "length_scale")
    epsilon = read_parameter("epsilon")
    alpha_used = _safe_float(
        getattr(model, "coda_alpha_used_", np.nan), np.nan
    )
    return variance, lengthscale, epsilon, alpha_used

def _kernel_prior_variance(
    model: GaussianProcessRegressor,
    x: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Return the fitted-kernel prior variance at x."""
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    kernel = getattr(model, "kernel_", None)
    if kernel is None:
        return float("nan")

    variance = np.nan
    try:
        variance = float(np.asarray(kernel.diag(x)).reshape(-1)[0])
    except Exception:
        try:
            variance = float(np.asarray(kernel(x, x)).reshape(1, 1)[0, 0])
        except Exception:
            variance = np.nan

    if not np.isfinite(variance) or variance < eps:
        return float("nan")
    return variance


def _gp_posterior_prior_std(
    observed_model: GaussianProcessRegressor,
    fixed_context: np.ndarray,
    hp_normalized: np.ndarray,
    reference_entropy_normalized: float,
    *,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """Return posterior std, fitted-kernel prior std, and prior variance.

    The query is evaluated using only the GP fitted to completed real
    observations. The actuator coordinate is fixed to the donor's latest
    applied entropy, which is causally known before the receiver decision.
    """
    hp = np.asarray(hp_normalized, dtype=np.float64).reshape(-1)
    x_ref = np.concatenate(
        [
            np.asarray(fixed_context, dtype=np.float64).reshape(-1),
            hp,
            [float(reference_entropy_normalized)],
        ]
    ).reshape(1, -1)

    _, std = observed_model.predict(x_ref, return_std=True)
    posterior_std = float(np.asarray(std).reshape(-1)[0])

    prior_variance = _kernel_prior_variance(observed_model, x_ref, eps=eps)
    prior_std = (
        float(np.sqrt(max(prior_variance, eps)))
        if np.isfinite(prior_variance)
        else float("nan")
    )
    return posterior_std, prior_std, prior_variance


def _incremental_observed_uncertainty_signal(
    observed_model: GaussianProcessRegressor,
    fixed_context: np.ndarray,
    candidate_hp_normalized: np.ndarray,
    reference_entropy_normalized: float,
    donor_posterior_std: float,
    *,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """Return candidate posterior/prior std, incremental U, and prior variance.

    The donor-relative signal is

        U_delta = clip(
            max(sigma_candidate - sigma_donor, 0)
            / (sigma_prior_candidate + eps),
            0, 1,
        ).

    The donor and candidate use the same causal context and donor actuator
    condition. This removes the absolute-uncertainty saturation case in which
    both points are equally unknown relative to the observed GP.
    """
    candidate_posterior_std, candidate_prior_std, candidate_prior_variance = (
        _gp_posterior_prior_std(
            observed_model,
            fixed_context,
            candidate_hp_normalized,
            reference_entropy_normalized,
            eps=eps,
        )
    )

    if not np.isfinite(candidate_posterior_std):
        raise ValueError("Candidate GP posterior standard deviation is non-finite")
    if not np.isfinite(donor_posterior_std):
        raise ValueError("Donor GP posterior standard deviation is non-finite")

    additional_std = max(
        float(candidate_posterior_std) - float(donor_posterior_std),
        0.0,
    )

    if np.isfinite(candidate_prior_std):
        normalized = float(
            np.clip(
                additional_std / (float(candidate_prior_std) + eps),
                0.0,
                1.0,
            )
        )
    else:
        # The fitted TV-GP normally exposes a finite kernel diagonal. If it
        # does not, preserve the incremental interpretation instead of mapping
        # equal donor/candidate uncertainty to one.
        normalized = 1.0 if additional_std > eps else 0.0

    return (
        candidate_posterior_std,
        candidate_prior_std,
        normalized,
        candidate_prior_variance,
    )


@dataclass
class _DecisionState:
    observed_model: GaussianProcessRegressor
    fixed_context: np.ndarray
    hp_names: Tuple[str, ...]
    hp_model_bounds: np.ndarray
    reference_hp_normalized: np.ndarray
    reference_entropy_coeff: float
    reference_entropy_normalized: float
    donor_posterior_std: float
    donor_prior_std: float
    donor_prior_variance: float
    gp_kernel_variance_parameter: float
    gp_lengthscale: float
    gp_temporal_epsilon: float
    gp_alpha_used: float


@dataclass(frozen=True)
class _ProposalDiagnostics:
    # Backward-compatible names: raw_std is the candidate posterior std,
    # prior_std is the candidate prior std, and normalized is U_delta.
    uncertainty_raw_std: float
    donor_uncertainty_raw_std: float
    uncertainty_prior_std: float
    donor_uncertainty_prior_std: float
    uncertainty_normalized: float
    kernel_prior_variance: float
    donor_kernel_prior_variance: float
    acquisition_value: float
    reference_entropy_coeff: float
    gp_kernel_variance_parameter: float
    gp_lengthscale: float
    gp_temporal_epsilon: float
    gp_alpha_used: float
    gp_data_count: int


def _uncertainty_for_executable_config(
    state: _DecisionState,
    config: Dict[str, float],
) -> Tuple[float, float, float, float, float, float, float]:
    """Recompute donor-relative uncertainty for executable hyperparameters."""
    raw = np.asarray(
        [[config[name] for name in state.hp_names]], dtype=np.float64
    )
    transformed = _transform_hyperparameters(raw, state.hp_names)
    normalized = np.clip(
        _normalize(transformed, state.hp_model_bounds).reshape(-1),
        0.0,
        1.0,
    )

    (
        candidate_posterior_std,
        candidate_prior_std,
        incremental_uncertainty,
        candidate_prior_variance,
    ) = _incremental_observed_uncertainty_signal(
        state.observed_model,
        state.fixed_context,
        normalized,
        state.reference_entropy_normalized,
        state.donor_posterior_std,
    )

    return (
        candidate_posterior_std,
        state.donor_posterior_std,
        candidate_prior_std,
        state.donor_prior_std,
        incremental_uncertainty,
        candidate_prior_variance,
        state.donor_prior_variance,
    )

def _ucb_value(
    mean_model: GaussianProcessRegressor,
    variance_model: GaussianProcessRegressor,
    fixed_context: np.ndarray,
    hp_normalized: np.ndarray,
    candidate_model_transform,
) -> float:
    """PB2 UCB using a deterministic search-to-execution transform."""
    model_normalized = np.asarray(
        candidate_model_transform(np.asarray(hp_normalized, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    x = np.concatenate([fixed_context, model_normalized]).reshape(1, -1)
    mean = float(mean_model.predict(x).reshape(-1)[0])
    _, std = variance_model.predict(x, return_std=True)
    variance = float(np.asarray(std).reshape(-1)[0] ** 2)

    n_obs = int(getattr(mean_model, "X_train_", np.empty((0,))).shape[0])
    beta_t = 0.2 + max(0.0, np.log(0.4 * max(n_obs, 1)))
    kappa_t = float(np.sqrt(beta_t))
    return mean + kappa_t * variance


def _optimize_acquisition(
    mean_model: GaussianProcessRegressor,
    variance_model: GaussianProcessRegressor,
    fixed_context: np.ndarray,
    normalized_search_bounds: Sequence[Tuple[float, float]],
    candidate_model_transform,
    *,
    n_restarts: int = 10,
) -> Tuple[np.ndarray, float]:
    """Optimize PB2 UCB over outer-loop coordinates only."""
    bounds = [(float(lo), float(hi)) for lo, hi in normalized_search_bounds]
    best_value = -np.inf
    best_theta: Optional[np.ndarray] = None

    for _ in range(n_restarts):
        x0 = np.asarray([np.random.uniform(lo, hi) for lo, hi in bounds])
        result = minimize(
            lambda x: -_ucb_value(
                mean_model,
                variance_model,
                fixed_context,
                x,
                candidate_model_transform,
            ),
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(theta)):
            continue
        value = _ucb_value(
            mean_model,
            variance_model,
            fixed_context,
            theta,
            candidate_model_transform,
        )
        if value > best_value:
            best_value = value
            best_theta = theta

    if best_theta is None:
        raise RuntimeError(
            "CODA acquisition optimization failed to produce a candidate"
        )
    return best_theta, float(best_value)


def _select_config(
    *,
    Xraw: np.ndarray,
    yraw: np.ndarray,
    current_nominal: Optional[np.ndarray],
    new_context: np.ndarray,
    reference_hyperparams: Dict[str, float],
    search_bounds: Dict[str, Tuple[float, float]],
    model_bounds: Dict[str, Tuple[float, float]],
    execution_feature_bounds: Dict[str, Tuple[float, float]],
    context_columns: Sequence[str],
    context_bounds: Dict[str, Tuple[float, float]],
    use_o2i_feedback: bool,
    entropy_param: str,
    base_entropy_coeff: float,
    reference_entropy_coeff: float,
    o2i_uncertainty_scale: float,
    max_entropy_increment: float,
    entropy_guard: float,
    reward_z_clip: float = REWARD_Z_CLIP,
) -> Tuple[np.ndarray, _DecisionState, float]:
    """Fit TV-GPs and return the nominal outer-loop proposal.

    Historical applied entropy remains an execution feature. For every
    candidate, CODA compares its completed-data GP posterior standard deviation
    with the donor configuration's posterior standard deviation under the same
    donor context and entropy. The positive normalized difference determines
    the candidate's new entropy from the fixed baseline. UCB then evaluates the
    candidate under that induced execution condition.
    """
    hp_names = list(search_bounds.keys())
    if list(model_bounds.keys()) != hp_names:
        raise ValueError(
            "search_bounds and model_bounds must have identical HP coordinates"
        )
    if list(execution_feature_bounds.keys()) != [entropy_param]:
        raise ValueError(
            "CODA currently expects entropy as the sole execution feature"
        )

    search_vals = _transformed_bounds(search_bounds, hp_names)
    hp_model_vals = _transformed_bounds(model_bounds, hp_names)
    execution_vals = np.asarray(
        [execution_feature_bounds[entropy_param]], dtype=np.float64
    ).T

    n_context = len(context_columns)
    n_hp = len(hp_names)
    expected_width = n_context + n_hp + 1
    if Xraw.ndim != 2 or Xraw.shape[1] != expected_width:
        raise ValueError(
            "Xraw shape is inconsistent with context/execution dimensions"
        )

    X_context_raw = Xraw[:, :n_context]
    X_hp_raw = Xraw[:, n_context:n_context + n_hp]
    X_entropy_raw = Xraw[:, n_context + n_hp:].reshape(-1, 1)

    X_context, fixed = _normalize_context(
        X_context_raw,
        new_context,
        context_columns,
        context_bounds,
        reward_z_clip=reward_z_clip,
    )
    X_hp_model = _transform_hyperparameters(X_hp_raw, hp_names)
    X_hp_norm = _normalize(X_hp_model, hp_model_vals)
    X_entropy_norm = _normalize(X_entropy_raw, execution_vals)
    X_execution = np.hstack((X_hp_norm, X_entropy_norm))
    X = np.hstack((X_context, X_execution))
    y = _standardize_response(yraw).reshape(-1, 1)

    observed_model = _fit_gp(X, y)
    (
        gp_kernel_variance_parameter,
        gp_lengthscale,
        gp_temporal_epsilon,
        gp_alpha_used,
    ) = _extract_gp_fit_diagnostics(observed_model)

    if current_nominal is None or len(current_nominal) == 0:
        variance_model = deepcopy(observed_model)
    else:
        pending_raw = np.asarray(current_nominal, dtype=np.float64)
        pending_hp = _transform_hyperparameters(
            pending_raw[:, :n_hp], hp_names
        )
        pending_hp_norm = _normalize(pending_hp, hp_model_vals)
        pending_entropy_norm = _normalize(
            pending_raw[:, n_hp:].reshape(-1, 1),
            execution_vals,
        )
        current_norm = np.hstack((pending_hp_norm, pending_entropy_norm))
        padding = np.tile(fixed, (current_norm.shape[0], 1))
        X_pending = np.hstack((padding, current_norm))
        X_aug = np.vstack((X, X_pending))
        y_aug = np.vstack((y, np.zeros((current_norm.shape[0], 1))))
        variance_model = _fit_gp(X_aug, y_aug)

    entropy_low = float(execution_vals[0, 0])
    entropy_high = float(execution_vals[1, 0])

    baseline_entropy = float(
        np.clip(base_entropy_coeff, entropy_low, min(entropy_high, entropy_guard))
    )

    reference_entropy = _safe_float(
        reference_entropy_coeff, baseline_entropy
    )
    reference_entropy = float(
        np.clip(reference_entropy, entropy_low, min(entropy_high, entropy_guard))
    )
    reference_entropy_norm = float(
        _normalize(
            np.asarray([[reference_entropy]], dtype=np.float64),
            execution_vals,
        )[0, 0]
    )

    try:
        reference_hp_raw = np.asarray(
            [[reference_hyperparams[name] for name in hp_names]],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(
            f"Missing donor/reference hyperparameter for {exc.args[0]!r}"
        ) from exc

    if not np.all(np.isfinite(reference_hp_raw)):
        raise ValueError("Donor/reference hyperparameters contain non-finite values")

    reference_hp_model = _transform_hyperparameters(
        reference_hp_raw,
        hp_names,
    )
    reference_hp_norm = np.clip(
        _normalize(reference_hp_model, hp_model_vals).reshape(-1),
        0.0,
        1.0,
    )

    if use_o2i_feedback:
        (
            donor_posterior_std,
            donor_prior_std,
            donor_prior_variance,
        ) = _gp_posterior_prior_std(
            observed_model,
            fixed,
            reference_hp_norm,
            reference_entropy_norm,
        )

        if not np.isfinite(donor_posterior_std):
            raise ValueError(
                "Donor GP posterior standard deviation is non-finite"
            )
    else:
        donor_posterior_std = 0.0
        donor_prior_std = 0.0
        donor_prior_variance = 0.0

    def candidate_model_transform(hp_norm: np.ndarray) -> np.ndarray:
        hp_norm = np.asarray(hp_norm, dtype=np.float64).reshape(-1)
        if hp_norm.size != len(hp_names):
            raise ValueError("Candidate HP vector has incorrect dimensionality")

        hp_norm_clipped = np.clip(hp_norm, 0.0, 1.0)

        if use_o2i_feedback:
            _, _, incremental_uncertainty, _ = (
                _incremental_observed_uncertainty_signal(
                    observed_model,
                    fixed,
                    hp_norm_clipped,
                    reference_entropy_norm,
                    donor_posterior_std,
                )
            )
            increment = min(
                float(max_entropy_increment),
                float(o2i_uncertainty_scale) * incremental_uncertainty,
            )
        else:
            increment = 0.0

        # The donor entropy is only a GP query condition. The new actuator is
        # generated from the fixed baseline, avoiding accumulated entropy.
        applied_entropy = float(
            np.clip(
                baseline_entropy + increment,
                entropy_low,
                min(entropy_high, float(entropy_guard)),
            )
        )
        ent_norm = float(
            _normalize(
                np.asarray([[applied_entropy]], dtype=np.float64),
                execution_vals,
            )[0, 0]
        )
        return np.concatenate((hp_norm_clipped, [ent_norm]))

    normalized_search_bounds = []
    for j in range(search_vals.shape[1]):
        model_lo, model_hi = hp_model_vals[:, j]
        search_lo, search_hi = search_vals[:, j]
        denom = model_hi - model_lo
        normalized_search_bounds.append(
            ((search_lo - model_lo) / denom, (search_hi - model_lo) / denom)
        )

    xt_norm, acquisition_value = _optimize_acquisition(
        observed_model,
        variance_model,
        fixed,
        normalized_search_bounds,
        candidate_model_transform,
        n_restarts=10,
    )

    xt_model = _denormalize(xt_norm, hp_model_vals).astype(np.float64)
    xt = _inverse_transform_hyperparameters(xt_model, hp_names)

    state = _DecisionState(
        observed_model=observed_model,
        fixed_context=np.asarray(fixed, dtype=np.float64),
        hp_names=tuple(hp_names),
        hp_model_bounds=np.asarray(hp_model_vals, dtype=np.float64),
        reference_hp_normalized=np.asarray(
            reference_hp_norm, dtype=np.float64
        ),
        reference_entropy_coeff=reference_entropy,
        reference_entropy_normalized=reference_entropy_norm,
        donor_posterior_std=float(donor_posterior_std),
        donor_prior_std=float(donor_prior_std),
        donor_prior_variance=float(donor_prior_variance),
        gp_kernel_variance_parameter=gp_kernel_variance_parameter,
        gp_lengthscale=gp_lengthscale,
        gp_temporal_epsilon=gp_temporal_epsilon,
        gp_alpha_used=gp_alpha_used,
    )
    return np.asarray(xt, dtype=np.float64), state, acquisition_value


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------

class CODAPPOOptimizer(PopulationBasedTraining):
    """CODA with KL-only I2O and donor-relative incremental uncertainty O2I."""

    VALID_VARIANTS = {"full", "i2o", "o2i"}

    def __init__(
        self,
        *,
        time_attr: str = "time_total_s",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        perturbation_interval: float = 60.0,
        hyperparam_bounds: Optional[
            Dict[str, Union[dict, list, tuple]]
        ] = None,
        context_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        quantile_fraction: float = QUANTILE_FRACTION,
        variant: str = "full",
        min_valid_transitions: int = MIN_VALID_TRANSITIONS,
        max_gp_points: int = MAX_GP_POINTS,
        entropy_param: str = "entropy_coeff",
        base_entropy_coeff: float = 0.0,
        o2i_uncertainty_scale: float = 0.008,
        max_entropy_increment: float = 0.008,
        entropy_guard: float = 0.008,
        reward_z_clip: float = REWARD_Z_CLIP,
        log_config: bool = True,
        require_attrs: bool = True,
        synch: bool = False,
    ):
        hyperparam_bounds = hyperparam_bounds or {}
        if not hyperparam_bounds:
            raise TuneError("`hyperparam_bounds` must be specified for CODA.")

        variant = str(variant).lower()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(self.VALID_VARIANTS)}"
            )

        self.variant = variant
        self.use_learner_context = variant in {"full", "i2o"}
        self.use_o2i_feedback = variant in {"full", "o2i"}
        self.min_valid_transitions = int(min_valid_transitions)
        self.max_gp_points = int(max_gp_points)
        self.entropy_param = entropy_param
        self.base_entropy_coeff = float(base_entropy_coeff)
        self.o2i_uncertainty_scale = float(o2i_uncertainty_scale)
        self.max_entropy_increment = float(max_entropy_increment)
        self.entropy_guard = float(entropy_guard)
        self.reward_z_clip = float(reward_z_clip)

        if self.min_valid_transitions < 2:
            raise ValueError("min_valid_transitions must be >= 2")
        if self.reward_z_clip <= 0:
            raise ValueError("reward_z_clip must be > 0")
        if self.base_entropy_coeff < 0:
            raise ValueError("base_entropy_coeff must be >= 0")
        if self.o2i_uncertainty_scale < 0:
            raise ValueError("o2i_uncertainty_scale must be >= 0")
        if self.max_entropy_increment < 0:
            raise ValueError("max_entropy_increment must be >= 0")
        if self.entropy_guard <= 0:
            raise ValueError("entropy_guard must be > 0")

        super().__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            hyperparam_mutations=hyperparam_bounds,
            quantile_fraction=quantile_fraction,
            resample_probability=0,
            custom_explore_fn=None,
            log_config=log_config,
            require_attrs=require_attrs,
            synch=synch,
        )

        self.data = pd.DataFrame()
        self._hyperparam_bounds = hyperparam_bounds
        self._hyperparam_bounds_flat = flatten_dict(
            hyperparam_bounds, prevent_delimiter=True
        )
        self._validate_bounds(
            self._hyperparam_bounds_flat,
            "hyperparam_bounds",
        )

        self._context_bounds = context_bounds or {}
        required_context = {"T_before"}
        if self.use_learner_context:
            required_context |= {"S_before"}
        missing = required_context.difference(self._context_bounds)
        if missing:
            raise ValueError(
                f"Missing CODA context bounds: {sorted(missing)}"
            )
        self._validate_bounds(
            {k: self._context_bounds[k] for k in required_context},
            "context_bounds",
        )

        self._model_bounds_flat = deepcopy(self._hyperparam_bounds_flat)
        max_applied_entropy = self.base_entropy_coeff + self.max_entropy_increment
        if max_applied_entropy > self.entropy_guard + 1e-12:
            raise ValueError(
                "entropy_guard must be >= base_entropy_coeff + max increment"
            )
        if max_applied_entropy <= 0.0:
            max_applied_entropy = 1e-12
        self._execution_feature_bounds_flat = {
            self.entropy_param: [0.0, float(max_applied_entropy)]
        }

        self.last_exploration_time = -np.inf
        self.current_nominal: Optional[np.ndarray] = None
        self._lineage_generation = 0

    @staticmethod
    def _validate_bounds(bounds: Dict, name: str) -> None:
        for key, value in bounds.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"{name}[{key!r}] must be [low, high], got {value}"
                )
            if float(value[0]) >= float(value[1]):
                raise ValueError(
                    f"{name}[{key!r}] must satisfy low < high, got {value}"
                )

    def _context_columns(self) -> list[str]:
        columns = ["T_before"]
        if self.use_learner_context:
            columns += ["R_before", "S_before"]
        return columns

    @staticmethod
    def _bridge(config: Dict) -> Dict:
        model = config.setdefault("model", {})
        custom = model.setdefault("custom_model_config", {})
        return custom.setdefault("_coda_bridge", {})

    def _set_trial_identity(self, config: Dict, trial: Trial) -> None:
        bridge = self._bridge(config)
        bridge["trial_id"] = str(trial.trial_id)
        bridge["trial_name"] = str(trial)

    def on_trial_add(
        self,
        tune_controller: "TuneController",
        trial: Trial,
    ):
        filled = _fill_config(trial.config, self._hyperparam_bounds)
        trial.config[self.entropy_param] = float(self.base_entropy_coeff)
        trial.evaluated_params.update(flatten_dict(filled))
        self._set_trial_identity(trial.config, trial)

        bridge = self._bridge(trial.config)
        bridge.setdefault("lineage_generation", 0)
        bridge.setdefault("lineage_ema_seed", None)
        bridge.setdefault("guided_update", False)
        bridge.setdefault("gp_data_count", 0)
        bridge.setdefault("o2i_uncertainty_raw_std", 0.0)
        bridge.setdefault("o2i_candidate_posterior_std", 0.0)
        bridge.setdefault("o2i_donor_uncertainty_raw_std", 0.0)
        bridge.setdefault("o2i_donor_posterior_std", 0.0)
        bridge.setdefault("o2i_uncertainty_prior_std", 0.0)
        bridge.setdefault("o2i_candidate_prior_std", 0.0)
        bridge.setdefault("o2i_donor_uncertainty_prior_std", 0.0)
        bridge.setdefault("o2i_donor_prior_std", 0.0)
        bridge.setdefault("o2i_uncertainty_normalized", 0.0)
        bridge.setdefault("o2i_incremental_uncertainty", 0.0)
        bridge.setdefault("o2i_kernel_variance", 0.0)
        bridge.setdefault("o2i_donor_kernel_variance", 0.0)
        bridge.setdefault("gp_kernel_variance_parameter", None)
        bridge.setdefault("gp_lengthscale", None)
        bridge.setdefault("gp_temporal_epsilon", None)
        bridge.setdefault("gp_alpha_used", None)
        bridge.setdefault("o2i_acquisition_value", None)
        bridge.setdefault(
            "o2i_reference_entropy_coeff", float(self.base_entropy_coeff)
        )
        bridge.setdefault("entropy_increment", 0.0)
        bridge.setdefault("base_entropy_coeff", float(self.base_entropy_coeff))
        bridge.setdefault("nominal_entropy_coeff", float(self.base_entropy_coeff))
        bridge.setdefault("applied_entropy_coeff", float(self.base_entropy_coeff))

        super().on_trial_add(tune_controller, trial)

    def _save_trial_state(
        self,
        state: _PBTTrialState,
        time: int,
        result: Dict,
        trial: Trial,
    ):
        score = super()._save_trial_state(state, time, result, trial)
        custom = result.get("custom_metrics", {}) or {}

        def metric(name: str, default=np.nan) -> float:
            candidates = (
                custom.get(name),
                result.get(f"custom_metrics/{name}"),
                result.get(name),
            )
            for candidate in candidates:
                value = _safe_float(candidate, np.nan)
                if np.isfinite(value):
                    return value
            return float(default)

        policy_state = metric("policy_update_state")
        policy_state_raw = metric("policy_update_state_raw")
        policy_state_valid = metric("policy_update_state_valid", 0.0)

        flat = flatten_dict(trial.config, prevent_delimiter=True)
        hp_names = list(self._hyperparam_bounds_flat.keys())
        hp_values = [flat[name] for name in hp_names]
        applied_entropy = _safe_float(
            flat.get(self.entropy_param, self.base_entropy_coeff),
            self.base_entropy_coeff,
        )

        bridge = (
            trial.config.get("model", {})
            .get("custom_model_config", {})
            .get("_coda_bridge", {})
        )

        row = {
            "Trial": _trial_key(trial),
            "Time": result[self._time_attr],
            **dict(zip(hp_names, hp_values)),
            self.entropy_param: applied_entropy,
            "Reward": score,
            "policy_update_state": policy_state,
            "policy_update_state_raw": policy_state_raw,
            "policy_update_state_valid": policy_state_valid,
            "is_synthetic": False,
            "guided_update": bool(bridge.get("guided_update", False)),
            "gp_data_count": _safe_float(bridge.get("gp_data_count", 0), 0.0),
            "o2i_uncertainty_raw_std": _safe_float(
                bridge.get("o2i_uncertainty_raw_std", 0.0), 0.0
            ),
            "o2i_candidate_posterior_std": _safe_float(
                bridge.get("o2i_candidate_posterior_std", 0.0), 0.0
            ),
            "o2i_donor_uncertainty_raw_std": _safe_float(
                bridge.get("o2i_donor_uncertainty_raw_std", 0.0), 0.0
            ),
            "o2i_donor_posterior_std": _safe_float(
                bridge.get("o2i_donor_posterior_std", 0.0), 0.0
            ),
            "o2i_uncertainty_prior_std": _safe_float(
                bridge.get("o2i_uncertainty_prior_std", 0.0), 0.0
            ),
            "o2i_candidate_prior_std": _safe_float(
                bridge.get("o2i_candidate_prior_std", 0.0), 0.0
            ),
            "o2i_donor_uncertainty_prior_std": _safe_float(
                bridge.get("o2i_donor_uncertainty_prior_std", 0.0), 0.0
            ),
            "o2i_donor_prior_std": _safe_float(
                bridge.get("o2i_donor_prior_std", 0.0), 0.0
            ),
            "o2i_uncertainty_normalized": _safe_float(
                bridge.get("o2i_uncertainty_normalized", 0.0), 0.0
            ),
            "o2i_incremental_uncertainty": _safe_float(
                bridge.get("o2i_incremental_uncertainty", 0.0), 0.0
            ),
            "o2i_kernel_variance": _safe_float(
                bridge.get("o2i_kernel_variance", 0.0), 0.0
            ),
            "o2i_donor_kernel_variance": _safe_float(
                bridge.get("o2i_donor_kernel_variance", 0.0), 0.0
            ),
            "gp_kernel_variance_parameter": _safe_float(
                bridge.get("gp_kernel_variance_parameter", np.nan), np.nan
            ),
            "gp_lengthscale": _safe_float(
                bridge.get("gp_lengthscale", np.nan), np.nan
            ),
            "gp_temporal_epsilon": _safe_float(
                bridge.get("gp_temporal_epsilon", np.nan), np.nan
            ),
            "gp_alpha_used": _safe_float(
                bridge.get("gp_alpha_used", np.nan), np.nan
            ),
            "o2i_acquisition_value": _safe_float(
                bridge.get("o2i_acquisition_value", np.nan), np.nan
            ),
            "o2i_reference_entropy_coeff": _safe_float(
                bridge.get(
                    "o2i_reference_entropy_coeff", self.base_entropy_coeff
                ),
                self.base_entropy_coeff,
            ),
            "entropy_increment": _safe_float(
                bridge.get("entropy_increment", 0.0), 0.0
            ),
            "base_entropy_coeff": _safe_float(
                bridge.get("base_entropy_coeff", self.base_entropy_coeff),
                self.base_entropy_coeff,
            ),
            "nominal_entropy_coeff": _safe_float(
                bridge.get("nominal_entropy_coeff", self.base_entropy_coeff),
                self.base_entropy_coeff,
            ),
            "applied_entropy_coeff": _safe_float(
                bridge.get("applied_entropy_coeff", np.nan), np.nan
            ),
            "lineage_generation": bridge.get("lineage_generation", 0),
        }

        self.data = pd.concat(
            [self.data, pd.DataFrame([row])], ignore_index=True
        )
        return score

    def _latest_trial_row(self, trial: Trial) -> Optional[pd.Series]:
        """Return the latest real observation in causal insertion order."""
        if self.data.empty:
            return None

        rows = self.data[
            self.data["Trial"].eq(_trial_key(trial))
        ].copy()
        if "is_synthetic" in rows.columns:
            rows = rows[
                ~rows["is_synthetic"].fillna(False).astype(bool)
            ]
        if rows.empty:
            return None

        rows = rows.reset_index(drop=False).rename(
            columns={"index": "_insertion_order"}
        )
        rows = rows.sort_values(
            ["_insertion_order"], kind="mergesort"
        )
        return rows.iloc[-1]

    def _append_lineage_anchor(
        self,
        *,
        donor_row: pd.Series,
        receiver: Trial,
    ) -> None:
        """Insert the donor state as the receiver's synthetic predecessor."""
        hp_names = list(self._hyperparam_bounds_flat.keys())
        row = {
            "Trial": _trial_key(receiver),
            "Time": donor_row["Time"],
            "Reward": donor_row["Reward"],
            "policy_update_state": donor_row.get(
                "policy_update_state", np.nan
            ),
            "policy_update_state_raw": donor_row.get(
                "policy_update_state_raw", np.nan
            ),
            "policy_update_state_valid": donor_row.get(
                "policy_update_state_valid", 0.0
            ),
            "is_synthetic": True,
            "guided_update": False,
            "gp_data_count": 0.0,
            "o2i_uncertainty_raw_std": 0.0,
            "o2i_candidate_posterior_std": 0.0,
            "o2i_donor_uncertainty_raw_std": 0.0,
            "o2i_donor_posterior_std": 0.0,
            "o2i_uncertainty_prior_std": 0.0,
            "o2i_candidate_prior_std": 0.0,
            "o2i_donor_uncertainty_prior_std": 0.0,
            "o2i_donor_prior_std": 0.0,
            "o2i_uncertainty_normalized": 0.0,
            "o2i_incremental_uncertainty": 0.0,
            "o2i_kernel_variance": 0.0,
            "o2i_donor_kernel_variance": 0.0,
            "gp_kernel_variance_parameter": np.nan,
            "gp_lengthscale": np.nan,
            "gp_temporal_epsilon": np.nan,
            "gp_alpha_used": np.nan,
            "o2i_acquisition_value": np.nan,
            "o2i_reference_entropy_coeff": _safe_float(
                donor_row.get(self.entropy_param, self.base_entropy_coeff),
                self.base_entropy_coeff,
            ),
            "entropy_increment": 0.0,
            "base_entropy_coeff": float(self.base_entropy_coeff),
            "nominal_entropy_coeff": float(self.base_entropy_coeff),
            "applied_entropy_coeff": _safe_float(
                donor_row.get(self.entropy_param, self.base_entropy_coeff),
                self.base_entropy_coeff,
            ),
            "lineage_generation": self._lineage_generation + 1,
        }

        for name in hp_names:
            row[name] = donor_row[name]
        row[self.entropy_param] = _safe_float(
            donor_row.get(self.entropy_param, self.base_entropy_coeff),
            self.base_entropy_coeff,
        )

        self.data = pd.concat(
            [self.data, pd.DataFrame([row])], ignore_index=True
        )

    def _new_context(self, donor_row: pd.Series) -> np.ndarray:
        values = [donor_row["Time"]]
        if self.use_learner_context:
            values += [
                donor_row["Reward"],
                donor_row["policy_update_state"],
            ]
        return np.asarray(values, dtype=np.float64)

    def _propose_from_gp(
        self,
        donor: Trial,
        donor_config_flat: Dict,
    ) -> Tuple[Dict, bool, _ProposalDiagnostics]:
        """Return a proposal and donor-referenced GP diagnostics."""
        transitions, context_columns = _prepare_gp_transitions(
            self.data,
            self._hyperparam_bounds_flat,
            self._execution_feature_bounds_flat,
            use_learner_context=self.use_learner_context,
            max_gp_points=self.max_gp_points,
        )

        hp_names = list(self._hyperparam_bounds_flat.keys())
        donor_row = self._latest_trial_row(donor)
        n_data = int(len(transitions))

        reference_entropy = float(self.base_entropy_coeff)
        if donor_row is not None:
            reference_entropy = _safe_float(
                donor_row.get(self.entropy_param, self.base_entropy_coeff),
                self.base_entropy_coeff,
            )
        reference_entropy = float(
            np.clip(reference_entropy, 0.0, self.entropy_guard)
        )

        fallback_diagnostics = _ProposalDiagnostics(
            uncertainty_raw_std=0.0,
            donor_uncertainty_raw_std=0.0,
            uncertainty_prior_std=0.0,
            donor_uncertainty_prior_std=0.0,
            uncertainty_normalized=0.0,
            kernel_prior_variance=0.0,
            donor_kernel_prior_variance=0.0,
            acquisition_value=np.nan,
            reference_entropy_coeff=reference_entropy,
            gp_kernel_variance_parameter=np.nan,
            gp_lengthscale=np.nan,
            gp_temporal_epsilon=np.nan,
            gp_alpha_used=np.nan,
            gp_data_count=n_data,
        )
        fallback = (
            deepcopy(donor_config_flat),
            False,
            fallback_diagnostics,
        )

        if donor_row is None:
            return fallback
        if n_data < self.min_valid_transitions:
            return fallback

        new_context = self._new_context(donor_row)
        if not np.all(np.isfinite(new_context)):
            return fallback

        gp_inputs = context_columns + hp_names + [self.entropy_param]
        Xraw = transitions[gp_inputs].to_numpy(dtype=np.float64)
        yraw = transitions["y"].to_numpy(dtype=np.float64)

        reference_hyperparams = {
            name: _safe_float(
                donor_row.get(name, donor_config_flat[name]),
                donor_config_flat[name],
            )
            for name in hp_names
        }

        new_values, decision_state, acquisition_value = _select_config(
            Xraw=Xraw,
            yraw=yraw,
            current_nominal=self.current_nominal,
            new_context=new_context,
            reference_hyperparams=reference_hyperparams,
            search_bounds=self._hyperparam_bounds_flat,
            model_bounds=self._model_bounds_flat,
            execution_feature_bounds=self._execution_feature_bounds_flat,
            context_columns=context_columns,
            context_bounds=self._context_bounds,
            use_o2i_feedback=self.use_o2i_feedback,
            entropy_param=self.entropy_param,
            base_entropy_coeff=self.base_entropy_coeff,
            reference_entropy_coeff=reference_entropy,
            o2i_uncertainty_scale=self.o2i_uncertainty_scale,
            max_entropy_increment=self.max_entropy_increment,
            entropy_guard=self.entropy_guard,
            reward_z_clip=self.reward_z_clip,
        )

        new_flat = deepcopy(donor_config_flat)
        for idx, name in enumerate(hp_names):
            template = donor_config_flat[name]
            new_flat[name] = _cast_and_clip(
                new_values[idx],
                template,
                self._hyperparam_bounds_flat[name],
            )

        if self.use_o2i_feedback:
            (
                raw_std,
                donor_raw_std,
                prior_std,
                donor_prior_std,
                uncertainty,
                kernel_prior_variance,
                donor_kernel_prior_variance,
            ) = _uncertainty_for_executable_config(
                decision_state,
                {name: new_flat[name] for name in hp_names},
            )
        else:
            raw_std = 0.0
            donor_raw_std = 0.0
            prior_std = 0.0
            donor_prior_std = 0.0
            uncertainty = 0.0
            kernel_prior_variance = 0.0
            donor_kernel_prior_variance = 0.0

        diagnostics = _ProposalDiagnostics(
            uncertainty_raw_std=float(raw_std),
            donor_uncertainty_raw_std=float(donor_raw_std),
            uncertainty_prior_std=float(prior_std),
            donor_uncertainty_prior_std=float(donor_prior_std),
            uncertainty_normalized=float(
                np.clip(uncertainty, 0.0, 1.0)
            ),
            kernel_prior_variance=float(kernel_prior_variance),
            donor_kernel_prior_variance=float(
                donor_kernel_prior_variance
            ),
            acquisition_value=float(acquisition_value),
            reference_entropy_coeff=float(
                decision_state.reference_entropy_coeff
            ),
            gp_kernel_variance_parameter=float(
                decision_state.gp_kernel_variance_parameter
            ),
            gp_lengthscale=float(decision_state.gp_lengthscale),
            gp_temporal_epsilon=float(
                decision_state.gp_temporal_epsilon
            ),
            gp_alpha_used=float(decision_state.gp_alpha_used),
            gp_data_count=n_data,
        )
        return new_flat, True, diagnostics

    def _apply_o2i_feedback(
        self,
        config: Dict,
        *,
        diagnostics: _ProposalDiagnostics,
        guided_update: bool,
    ) -> Dict:
        """Map donor-relative incremental completed-GP uncertainty to entropy."""
        out = deepcopy(config)
        bridge = self._bridge(out)

        bridge["guided_update"] = bool(guided_update)
        bridge["gp_data_count"] = int(diagnostics.gp_data_count)
        bridge["o2i_uncertainty_raw_std"] = float(
            diagnostics.uncertainty_raw_std if guided_update else 0.0
        )
        bridge["o2i_candidate_posterior_std"] = float(
            diagnostics.uncertainty_raw_std if guided_update else 0.0
        )
        bridge["o2i_donor_uncertainty_raw_std"] = float(
            diagnostics.donor_uncertainty_raw_std if guided_update else 0.0
        )
        bridge["o2i_donor_posterior_std"] = float(
            diagnostics.donor_uncertainty_raw_std if guided_update else 0.0
        )
        bridge["o2i_uncertainty_prior_std"] = float(
            diagnostics.uncertainty_prior_std if guided_update else 0.0
        )
        bridge["o2i_candidate_prior_std"] = float(
            diagnostics.uncertainty_prior_std if guided_update else 0.0
        )
        bridge["o2i_donor_uncertainty_prior_std"] = float(
            diagnostics.donor_uncertainty_prior_std if guided_update else 0.0
        )
        bridge["o2i_donor_prior_std"] = float(
            diagnostics.donor_uncertainty_prior_std if guided_update else 0.0
        )
        bridge["o2i_uncertainty_normalized"] = float(
            diagnostics.uncertainty_normalized if guided_update else 0.0
        )
        bridge["o2i_incremental_uncertainty"] = float(
            diagnostics.uncertainty_normalized if guided_update else 0.0
        )
        bridge["o2i_kernel_variance"] = float(
            diagnostics.kernel_prior_variance if guided_update else 0.0
        )
        bridge["o2i_donor_kernel_variance"] = float(
            diagnostics.donor_kernel_prior_variance
            if guided_update
            else 0.0
        )
        bridge["gp_kernel_variance_parameter"] = (
            float(diagnostics.gp_kernel_variance_parameter)
            if guided_update
            and np.isfinite(diagnostics.gp_kernel_variance_parameter)
            else None
        )
        bridge["gp_lengthscale"] = (
            float(diagnostics.gp_lengthscale)
            if guided_update and np.isfinite(diagnostics.gp_lengthscale)
            else None
        )
        bridge["gp_temporal_epsilon"] = (
            float(diagnostics.gp_temporal_epsilon)
            if guided_update and np.isfinite(diagnostics.gp_temporal_epsilon)
            else None
        )
        bridge["gp_alpha_used"] = (
            float(diagnostics.gp_alpha_used)
            if guided_update and np.isfinite(diagnostics.gp_alpha_used)
            else None
        )
        bridge["o2i_acquisition_value"] = (
            float(diagnostics.acquisition_value)
            if guided_update and np.isfinite(diagnostics.acquisition_value)
            else None
        )

        base_entropy = float(self.base_entropy_coeff)
        bridge["o2i_reference_entropy_coeff"] = float(
            diagnostics.reference_entropy_coeff
        )
        bridge["base_entropy_coeff"] = base_entropy
        bridge["nominal_entropy_coeff"] = base_entropy

        if guided_update and self.use_o2i_feedback:
            increment = float(
                self.o2i_uncertainty_scale
                * np.clip(diagnostics.uncertainty_normalized, 0.0, 1.0)
            )
            increment = float(min(self.max_entropy_increment, increment))
        else:
            increment = 0.0

        applied_entropy = float(
            np.clip(
                base_entropy + increment,
                0.0,
                self.entropy_guard,
            )
        )
        out[self.entropy_param] = applied_entropy

        bridge["entropy_increment"] = increment
        bridge["applied_entropy_coeff"] = applied_entropy
        return out

    def _set_lineage_bridge(
        self,
        config: Dict,
        *,
        receiver: Trial,
        donor_row: Optional[pd.Series],
    ) -> None:
        self._lineage_generation += 1
        bridge = self._bridge(config)
        bridge["trial_id"] = str(receiver.trial_id)
        bridge["trial_name"] = str(receiver)
        bridge["lineage_generation"] = int(self._lineage_generation)

        seed = np.nan
        if donor_row is not None:
            valid = (
                _safe_float(
                    donor_row.get("policy_update_state_valid", 0.0), 0.0
                )
                >= 0.5
            )
            if valid:
                seed = _safe_float(
                    donor_row.get("policy_update_state", np.nan), np.nan
                )
        bridge["lineage_ema_seed"] = (
            float(seed) if np.isfinite(seed) else None
        )

    def _update_pending_nominal(
        self,
        applied_flat: Dict,
        guided_update: bool,
    ) -> None:
        """Store full execution-space proposals for batch diversification."""
        if not guided_update:
            return

        model_names = list(self._hyperparam_bounds_flat.keys()) + [
            self.entropy_param
        ]
        vector = np.asarray(
            [applied_flat[name] for name in model_names],
            dtype=np.float64,
        ).reshape(1, -1)

        current_time = (
            float(self.data["Time"].max()) if not self.data.empty else -np.inf
        )

        if current_time > self.last_exploration_time:
            self.last_exploration_time = current_time
            self.current_nominal = vector.copy()
        elif self.current_nominal is None:
            self.current_nominal = vector.copy()
        else:
            self.current_nominal = np.vstack((self.current_nominal, vector))

    def _get_new_config(
        self,
        trial: Trial,
        trial_to_clone: Trial,
    ) -> Tuple[Dict, Dict]:
        donor_row = self._latest_trial_row(trial_to_clone)

        if donor_row is not None:
            self._append_lineage_anchor(
                donor_row=donor_row,
                receiver=trial,
            )

        donor_flat = flatten_dict(
            trial_to_clone.config,
            prevent_delimiter=True,
        )

        current_time = (
            float(self.data["Time"].max())
            if not self.data.empty
            else -np.inf
        )
        if current_time > self.last_exploration_time:
            self.current_nominal = None

        nominal_flat, guided, diagnostics = self._propose_from_gp(
            trial_to_clone,
            donor_flat,
        )

        new_config = unflatten_dict(nominal_flat)
        new_config = self._apply_o2i_feedback(
            new_config,
            diagnostics=diagnostics,
            guided_update=guided,
        )

        applied_flat = flatten_dict(new_config, prevent_delimiter=True)
        self._update_pending_nominal(applied_flat, guided)

        self._set_lineage_bridge(
            new_config,
            receiver=trial,
            donor_row=donor_row,
        )

        return new_config, {}


# Convenient aliases for experiment scripts and older imports.
CODADonorRelativeIncrementalScheduler = CODAPPOOptimizer
CODADonorReferenceScheduler = CODAPPOOptimizer
CODAScheduler = CODAPPOOptimizer
W_PB2 = CODAPPOOptimizer


# Public API
CODAScheduler = CODAPPOOptimizer
CODAPPOOptimizerScheduler = CODAPPOOptimizer

__all__ = [
    "CODAPPOOptimizer",
    "CODAScheduler",
    "CODAPPOOptimizerScheduler",
]
