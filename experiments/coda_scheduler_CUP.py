#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CODA scheduler for Ray Tune / RLlib.

Experimental variants
---------------------
- ``full``: CODA (I2O contextual surrogate + O2I uncertainty feedback)
- ``i2o``:  CODA-I2O (contextual surrogate only)
- ``o2i``:  CODA-O2I (PB2-style surrogate + O2I uncertainty feedback)

Final O2I design
----------------
CODA uses a Contextual Uncertainty Percentile (CUP). After PB2 selects a
configuration, the observation-based GP evaluates its posterior standard
deviation and compares it with posterior standard deviations at a fixed set of
reference hyperparameter configurations under exactly the same current context.
The resulting empirical mid-rank percentile lies in [0, 1] and directly scales
the PPO entropy increment.

Design goals
------------
1. Keep population exploitation/checkpoint inheritance from Ray PBT/PB2.
2. Build temporally aligned GP transitions using the *applied* configuration.
3. Keep synthetic donor records out of GP training while using them as the
   predecessor of the receiver's first post-exploitation observation.
4. Normalize reward context causally with a task-agnostic robust transform.
5. Distinguish the outer PB2 search domain from the GP model domain.
6. Preserve the inherited configuration exactly when no surrogate-guided
   proposal can be produced.
7. Seed the receiver's learner-state EMA from the donor state after cloning.
8. Use the batch-adjusted GP only for PB2 candidate diversification; use the
   observation-based GP for CUP/O2I uncertainty.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import _PBTTrialState
from ray.tune.utils.util import flatten_dict, unflatten_dict

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
    """Return the stable Tune trial identifier used in CODA's internal table."""
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
    reward_z_clip: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize CODA context coordinates using only pre-decision data."""
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


def _contextual_uncertainty_percentile(
    selected_std: float,
    reference_std: np.ndarray,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> float:
    """Empirical mid-rank percentile of selected posterior uncertainty.

    Ties receive half weight. Therefore, if all posterior standard deviations
    are numerically identical, CUP is approximately 0.5 rather than 1.0.
    """
    selected_std = float(selected_std)
    ref = np.asarray(reference_std, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]

    if not np.isfinite(selected_std) or ref.size == 0:
        return 0.0

    equal = np.isclose(ref, selected_std, rtol=rtol, atol=atol)
    strictly_lower = (ref < selected_std) & (~equal)

    percentile = (
        np.sum(strictly_lower) + 0.5 * np.sum(equal)
    ) / float(ref.size)

    return float(np.clip(percentile, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Transition construction
# -----------------------------------------------------------------------------

def _prepare_gp_transitions(
    data: pd.DataFrame,
    hyperparam_bounds: Dict[str, Tuple[float, float]],
    use_learner_context: bool,
    max_gp_points: int = 1000,
) -> Tuple[pd.DataFrame, list[str]]:
    """Construct valid temporally aligned transitions for the GP."""
    if data.empty:
        return pd.DataFrame(), []

    hp_names = list(hyperparam_bounds.keys())
    df = data.reset_index(drop=False).rename(columns={"index": "_insertion_order"})
    df = df.copy()

    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)

    # Causal order is insertion order, not Time. Checkpoint inheritance may
    # restore a smaller training counter than one previously seen by a receiver.
    df = df.sort_values(
        ["Trial", "_insertion_order"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = df.groupby("Trial", sort=False, dropna=False)

    df["T_before"] = grouped["Time"].shift(1)
    df["R_before"] = grouped["Reward"].shift(1)
    df["S_before"] = grouped["stability_index"].shift(1)
    df["S_valid_before"] = grouped["stability_valid"].shift(1)

    df["time_change"] = df["Time"] - df["T_before"]
    df["reward_change"] = df["Reward"] - df["R_before"]
    df["y"] = df["reward_change"] / df["time_change"]

    context_columns = ["T_before"]
    if use_learner_context:
        context_columns += ["R_before", "S_before"]

    gp_input_columns = context_columns + hp_names
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
        valid &= df["S_valid_before"].fillna(0.0).astype(float) >= 0.5

    transitions = (
        df.loc[valid]
        .sort_values(["_insertion_order"], kind="mergesort")
        .reset_index(drop=True)
    )

    if max_gp_points and len(transitions) > max_gp_points:
        transitions = transitions.iloc[-max_gp_points:].reset_index(drop=True)

    return transitions, context_columns


# -----------------------------------------------------------------------------
# GP + acquisition
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


def _ucb_value(
    mean_model: GaussianProcessRegressor,
    variance_model: GaussianProcessRegressor,
    fixed_context: np.ndarray,
    hp_normalized: np.ndarray,
) -> float:
    """Ray-PB2-compatible UCB: posterior mean + kappa * posterior variance."""
    x = np.concatenate([fixed_context, hp_normalized]).reshape(1, -1)
    mean = float(mean_model.predict(x).reshape(-1)[0])
    _, std = variance_model.predict(x, return_std=True)
    variance = float(std.reshape(-1)[0] ** 2)

    n_obs = int(getattr(mean_model, "X_train_", np.empty((0,))).shape[0])
    beta_t = 0.2 + max(0.0, np.log(0.4 * max(n_obs, 1)))
    kappa_t = float(np.sqrt(beta_t))
    return mean + kappa_t * variance


def _optimize_acquisition(
    mean_model: GaussianProcessRegressor,
    variance_model: GaussianProcessRegressor,
    fixed_context: np.ndarray,
    normalized_search_bounds: Sequence[Tuple[float, float]],
    *,
    n_restarts: int = 10,
) -> np.ndarray:
    """Optimize PB2 UCB with per-coordinate normalized search bounds."""
    bounds = [(float(lo), float(hi)) for lo, hi in normalized_search_bounds]
    best_value = -np.inf
    best_theta: Optional[np.ndarray] = None

    for _ in range(n_restarts):
        x0 = np.asarray([np.random.uniform(lo, hi) for lo, hi in bounds])
        result = minimize(
            lambda x: -_ucb_value(
                mean_model, variance_model, fixed_context, x
            ),
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        value = _ucb_value(
            mean_model, variance_model, fixed_context, theta
        )
        if value > best_value:
            best_value = value
            best_theta = theta

    if best_theta is None:
        raise RuntimeError(
            "CODA acquisition optimization failed to produce a candidate"
        )
    return best_theta


def _select_config(
    *,
    Xraw: np.ndarray,
    yraw: np.ndarray,
    current_nominal: Optional[np.ndarray],
    new_context: np.ndarray,
    search_bounds: Dict[str, Tuple[float, float]],
    model_bounds: Dict[str, Tuple[float, float]],
    context_columns: Sequence[str],
    context_bounds: Dict[str, Tuple[float, float]],
    reward_z_clip: float = 4.0,
    reference_unit: Optional[np.ndarray] = None,
    compute_o2i_uncertainty: bool = True,
) -> Tuple[np.ndarray, float, float, float, float, float]:
    """Fit TV-GPs and return PB2 proposal plus CUP audit quantities."""
    search_vals = np.asarray(list(search_bounds.values()), dtype=np.float64).T
    model_vals = np.asarray(list(model_bounds.values()), dtype=np.float64).T

    if search_vals.shape != model_vals.shape:
        raise ValueError(
            "search_bounds and model_bounds must have identical coordinates"
        )

    n_context = len(context_columns)
    if Xraw.ndim != 2 or Xraw.shape[1] != n_context + model_vals.shape[1]:
        raise ValueError(
            "Xraw shape is inconsistent with context/hyperparameter dimensions"
        )

    X_context_raw = Xraw[:, :n_context]
    X_hyper_raw = Xraw[:, n_context:]

    X_context, fixed = _normalize_context(
        X_context_raw,
        new_context,
        context_columns,
        context_bounds,
        reward_z_clip=reward_z_clip,
    )
    X_hyper = _normalize(X_hyper_raw, model_vals)
    X = np.hstack((X_context, X_hyper))
    y = _standardize_response(yraw).reshape(-1, 1)

    # Observation-based GP: real observed transitions only.
    mean_model = _fit_gp(X, y)

    # PB2 batch adjustment: pending nominal candidates reduce uncertainty around
    # configurations already proposed during the same population event.
    if current_nominal is None or len(current_nominal) == 0:
        variance_model = deepcopy(mean_model)
    else:
        current_norm = _normalize(current_nominal, model_vals)
        padding = np.tile(fixed, (current_norm.shape[0], 1))
        X_pending = np.hstack((padding, current_norm))
        X_aug = np.vstack((X, X_pending))
        y_aug = np.vstack((y, np.zeros((current_norm.shape[0], 1))))
        variance_model = _fit_gp(X_aug, y_aug)

    # Acquisition candidates remain inside H_search, even though the GP can
    # observe applied entropy values outside the nominal PB2 interval.
    normalized_search_bounds = []
    for j in range(search_vals.shape[1]):
        model_lo, model_hi = model_vals[:, j]
        search_lo, search_hi = search_vals[:, j]
        denom = model_hi - model_lo
        normalized_search_bounds.append(
            (
                (search_lo - model_lo) / denom,
                (search_hi - model_lo) / denom,
            )
        )

    xt_norm = _optimize_acquisition(
        mean_model,
        variance_model,
        fixed,
        normalized_search_bounds,
        n_restarts=10,
    )

    # -------------------------------------------------------------------------
    # O2I: Contextual Uncertainty Percentile (CUP)
    # -------------------------------------------------------------------------
    if compute_o2i_uncertainty:
        if reference_unit is None:
            raise ValueError(
                "reference_unit is required when compute_o2i_uncertainty=True"
            )

        reference_unit = np.asarray(reference_unit, dtype=np.float64)
        if (
            reference_unit.ndim != 2
            or reference_unit.shape[1] != len(normalized_search_bounds)
        ):
            raise ValueError(
                "reference_unit has inconsistent dimensions"
            )

        # Selected PB2 candidate uncertainty from observation-based GP.
        x_selected = np.concatenate((fixed, xt_norm)).reshape(1, -1)
        _, selected_std = mean_model.predict(
            x_selected,
            return_std=True,
        )
        uncertainty_raw = float(selected_std.reshape(-1)[0])

        # Map the fixed unit reference design into the current normalized PB2
        # search domain. This correctly handles the larger entropy *model* domain.
        ref_lower = np.asarray(
            [bounds[0] for bounds in normalized_search_bounds],
            dtype=np.float64,
        )
        ref_upper = np.asarray(
            [bounds[1] for bounds in normalized_search_bounds],
            dtype=np.float64,
        )
        reference_hp_norm = (
            ref_lower + reference_unit * (ref_upper - ref_lower)
        )

        # Evaluate all references under exactly the same current context.
        M = reference_hp_norm.shape[0]
        reference_context = np.tile(fixed, (M, 1))
        X_reference = np.hstack((reference_context, reference_hp_norm))

        _, reference_std = mean_model.predict(
            X_reference,
            return_std=True,
        )
        reference_std = np.asarray(
            reference_std, dtype=np.float64
        ).reshape(-1)

        uncertainty = _contextual_uncertainty_percentile(
            uncertainty_raw,
            reference_std,
        )

        reference_uncertainty_median = float(np.median(reference_std))
        q25, q75 = np.percentile(reference_std, [25.0, 75.0])
        reference_uncertainty_iqr = float(q75 - q25)

        # Audit only; no longer used to normalize the O2I signal.
        kernel_variance = float(mean_model.kernel_.variance)

    else:
        # CODA-I2O does not pay the additional O2I prediction cost.
        uncertainty = 0.0
        uncertainty_raw = 0.0
        kernel_variance = np.nan
        reference_uncertainty_median = np.nan
        reference_uncertainty_iqr = np.nan

    xt = _denormalize(xt_norm, model_vals).astype(np.float64)

    return (
        xt,
        float(uncertainty),
        float(uncertainty_raw),
        float(kernel_variance) if np.isfinite(kernel_variance) else np.nan,
        float(reference_uncertainty_median)
        if np.isfinite(reference_uncertainty_median)
        else np.nan,
        float(reference_uncertainty_iqr)
        if np.isfinite(reference_uncertainty_iqr)
        else np.nan,
    )


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------

class CODAScheduler(PopulationBasedTraining):
    """Closed-Loop Online Diagnostic-Aware AutoRL scheduler."""

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
        quantile_fraction: float = 0.25,
        variant: str = "full",
        min_valid_transitions: int = 2,
        max_gp_points: int = 1000,
        entropy_param: str = "entropy_coeff",
        uncertainty_scale: float = 0.005,
        max_entropy_increment: float = 0.005,
        entropy_guard: float = 0.05,
        reward_z_clip: float = 4.0,
        uncertainty_reference_size: int = 64,
        uncertainty_reference_seed: int = 2026,
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
        self.use_uncertainty_feedback = variant in {"full", "o2i"}
        self.min_valid_transitions = int(min_valid_transitions)
        self.max_gp_points = int(max_gp_points)
        self.entropy_param = entropy_param
        self.uncertainty_scale = float(uncertainty_scale)
        self.max_entropy_increment = float(max_entropy_increment)
        self.entropy_guard = float(entropy_guard)
        self.reward_z_clip = float(reward_z_clip)
        self.uncertainty_reference_size = int(uncertainty_reference_size)
        self.uncertainty_reference_seed = int(uncertainty_reference_seed)

        if self.min_valid_transitions < 2:
            raise ValueError("min_valid_transitions must be >= 2")
        if self.reward_z_clip <= 0:
            raise ValueError("reward_z_clip must be > 0")
        if self.uncertainty_reference_size < 2:
            raise ValueError("uncertainty_reference_size must be >= 2")
        if self.uncertainty_scale < 0:
            raise ValueError("uncertainty_scale must be >= 0")
        if self.max_entropy_increment < 0:
            raise ValueError("max_entropy_increment must be >= 0")

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

        # Fixed deterministic reference design in the normalized unit
        # hyperparameter cube. A local RNG avoids modifying NumPy's global RNG
        # and therefore does not perturb PB2 acquisition randomness.
        reference_rng = np.random.default_rng(
            self.uncertainty_reference_seed
        )
        n_hyperparameters = len(self._hyperparam_bounds_flat)
        self._uncertainty_reference_unit = reference_rng.random(
            (
                self.uncertainty_reference_size,
                n_hyperparameters,
            )
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

        # GP model bounds describe applied configurations. In full/O2I the
        # applied entropy may exceed the nominal PB2 search maximum.
        self._model_bounds_flat = deepcopy(self._hyperparam_bounds_flat)
        if (
            self.use_uncertainty_feedback
            and self.entropy_param in self._model_bounds_flat
        ):
            ent_low, ent_high = self._hyperparam_bounds_flat[
                self.entropy_param
            ]
            effective_high = float(ent_high) + self.max_entropy_increment
            if effective_high > self.entropy_guard + 1e-12:
                raise ValueError(
                    "entropy_guard must be >= search upper bound + max increment"
                )
            self._model_bounds_flat[self.entropy_param] = [
                float(ent_low),
                effective_high,
            ]

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

    def _context_limits(self) -> np.ndarray:
        """Fixed context limits for debugging/backward compatibility."""
        names = [
            name
            for name in self._context_columns()
            if name != "R_before"
        ]
        return np.asarray(
            [self._context_bounds[name] for name in names],
            dtype=np.float64,
        ).T

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
        trial.evaluated_params.update(flatten_dict(filled))
        self._set_trial_identity(trial.config, trial)

        bridge = self._bridge(trial.config)
        bridge.setdefault("lineage_generation", 0)
        bridge.setdefault("lineage_ema_seed", None)
        bridge.setdefault("guided_update", False)
        bridge.setdefault("gp_uncertainty", 0.0)
        bridge.setdefault("gp_uncertainty_raw", 0.0)
        bridge.setdefault("gp_kernel_variance", np.nan)
        bridge.setdefault("gp_reference_uncertainty_median", np.nan)
        bridge.setdefault("gp_reference_uncertainty_iqr", np.nan)
        bridge.setdefault("entropy_increment", 0.0)
        bridge.setdefault("nominal_entropy_coeff", np.nan)
        bridge.setdefault("applied_entropy_coeff", np.nan)

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

        stability_index = metric("stability_index")
        stability_raw = metric("stability_raw")
        stability_valid = metric("stability_valid", 0.0)

        flat = flatten_dict(trial.config, prevent_delimiter=True)
        hp_names = list(self._hyperparam_bounds_flat.keys())
        hp_values = [flat[name] for name in hp_names]

        bridge = (
            trial.config.get("model", {})
            .get("custom_model_config", {})
            .get("_coda_bridge", {})
        )

        row = {
            "Trial": _trial_key(trial),
            "Time": result[self._time_attr],
            **dict(zip(hp_names, hp_values)),
            "Reward": score,
            "stability_index": stability_index,
            "stability_raw": stability_raw,
            "stability_valid": stability_valid,
            "is_synthetic": False,
            "guided_update": bool(bridge.get("guided_update", False)),
            "gp_uncertainty": _safe_float(
                bridge.get("gp_uncertainty", 0.0), 0.0
            ),
            "gp_uncertainty_raw": _safe_float(
                bridge.get("gp_uncertainty_raw", 0.0), 0.0
            ),
            "gp_kernel_variance": _safe_float(
                bridge.get("gp_kernel_variance", np.nan), np.nan
            ),
            "gp_reference_uncertainty_median": _safe_float(
                bridge.get("gp_reference_uncertainty_median", np.nan),
                np.nan,
            ),
            "gp_reference_uncertainty_iqr": _safe_float(
                bridge.get("gp_reference_uncertainty_iqr", np.nan),
                np.nan,
            ),
            "entropy_increment": _safe_float(
                bridge.get("entropy_increment", 0.0), 0.0
            ),
            "nominal_entropy_coeff": _safe_float(
                bridge.get("nominal_entropy_coeff", np.nan), np.nan
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
            "stability_index": donor_row.get("stability_index", np.nan),
            "stability_raw": donor_row.get("stability_raw", np.nan),
            "stability_valid": donor_row.get("stability_valid", 0.0),
            "is_synthetic": True,
            "guided_update": False,
            "gp_uncertainty": 0.0,
            "gp_uncertainty_raw": 0.0,
            "gp_kernel_variance": np.nan,
            "gp_reference_uncertainty_median": np.nan,
            "gp_reference_uncertainty_iqr": np.nan,
            "entropy_increment": 0.0,
            "nominal_entropy_coeff": np.nan,
            "applied_entropy_coeff": np.nan,
            "lineage_generation": self._lineage_generation + 1,
        }

        for name in hp_names:
            row[name] = donor_row[name]

        self.data = pd.concat(
            [self.data, pd.DataFrame([row])], ignore_index=True
        )

    def _new_context(self, donor_row: pd.Series) -> np.ndarray:
        values = [donor_row["Time"]]
        if self.use_learner_context:
            values += [
                donor_row["Reward"],
                donor_row["stability_index"],
            ]
        return np.asarray(values, dtype=np.float64)

    def _propose_from_gp(
        self,
        donor: Trial,
        donor_config_flat: Dict,
    ) -> Tuple[Dict, bool, float, float, float, float, float]:
        transitions, context_columns = _prepare_gp_transitions(
            self.data,
            self._hyperparam_bounds_flat,
            use_learner_context=self.use_learner_context,
            max_gp_points=self.max_gp_points,
        )

        donor_row = self._latest_trial_row(donor)
        fallback = (
            deepcopy(donor_config_flat),
            False,
            0.0,
            0.0,
            np.nan,
            np.nan,
            np.nan,
        )

        if donor_row is None:
            return fallback
        if len(transitions) < self.min_valid_transitions:
            return fallback

        new_context = self._new_context(donor_row)
        if not np.all(np.isfinite(new_context)):
            return fallback

        hp_names = list(self._hyperparam_bounds_flat.keys())
        gp_inputs = context_columns + hp_names
        Xraw = transitions[gp_inputs].to_numpy(dtype=np.float64)
        yraw = transitions["y"].to_numpy(dtype=np.float64)

        (
            new_values,
            uncertainty,
            uncertainty_raw,
            kernel_variance,
            reference_uncertainty_median,
            reference_uncertainty_iqr,
        ) = _select_config(
            Xraw=Xraw,
            yraw=yraw,
            current_nominal=self.current_nominal,
            new_context=new_context,
            search_bounds=self._hyperparam_bounds_flat,
            model_bounds=self._model_bounds_flat,
            context_columns=context_columns,
            context_bounds=self._context_bounds,
            reward_z_clip=self.reward_z_clip,
            reference_unit=self._uncertainty_reference_unit,
            compute_o2i_uncertainty=self.use_uncertainty_feedback,
        )

        new_flat = deepcopy(donor_config_flat)
        for idx, name in enumerate(hp_names):
            template = donor_config_flat[name]
            new_flat[name] = _cast_and_clip(
                new_values[idx],
                template,
                self._hyperparam_bounds_flat[name],
            )

        return (
            new_flat,
            True,
            float(max(0.0, uncertainty)),
            float(max(0.0, uncertainty_raw)),
            float(kernel_variance)
            if np.isfinite(kernel_variance)
            else np.nan,
            float(reference_uncertainty_median)
            if np.isfinite(reference_uncertainty_median)
            else np.nan,
            float(reference_uncertainty_iqr)
            if np.isfinite(reference_uncertainty_iqr)
            else np.nan,
        )

    def _apply_uncertainty_feedback(
        self,
        config: Dict,
        *,
        uncertainty: float,
        uncertainty_raw: float,
        kernel_variance: float,
        reference_uncertainty_median: float,
        reference_uncertainty_iqr: float,
        guided_update: bool,
    ) -> Dict:
        """Use CUP in [0,1] to modulate the applied entropy coefficient."""
        out = deepcopy(config)
        bridge = self._bridge(out)

        bridge["guided_update"] = bool(guided_update)
        bridge["gp_uncertainty"] = float(
            uncertainty if guided_update else 0.0
        )
        bridge["gp_uncertainty_raw"] = float(
            uncertainty_raw if guided_update else 0.0
        )
        bridge["gp_kernel_variance"] = (
            float(kernel_variance)
            if guided_update and np.isfinite(kernel_variance)
            else np.nan
        )
        bridge["gp_reference_uncertainty_median"] = (
            float(reference_uncertainty_median)
            if guided_update and np.isfinite(reference_uncertainty_median)
            else np.nan
        )
        bridge["gp_reference_uncertainty_iqr"] = (
            float(reference_uncertainty_iqr)
            if guided_update and np.isfinite(reference_uncertainty_iqr)
            else np.nan
        )

        if self.entropy_param not in out:
            bridge["entropy_increment"] = 0.0
            return out

        nominal_entropy = float(out[self.entropy_param])
        bridge["nominal_entropy_coeff"] = nominal_entropy

        if guided_update and self.use_uncertainty_feedback:
            search_low, search_high = self._hyperparam_bounds_flat[
                self.entropy_param
            ]
            base_entropy = float(
                np.clip(nominal_entropy, search_low, search_high)
            )

            # CUP directly scales the maximum intended O2I increment.
            increment = float(
                self.uncertainty_scale
                * np.clip(uncertainty, 0.0, 1.0)
            )
            increment = float(
                min(self.max_entropy_increment, increment)
            )

            applied_entropy = float(
                np.clip(
                    base_entropy + increment,
                    0.0,
                    self.entropy_guard,
                )
            )
            out[self.entropy_param] = applied_entropy
        else:
            increment = 0.0
            applied_entropy = float(out[self.entropy_param])

        bridge["entropy_increment"] = float(increment)
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
                    donor_row.get("stability_valid", 0.0), 0.0
                )
                >= 0.5
            )
            if valid:
                seed = _safe_float(
                    donor_row.get("stability_index", np.nan), np.nan
                )
        bridge["lineage_ema_seed"] = (
            float(seed) if np.isfinite(seed) else None
        )

    def _update_pending_nominal(
        self,
        new_flat: Dict,
        guided_update: bool,
    ) -> None:
        if not guided_update:
            return

        vector = np.asarray(
            [
                new_flat[name]
                for name in self._hyperparam_bounds_flat
            ],
            dtype=np.float64,
        ).reshape(1, -1)

        current_time = (
            float(self.data["Time"].max())
            if not self.data.empty
            else -np.inf
        )

        if current_time > self.last_exploration_time:
            self.last_exploration_time = current_time
            self.current_nominal = vector.copy()
        elif self.current_nominal is None:
            self.current_nominal = vector.copy()
        else:
            self.current_nominal = np.vstack(
                (self.current_nominal, vector)
            )

    def _get_new_config(
        self,
        trial: Trial,
        trial_to_clone: Trial,
    ) -> Tuple[Dict, Dict]:
        donor_row = self._latest_trial_row(trial_to_clone)

        # Always create the receiver's causal donor predecessor before checking
        # whether the GP is ready.
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

        (
            nominal_flat,
            guided,
            uncertainty,
            uncertainty_raw,
            kernel_variance,
            reference_uncertainty_median,
            reference_uncertainty_iqr,
        ) = self._propose_from_gp(
            trial_to_clone,
            donor_flat,
        )

        self._update_pending_nominal(
            nominal_flat,
            guided,
        )

        new_config = unflatten_dict(nominal_flat)
        new_config = self._apply_uncertainty_feedback(
            new_config,
            uncertainty=uncertainty,
            uncertainty_raw=uncertainty_raw,
            kernel_variance=kernel_variance,
            reference_uncertainty_median=reference_uncertainty_median,
            reference_uncertainty_iqr=reference_uncertainty_iqr,
            guided_update=guided,
        )

        self._set_lineage_bridge(
            new_config,
            receiver=trial,
            donor_row=donor_row,
        )

        return new_config, {}


# Backward-compatible alias if older scripts import W_PB2.
W_PB2 = CODAScheduler
