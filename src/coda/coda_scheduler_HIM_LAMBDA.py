#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CODA scheduler for Ray Tune / RLlib.

Experimental variants
---------------------
- ``full``: CODA (I2O contextual surrogate + O2I intervention-magnitude feedback)
- ``i2o``:  CODA-I2O (contextual surrogate only)
- ``o2i``:  CODA-O2I (PB2-style surrogate + O2I intervention-magnitude feedback)

O2I design: Hyperparameter Intervention Magnitude (HIM)
-------------------------------------------------------
The outer optimizer adapts four coordinates only:

    train_batch_size, lambda (GAE), lr, vf_loss_coeff.

The entropy coefficient is deliberately excluded from the search space and is
reserved as the O2I communication actuator.  After the contextual PB2 surrogate
proposes h_new, CODA measures the intervention relative to the donor's latest
applied outer-loop configuration h_current:

    intervention = mean_j |h_new_norm[j] - h_current_norm[j]|.

The signal is naturally bounded in [0, 1] and determines entropy directly:

    entropy_coeff = base_entropy_coeff + min(max_increment,
                                              o2i_entropy_scale * intervention).

With the default base_entropy_coeff=0, entropy is generated only by the O2I
channel.  Applied entropy is retained as an execution feature of the GP, but it
is never an independently optimized coordinate.

Design goals
------------
1. Keep population exploitation/checkpoint inheritance from Ray PBT/PB2.
2. Build temporally aligned GP transitions using the *applied* configuration.
3. Keep synthetic donor records out of GP training while using them as the
   predecessor of the receiver's first post-exploitation observation.
4. Normalize reward context causally with a task-agnostic robust transform.
5. Optimize GAE lambda instead of entropy; reserve entropy for O2I only.
6. Include applied entropy as a non-search execution feature in GP observations.
7. Preserve inherited outer-loop hyperparameters on fallback and reset the O2I
   actuator to its fixed baseline when no current guided intervention exists.
8. Seed the receiver's learner-state EMA from the donor state after cloning.
9. Keep O2I independent of GP posterior-uncertainty scaling.
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



def _normalized_intervention_magnitude(
    current_config: Dict[str, float],
    proposed_config: Dict[str, float],
    model_bounds: Dict[str, Tuple[float, float]],
) -> Tuple[float, Dict[str, float]]:
    """Return mean absolute normalized hyperparameter intervention in [0, 1].

    The comparison is performed only over the outer-loop optimized coordinates
    and uses their GP model bounds.  Entropy is excluded because it is the O2I
    actuator rather than an independently optimized hyperparameter.

    Returns
    -------
    magnitude:
        Mean absolute normalized coordinate change across all optimized
        hyperparameters.
    components:
        Per-coordinate absolute normalized changes, each in [0, 1].
    """
    names = list(model_bounds.keys())
    limits = np.asarray(
        [model_bounds[name] for name in names],
        dtype=np.float64,
    ).T

    current = np.asarray(
        [[current_config[name] for name in names]],
        dtype=np.float64,
    )
    proposed = np.asarray(
        [[proposed_config[name] for name in names]],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(current)):
        raise ValueError("Current configuration contains non-finite values")
    if not np.all(np.isfinite(proposed)):
        raise ValueError("Proposed configuration contains non-finite values")

    current_norm = np.clip(
        _normalize(current, limits),
        0.0,
        1.0,
    )
    proposed_norm = np.clip(
        _normalize(proposed, limits),
        0.0,
        1.0,
    )

    deltas = np.abs(
        proposed_norm - current_norm
    ).reshape(-1)

    components = {
        name: float(delta)
        for name, delta in zip(names, deltas)
    }

    magnitude = float(
        np.clip(
            np.mean(deltas),
            0.0,
            1.0,
        )
    )
    return magnitude, components


# -----------------------------------------------------------------------------
# Transition construction
# -----------------------------------------------------------------------------

def _prepare_gp_transitions(
    data: pd.DataFrame,
    hyperparam_bounds: Dict[str, Tuple[float, float]],
    execution_feature_bounds: Dict[str, Tuple[float, float]],
    use_learner_context: bool,
    max_gp_points: int = 1000,
) -> Tuple[pd.DataFrame, list[str]]:
    """Construct valid temporally aligned transitions for the GP.

    The GP models both the four outer-loop optimized hyperparameters and the
    applied entropy coefficient.  Entropy is an execution-only feature: it is
    observed by the surrogate because it affects PPO outcomes, but it is never
    an independently optimized acquisition coordinate.
    """
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
    candidate_model_transform,
) -> float:
    """PB2 UCB using a deterministic transform from search to execution space."""
    model_normalized = np.asarray(
        candidate_model_transform(np.asarray(hp_normalized, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    x = np.concatenate([fixed_context, model_normalized]).reshape(1, -1)
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
    candidate_model_transform,
    *,
    n_restarts: int = 10,
) -> np.ndarray:
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
    return best_theta


def _select_config(
    *,
    Xraw: np.ndarray,
    yraw: np.ndarray,
    current_nominal: Optional[np.ndarray],
    new_context: np.ndarray,
    current_hyperparams: Dict[str, float],
    search_bounds: Dict[str, Tuple[float, float]],
    model_bounds: Dict[str, Tuple[float, float]],
    execution_feature_bounds: Dict[str, Tuple[float, float]],
    context_columns: Sequence[str],
    context_bounds: Dict[str, Tuple[float, float]],
    use_o2i_feedback: bool,
    entropy_param: str,
    base_entropy_coeff: float,
    o2i_entropy_scale: float,
    max_entropy_increment: float,
    entropy_guard: float,
    reward_z_clip: float = 4.0,
) -> np.ndarray:
    """Fit the TV-GP and return the nominal four-dimensional PB2 proposal.

    Applied entropy is included in GP observations as an execution feature, but
    it is not a free acquisition coordinate.  For every candidate h, its entropy
    feature is deterministically generated from HIM(h_current, h).
    """
    hp_names = list(search_bounds.keys())
    if list(model_bounds.keys()) != hp_names:
        raise ValueError("search_bounds and model_bounds must have identical HP coordinates")
    if list(execution_feature_bounds.keys()) != [entropy_param]:
        raise ValueError("CODA currently expects entropy as the sole execution feature")

    search_vals = np.asarray(list(search_bounds.values()), dtype=np.float64).T
    hp_model_vals = np.asarray(list(model_bounds.values()), dtype=np.float64).T
    execution_vals = np.asarray(
        [execution_feature_bounds[entropy_param]], dtype=np.float64
    ).T
    full_model_vals = np.hstack((hp_model_vals, execution_vals))

    n_context = len(context_columns)
    if Xraw.ndim != 2 or Xraw.shape[1] != n_context + full_model_vals.shape[1]:
        raise ValueError(
            "Xraw shape is inconsistent with context/execution dimensions"
        )

    X_context_raw = Xraw[:, :n_context]
    X_execution_raw = Xraw[:, n_context:]

    X_context, fixed = _normalize_context(
        X_context_raw,
        new_context,
        context_columns,
        context_bounds,
        reward_z_clip=reward_z_clip,
    )
    X_execution = _normalize(X_execution_raw, full_model_vals)
    X = np.hstack((X_context, X_execution))
    y = _standardize_response(yraw).reshape(-1, 1)

    mean_model = _fit_gp(X, y)

    if current_nominal is None or len(current_nominal) == 0:
        variance_model = deepcopy(mean_model)
    else:
        current_norm = _normalize(current_nominal, full_model_vals)
        padding = np.tile(fixed, (current_norm.shape[0], 1))
        X_pending = np.hstack((padding, current_norm))
        X_aug = np.vstack((X, X_pending))
        y_aug = np.vstack((y, np.zeros((current_norm.shape[0], 1))))
        variance_model = _fit_gp(X_aug, y_aug)

    current_hp = np.asarray(
        [[current_hyperparams[name] for name in hp_names]], dtype=np.float64
    )
    current_hp_norm = np.clip(
        _normalize(current_hp, hp_model_vals).reshape(-1), 0.0, 1.0
    )

    def candidate_model_transform(hp_norm: np.ndarray) -> np.ndarray:
        hp_norm = np.asarray(hp_norm, dtype=np.float64).reshape(-1)
        if hp_norm.size != len(hp_names):
            raise ValueError("Candidate HP vector has incorrect dimensionality")

        hp_norm_clipped = np.clip(hp_norm, 0.0, 1.0)
        if use_o2i_feedback:
            intervention = float(
                np.clip(np.mean(np.abs(hp_norm_clipped - current_hp_norm)), 0.0, 1.0)
            )
            increment = min(
                float(max_entropy_increment),
                float(o2i_entropy_scale) * intervention,
            )
        else:
            increment = 0.0

        applied_entropy = float(
            np.clip(
                float(base_entropy_coeff) + increment,
                0.0,
                float(entropy_guard),
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

    xt_norm = _optimize_acquisition(
        mean_model,
        variance_model,
        fixed,
        normalized_search_bounds,
        candidate_model_transform,
        n_restarts=10,
    )

    xt = _denormalize(xt_norm, hp_model_vals).astype(np.float64)
    return xt


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------

class CODAScheduler(PopulationBasedTraining):
    """CODA with lambda in the outer search and entropy reserved for O2I."""

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
        base_entropy_coeff: float = 0.0,
        o2i_entropy_scale: float = 0.005,
        max_entropy_increment: float = 0.005,
        entropy_guard: float = 0.05,
        reward_z_clip: float = 4.0,
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
        self.o2i_entropy_scale = float(o2i_entropy_scale)
        self.max_entropy_increment = float(max_entropy_increment)
        self.entropy_guard = float(entropy_guard)
        self.reward_z_clip = float(reward_z_clip)

        if self.min_valid_transitions < 2:
            raise ValueError("min_valid_transitions must be >= 2")
        if self.reward_z_clip <= 0:
            raise ValueError("reward_z_clip must be > 0")
        if self.base_entropy_coeff < 0:
            raise ValueError("base_entropy_coeff must be >= 0")
        if self.o2i_entropy_scale < 0:
            raise ValueError("o2i_entropy_scale must be >= 0")
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

        # The outer GP search/model coordinates are exactly the four optimized
        # hyperparameters. Entropy is excluded from search, but retained as an
        # execution-only GP feature so observed responses are conditioned on the
        # actual O2I actuator value used by PPO.
        self._model_bounds_flat = deepcopy(self._hyperparam_bounds_flat)
        max_applied_entropy = self.base_entropy_coeff + self.max_entropy_increment
        if max_applied_entropy > self.entropy_guard + 1e-12:
            raise ValueError(
                "entropy_guard must be >= base_entropy_coeff + max increment"
            )
        if max_applied_entropy <= 0.0:
            # A positive-width bound is required by affine normalization.
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
        # Entropy is not an outer-loop search coordinate. Every trial starts from
        # the same fixed actuator baseline and only CODA O2I may change it.
        trial.config[self.entropy_param] = float(self.base_entropy_coeff)
        trial.evaluated_params.update(flatten_dict(filled))
        self._set_trial_identity(trial.config, trial)

        bridge = self._bridge(trial.config)
        bridge.setdefault("lineage_generation", 0)
        bridge.setdefault("lineage_ema_seed", None)
        bridge.setdefault("guided_update", False)
        bridge.setdefault("o2i_intervention_magnitude", 0.0)
        for name in self._hyperparam_bounds_flat:
            bridge.setdefault(f"o2i_delta_{name}", 0.0)
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

        stability_index = metric("stability_index")
        stability_raw = metric("stability_raw")
        stability_valid = metric("stability_valid", 0.0)

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
            "stability_index": stability_index,
            "stability_raw": stability_raw,
            "stability_valid": stability_valid,
            "is_synthetic": False,
            "guided_update": bool(bridge.get("guided_update", False)),
            "o2i_intervention_magnitude": _safe_float(
                bridge.get("o2i_intervention_magnitude", 0.0), 0.0
            ),
            **{
                f"o2i_delta_{name}": _safe_float(
                    bridge.get(f"o2i_delta_{name}", 0.0), 0.0
                )
                for name in hp_names
            },
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
            "stability_index": donor_row.get("stability_index", np.nan),
            "stability_raw": donor_row.get("stability_raw", np.nan),
            "stability_valid": donor_row.get("stability_valid", 0.0),
            "is_synthetic": True,
            "guided_update": False,
            "o2i_intervention_magnitude": 0.0,
            **{
                f"o2i_delta_{name}": 0.0
                for name in hp_names
            },
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
                donor_row["stability_index"],
            ]
        return np.asarray(values, dtype=np.float64)


    def _propose_from_gp(
        self,
        donor: Trial,
        donor_config_flat: Dict,
    ) -> Tuple[Dict, bool, float, Dict[str, float]]:
        """Return nominal PB2 proposal and HIM over optimized coordinates."""
        transitions, context_columns = _prepare_gp_transitions(
            self.data,
            self._hyperparam_bounds_flat,
            self._execution_feature_bounds_flat,
            use_learner_context=self.use_learner_context,
            max_gp_points=self.max_gp_points,
        )

        hp_names = list(self._hyperparam_bounds_flat.keys())
        zero_components = {name: 0.0 for name in hp_names}

        donor_row = self._latest_trial_row(donor)
        fallback = (deepcopy(donor_config_flat), False, 0.0, zero_components)

        if donor_row is None:
            return fallback
        if len(transitions) < self.min_valid_transitions:
            return fallback

        new_context = self._new_context(donor_row)
        if not np.all(np.isfinite(new_context)):
            return fallback

        current_observed = {
            name: _safe_float(
                donor_row.get(name, donor_config_flat[name]),
                donor_config_flat[name],
            )
            for name in hp_names
        }

        gp_inputs = context_columns + hp_names + [self.entropy_param]
        Xraw = transitions[gp_inputs].to_numpy(dtype=np.float64)
        yraw = transitions["y"].to_numpy(dtype=np.float64)

        new_values = _select_config(
            Xraw=Xraw,
            yraw=yraw,
            current_nominal=self.current_nominal,
            new_context=new_context,
            current_hyperparams=current_observed,
            search_bounds=self._hyperparam_bounds_flat,
            model_bounds=self._model_bounds_flat,
            execution_feature_bounds=self._execution_feature_bounds_flat,
            context_columns=context_columns,
            context_bounds=self._context_bounds,
            use_o2i_feedback=self.use_o2i_feedback,
            entropy_param=self.entropy_param,
            base_entropy_coeff=self.base_entropy_coeff,
            o2i_entropy_scale=self.o2i_entropy_scale,
            max_entropy_increment=self.max_entropy_increment,
            entropy_guard=self.entropy_guard,
            reward_z_clip=self.reward_z_clip,
        )

        # Convert the continuous BO output to the actual executable HP values
        # before computing the final HIM used by the learner.
        new_flat = deepcopy(donor_config_flat)
        for idx, name in enumerate(hp_names):
            template = donor_config_flat[name]
            new_flat[name] = _cast_and_clip(
                new_values[idx],
                template,
                self._hyperparam_bounds_flat[name],
            )

        if self.use_o2i_feedback:
            intervention_magnitude, components = (
                _normalized_intervention_magnitude(
                    current_observed,
                    {name: new_flat[name] for name in hp_names},
                    self._model_bounds_flat,
                )
            )
        else:
            intervention_magnitude = 0.0
            components = zero_components

        return (
            new_flat,
            True,
            float(np.clip(intervention_magnitude, 0.0, 1.0)),
            components,
        )

    def _apply_o2i_feedback(
        self,
        config: Dict,
        *,
        intervention_magnitude: float,
        intervention_components: Dict[str, float],
        guided_update: bool,
    ) -> Dict:
        """Use entropy exclusively as the O2I actuator.

        Entropy is not selected by PB2. For a valid guided intervention:

            I = mean_j |h_new_norm[j] - h_current_norm[j]|
            entropy = base_entropy + min(max_increment, scale * I)

        If no current guided O2I intervention exists, entropy is reset to the
        fixed baseline. The inherited *outer-loop* hyperparameters remain
        unchanged on fallback.
        """
        out = deepcopy(config)
        bridge = self._bridge(out)

        bridge["guided_update"] = bool(guided_update)
        bridge["o2i_intervention_magnitude"] = float(
            intervention_magnitude if guided_update else 0.0
        )

        for name in self._hyperparam_bounds_flat:
            bridge[f"o2i_delta_{name}"] = float(
                intervention_components.get(name, 0.0) if guided_update else 0.0
            )

        base_entropy = float(self.base_entropy_coeff)
        bridge["base_entropy_coeff"] = base_entropy
        # Kept for compatibility with earlier audit scripts. It now means the
        # fixed pre-O2I baseline, not an outer-optimizer proposal.
        bridge["nominal_entropy_coeff"] = base_entropy

        if guided_update and self.use_o2i_feedback:
            increment = float(
                self.o2i_entropy_scale
                * np.clip(intervention_magnitude, 0.0, 1.0)
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
        applied_flat: Dict,
        guided_update: bool,
    ) -> None:
        """Store full execution-space proposals for PB2 batch diversification."""
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
        donor_row = self._latest_trial_row(
            trial_to_clone
        )

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
            float(
                self.data["Time"].max()
            )
            if not self.data.empty
            else -np.inf
        )
        if current_time > self.last_exploration_time:
            self.current_nominal = None

        (
            nominal_flat,
            guided,
            intervention_magnitude,
            intervention_components,
        ) = self._propose_from_gp(
            trial_to_clone,
            donor_flat,
        )

        new_config = unflatten_dict(
            nominal_flat
        )
        new_config = self._apply_o2i_feedback(
            new_config,
            intervention_magnitude=intervention_magnitude,
            intervention_components=intervention_components,
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


# Backward-compatible alias if older scripts import W_PB2.
W_PB2 = CODAScheduler
