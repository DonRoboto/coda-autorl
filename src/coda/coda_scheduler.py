#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CODA scheduler for Ray Tune / RLlib.

This module implements the three experimental variants from a single code path:

- ``full``: CODA (inner-to-outer context + outer-to-inner uncertainty feedback)
- ``i2o``:  CODA-I2O (contextual surrogate only)
- ``o2i``:  CODA-O2I (PB2-style surrogate + uncertainty feedback)

Design goals
------------
1. Keep population exploitation/checkpoint inheritance from Ray PBT/PB2.
2. Build temporally aligned GP transitions using the *applied* configuration.
3. Keep synthetic donor records out of GP training while using them as the
   predecessor of the receiver's first post-exploitation observation.
4. Use environment-specific, explicit context bounds.
5. Distinguish the outer PB2 search domain from the GP model domain.  In full
   CODA/O2I, the applied entropy coefficient can exceed the nominal PB2 search
   maximum after uncertainty modulation, so the GP model domain uses the
   effective applied maximum while acquisition remains restricted to the
   nominal PB2 search interval.
6. Preserve the inherited configuration exactly when no surrogate-guided
   proposal can be produced.
7. Seed the receiver's learner-state EMA from the donor state after cloning.
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
except ImportError as exc:  # pragma: no cover - runtime dependency check
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
        raise ValueError(f"All normalization bounds must have positive width: {limits}")
    return (data - low) / scale


def _denormalize(data: np.ndarray, limits: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    low, high = limits[0], limits[1]
    return data * (high - low) + low


def _standardize_response(y: np.ndarray) -> np.ndarray:
    """PB2-compatible standardization, including the [-2, 2] clipping."""
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
    if isinstance(template, (np.floating,)):
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
    use_learner_context: bool,
    max_gp_points: int = 1000,
) -> Tuple[pd.DataFrame, list[str]]:
    """Construct valid temporally aligned transitions for the GP.

    Real rows store the configuration that was actually active during the
    interval ending at that row.  Synthetic rows are only lineage anchors and
    are excluded as supervised GP observations.
    """
    if data.empty:
        return pd.DataFrame(), []

    hp_names = list(hyperparam_bounds.keys())
    df = data.reset_index(drop=False).rename(columns={"index": "_insertion_order"})
    df = df.copy()

    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)

    df = df.sort_values(
        ["Trial", "Time", "_insertion_order"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = df.groupby("Trial", sort=False, dropna=False)

    # Pre-decision state for the interval ending at the current row.
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
        .sort_values(["Time", "_insertion_order"], kind="mergesort")
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
                "TV-GP fit failed with alpha=%g; retrying with more regularization: %s",
                alpha,
                exc,
            )

    raise RuntimeError("Unable to fit CODA TV-GP after regularization retries") from last_error


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
            lambda x: -_ucb_value(mean_model, variance_model, fixed_context, x),
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        value = _ucb_value(mean_model, variance_model, fixed_context, theta)
        if value > best_value:
            best_value = value
            best_theta = theta

    if best_theta is None:
        raise RuntimeError("CODA acquisition optimization failed to produce a candidate")
    return best_theta


def _select_config(
    *,
    Xraw: np.ndarray,
    yraw: np.ndarray,
    current_nominal: Optional[np.ndarray],
    new_context: np.ndarray,
    search_bounds: Dict[str, Tuple[float, float]],
    model_bounds: Dict[str, Tuple[float, float]],
    context_limits: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit the contextual TV-GP and return nominal PB2 proposal + std."""
    search_vals = np.asarray(list(search_bounds.values()), dtype=np.float64).T
    model_vals = np.asarray(list(model_bounds.values()), dtype=np.float64).T

    if search_vals.shape != model_vals.shape:
        raise ValueError("search_bounds and model_bounds must have identical coordinates")

    limits = np.concatenate((context_limits, model_vals), axis=1)
    X = _normalize(Xraw, limits)
    y = _standardize_response(yraw).reshape(-1, 1)
    fixed = _normalize(new_context.reshape(1, -1), context_limits).reshape(-1)

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

    # Acquisition candidates remain inside H_search even though the GP may have
    # observed applied entropy values outside that nominal interval.
    normalized_search_bounds = []
    for j in range(search_vals.shape[1]):
        model_lo, model_hi = model_vals[:, j]
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
        n_restarts=10,
    )

    # O2I uncertainty uses the batch-adjusted posterior standard deviation.
    x_eval = np.concatenate((fixed, xt_norm)).reshape(1, -1)
    _, std = variance_model.predict(x_eval, return_std=True)
    uncertainty = float(std.reshape(-1)[0])

    xt = _denormalize(xt_norm, model_vals).astype(np.float64)
    return xt, uncertainty


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------

class CODAScheduler(PopulationBasedTraining):
    """Closed-Loop Online Diagnostic-Aware AutoRL scheduler.

    Parameters specific to CODA are explicit so a single implementation can be
    used for the complete method and both directional ablations.
    """

    VALID_VARIANTS = {"full", "i2o", "o2i"}

    def __init__(
        self,
        *,
        time_attr: str = "time_total_s",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        perturbation_interval: float = 60.0,
        hyperparam_bounds: Optional[Dict[str, Union[dict, list, tuple]]] = None,
        context_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        quantile_fraction: float = 0.25,
        variant: str = "full",
        min_valid_transitions: int = 2,
        max_gp_points: int = 1000,
        entropy_param: str = "entropy_coeff",
        uncertainty_scale: float = 0.005,
        max_entropy_increment: float = 0.015,
        entropy_guard: float = 0.05,
        log_config: bool = True,
        require_attrs: bool = True,
        synch: bool = False,
    ):
        hyperparam_bounds = hyperparam_bounds or {}
        if not hyperparam_bounds:
            raise TuneError("`hyperparam_bounds` must be specified for CODA.")

        variant = str(variant).lower()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VALID_VARIANTS)}")

        self.variant = variant
        self.use_learner_context = variant in {"full", "i2o"}
        self.use_uncertainty_feedback = variant in {"full", "o2i"}
        self.min_valid_transitions = int(min_valid_transitions)
        self.max_gp_points = int(max_gp_points)
        self.entropy_param = entropy_param
        self.uncertainty_scale = float(uncertainty_scale)
        self.max_entropy_increment = float(max_entropy_increment)
        self.entropy_guard = float(entropy_guard)

        if self.min_valid_transitions < 2:
            raise ValueError("min_valid_transitions must be >= 2")

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
        self._validate_bounds(self._hyperparam_bounds_flat, "hyperparam_bounds")

        self._context_bounds = context_bounds or {}
        required_context = {"T_before"}
        if self.use_learner_context:
            required_context |= {"R_before", "S_before"}
        missing = required_context.difference(self._context_bounds)
        if missing:
            raise ValueError(f"Missing CODA context bounds: {sorted(missing)}")
        self._validate_bounds(
            {k: self._context_bounds[k] for k in required_context},
            "context_bounds",
        )

        # GP model bounds describe *applied* configurations.  For full/O2I the
        # entropy coordinate can exceed the nominal PB2 search upper bound.
        self._model_bounds_flat = deepcopy(self._hyperparam_bounds_flat)
        if self.use_uncertainty_feedback and self.entropy_param in self._model_bounds_flat:
            ent_low, ent_high = self._hyperparam_bounds_flat[self.entropy_param]
            effective_high = float(ent_high) + self.max_entropy_increment
            if effective_high > self.entropy_guard + 1e-12:
                raise ValueError(
                    "entropy_guard must be >= search upper bound + max increment"
                )
            self._model_bounds_flat[self.entropy_param] = [float(ent_low), effective_high]

        self.last_exploration_time = -np.inf
        self.current_nominal: Optional[np.ndarray] = None
        self._lineage_generation = 0

    @staticmethod
    def _validate_bounds(bounds: Dict, name: str) -> None:
        for key, value in bounds.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{name}[{key!r}] must be [low, high], got {value}")
            if float(value[0]) >= float(value[1]):
                raise ValueError(f"{name}[{key!r}] must satisfy low < high, got {value}")

    def _context_columns(self) -> list[str]:
        columns = ["T_before"]
        if self.use_learner_context:
            columns += ["R_before", "S_before"]
        return columns

    def _context_limits(self) -> np.ndarray:
        return np.asarray(
            [self._context_bounds[name] for name in self._context_columns()],
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

    def on_trial_add(self, tune_controller: "TuneController", trial: Trial):
        filled = _fill_config(trial.config, self._hyperparam_bounds)
        trial.evaluated_params.update(flatten_dict(filled))
        self._set_trial_identity(trial.config, trial)
        bridge = self._bridge(trial.config)
        bridge.setdefault("lineage_generation", 0)
        bridge.setdefault("lineage_ema_seed", None)
        bridge.setdefault("guided_update", False)
        bridge.setdefault("gp_uncertainty", 0.0)
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
            "gp_uncertainty": _safe_float(bridge.get("gp_uncertainty", 0.0), 0.0),
            "nominal_entropy_coeff": _safe_float(
                bridge.get("nominal_entropy_coeff", np.nan), np.nan
            ),
            "applied_entropy_coeff": _safe_float(
                bridge.get("applied_entropy_coeff", np.nan), np.nan
            ),
            "lineage_generation": bridge.get("lineage_generation", 0),
        }
        self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)
        return score

    def _latest_trial_row(self, trial: Trial) -> Optional[pd.Series]:
        """Return the latest *real* observation for a trial."""
        if self.data.empty:
            return None
        rows = self.data[self.data["Trial"].eq(_trial_key(trial))].copy()
        if "is_synthetic" in rows.columns:
            rows = rows[~rows["is_synthetic"].fillna(False).astype(bool)]
        if rows.empty:
            return None
        rows = rows.reset_index(drop=False).rename(columns={"index": "_order"})
        rows = rows.sort_values(["Time", "_order"], kind="mergesort")
        return rows.iloc[-1]

    def _append_lineage_anchor(
        self,
        *,
        donor_row: pd.Series,
        receiver: Trial,
    ) -> None:
        """Always insert the donor state as receiver's synthetic predecessor."""
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
            "nominal_entropy_coeff": np.nan,
            "applied_entropy_coeff": np.nan,
            "lineage_generation": self._lineage_generation + 1,
        }
        # Synthetic row represents the donor state, so store the donor's *applied*
        # active configuration.  The row is never itself a supervised GP target.
        for name in hp_names:
            row[name] = donor_row[name]
        self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)

    def _new_context(self, donor_row: pd.Series) -> np.ndarray:
        values = [donor_row["Time"]]
        if self.use_learner_context:
            values += [donor_row["Reward"], donor_row["stability_index"]]
        return np.asarray(values, dtype=np.float64)

    def _propose_from_gp(
        self,
        donor: Trial,
        donor_config_flat: Dict,
    ) -> Tuple[Dict, bool, float]:
        transitions, context_columns = _prepare_gp_transitions(
            self.data,
            self._hyperparam_bounds_flat,
            use_learner_context=self.use_learner_context,
            max_gp_points=self.max_gp_points,
        )

        donor_row = self._latest_trial_row(donor)
        if donor_row is None:
            return deepcopy(donor_config_flat), False, 0.0

        if len(transitions) < self.min_valid_transitions:
            return deepcopy(donor_config_flat), False, 0.0

        new_context = self._new_context(donor_row)
        if not np.all(np.isfinite(new_context)):
            return deepcopy(donor_config_flat), False, 0.0

        hp_names = list(self._hyperparam_bounds_flat.keys())
        gp_inputs = context_columns + hp_names
        Xraw = transitions[gp_inputs].to_numpy(dtype=np.float64)
        yraw = transitions["y"].to_numpy(dtype=np.float64)

        new_values, uncertainty = _select_config(
            Xraw=Xraw,
            yraw=yraw,
            current_nominal=self.current_nominal,
            new_context=new_context,
            search_bounds=self._hyperparam_bounds_flat,
            model_bounds=self._model_bounds_flat,
            context_limits=self._context_limits(),
        )

        new_flat = deepcopy(donor_config_flat)
        for idx, name in enumerate(hp_names):
            template = donor_config_flat[name]
            new_flat[name] = _cast_and_clip(
                new_values[idx], template, self._hyperparam_bounds_flat[name]
            )

        return new_flat, True, float(max(0.0, uncertainty))

    def _apply_uncertainty_feedback(
        self,
        config: Dict,
        *,
        uncertainty: float,
        guided_update: bool,
    ) -> Dict:
        """Map posterior std to the applied entropy coefficient.

        Crucially, no clipping/modulation is performed when a GP proposal was
        unavailable; the inherited donor configuration is then preserved exactly.
        """
        out = deepcopy(config)
        bridge = self._bridge(out)
        bridge["guided_update"] = bool(guided_update)
        bridge["gp_uncertainty"] = float(uncertainty if guided_update else 0.0)

        if self.entropy_param not in out:
            return out

        nominal_entropy = float(out[self.entropy_param])
        bridge["nominal_entropy_coeff"] = nominal_entropy

        if guided_update and self.use_uncertainty_feedback:
            search_low, search_high = self._hyperparam_bounds_flat[self.entropy_param]
            base_entropy = float(np.clip(nominal_entropy, search_low, search_high))
            increment = min(
                self.max_entropy_increment,
                self.uncertainty_scale * float(max(0.0, uncertainty)),
            )
            applied_entropy = float(
                np.clip(base_entropy + increment, 0.0, self.entropy_guard)
            )
            out[self.entropy_param] = applied_entropy
        else:
            # I2O or a fallback event: preserve the proposal/inherited value.
            applied_entropy = float(out[self.entropy_param])

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
            valid = _safe_float(donor_row.get("stability_valid", 0.0), 0.0) >= 0.5
            if valid:
                seed = _safe_float(donor_row.get("stability_index", np.nan), np.nan)
        bridge["lineage_ema_seed"] = float(seed) if np.isfinite(seed) else None

    def _update_pending_nominal(self, new_flat: Dict, guided_update: bool) -> None:
        if not guided_update:
            return
        vector = np.asarray(
            [new_flat[name] for name in self._hyperparam_bounds_flat],
            dtype=np.float64,
        ).reshape(1, -1)

        current_time = float(self.data["Time"].max()) if not self.data.empty else -np.inf
        if current_time > self.last_exploration_time:
            self.last_exploration_time = current_time
            self.current_nominal = vector.copy()
        elif self.current_nominal is None:
            self.current_nominal = vector.copy()
        else:
            self.current_nominal = np.vstack((self.current_nominal, vector))

    def _get_new_config(self, trial: Trial, trial_to_clone: Trial) -> Tuple[Dict, Dict]:
        donor_row = self._latest_trial_row(trial_to_clone)

        # Always install a lineage anchor before testing GP readiness.  This makes
        # the receiver's first future real observation continue from the donor.
        if donor_row is not None:
            self._append_lineage_anchor(donor_row=donor_row, receiver=trial)

        donor_flat = flatten_dict(trial_to_clone.config, prevent_delimiter=True)

        # Reset the pending batch when a new population time is reached.
        current_time = float(self.data["Time"].max()) if not self.data.empty else -np.inf
        if current_time > self.last_exploration_time:
            self.current_nominal = None

        nominal_flat, guided, uncertainty = self._propose_from_gp(
            trial_to_clone, donor_flat
        )
        self._update_pending_nominal(nominal_flat, guided)

        new_config = unflatten_dict(nominal_flat)
        new_config = self._apply_uncertainty_feedback(
            new_config,
            uncertainty=uncertainty,
            guided_update=guided,
        )
        self._set_lineage_bridge(
            new_config,
            receiver=trial,
            donor_row=donor_row,
        )

        return new_config, {}


# Backward-compatible name for existing experiment scripts.
W_PB2 = CODAScheduler
