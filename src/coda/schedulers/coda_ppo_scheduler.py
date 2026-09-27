#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PB2-core CODA scheduler for PPO.

Design goal
===========
This implementation deliberately keeps Ray Tune PB2's shared optimization
machinery unchanged and adds only the two CODA communication channels:

* I2O: append the PPO diagnostic S as an additional *fixed/context* coordinate.
* O2I: append the applied entropy coefficient as an execution coordinate and
  derive it from donor-relative GP uncertainty.

The O2I signal uses a threshold + warm-up gate.  No ARD kernel, log-space GP
geometry, robust reward normalization, custom response normalization, custom
GP regularization schedule, or alternative history-window rule is used here.
The spatial/temporal GP kernel is Ray's standard ``TV_SquaredExp`` and PB2's
``normalize``, ``standardize``, ``select_length``, UCB constants, and
multi-start L-BFGS-B settings are retained.


Variants exposed for experiments
--------------------------------
* ``full``: I2O + gated O2I.
* ``i2o``:  I2O only; entropy remains at the PPO baseline.
* ``o2i``:  gated O2I only; no diagnostic coordinate in the GP.

A private-equivalence mode is intentionally not exposed as a paper ablation.
The class provides ``pb2_equivalence_config`` only as a diagnostic helper; the
paper baseline should remain Ray's native ``PB2`` scheduler.
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
from ray.tune.schedulers.pb2 import PB2, _select_config as _ray_pb2_select_config
from ray.tune.schedulers.pb2_utils import (
    TV_SquaredExp,
    normalize,
    select_length,
    standardize,
)
from ray.tune.utils.util import flatten_dict, unflatten_dict

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("CODA requires scikit-learn, as does Ray PB2.") from exc

logger = logging.getLogger(__name__)

# These are the constants used by Ray PB2's UCB implementation.
_PB2_UCB_C1 = 0.2
_PB2_UCB_C2 = 0.4
_PB2_ACQ_RESTARTS = 10
_PB2_ACQ_MAXITER = 200
_PB2_MAX_HISTORY = 1000


def _trial_key(trial: Trial) -> str:
    return str(trial)


def _safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, RuntimeError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _metric_from_result(result: Dict, name: str, default=np.nan) -> float:
    custom = result.get("custom_metrics", {}) or {}
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


def _effective_o2i_signal(
    raw_u: float,
    n_obs: int,
    *,
    threshold: float,
    warmup_start: int,
    warmup_end: int,
) -> Tuple[float, float, float]:
    """Threshold and warm-up a normalized O2I uncertainty signal.

    Returns ``(thresholded_u, warmup_gain, effective_u)``.

    thresholded_u = clip((raw_u - threshold)/(1-threshold), 0, 1)
    warmup_gain   = clip((n_obs-warmup_start)/(warmup_end-warmup_start), 0, 1)
    effective_u   = thresholded_u * warmup_gain
    """
    u = float(np.clip(raw_u, 0.0, 1.0))

    if threshold >= 1.0:
        u_thr = 0.0
    else:
        u_thr = float(np.clip((u - threshold) / (1.0 - threshold), 0.0, 1.0))

    if warmup_end <= warmup_start:
        gain = 1.0 if int(n_obs) >= int(warmup_end) else 0.0
    else:
        gain = float(
            np.clip(
                (float(n_obs) - float(warmup_start))
                / (float(warmup_end) - float(warmup_start)),
                0.0,
                1.0,
            )
        )

    return u_thr, gain, float(u_thr * gain)


def _pb2_kernel() -> TV_SquaredExp:
    """Return the same initial kernel used by Ray PB2 2.55.1."""
    return TV_SquaredExp(variance=1.0, lengthscale=1.0, epsilon=0.1)


def _fit_pb2_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """Fit with Ray PB2's standard GP setup.

    Ray PB2 uses ``alpha=1e-10`` and, on a linear-algebra failure, perturbs the
    design matrix before fitting again.  We mirror that behavior rather than
    CODA's previous regularization-retry schedule.
    """
    kernel = _pb2_kernel()
    try:
        model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer="fmin_l_bfgs_b",
            alpha=1e-10,
        )
        model.fit(X, y)
        return model
    except np.linalg.LinAlgError:
        # Mirror Ray PB2 exactly.  This path is rarely exercised, but keeping
        # it identical avoids introducing a CODA-specific stabilization rule.
        X_retry = np.asarray(X, dtype=np.float64).copy()
        X_retry += np.eye(X_retry.shape[0]) * 1e-3
        model = GaussianProcessRegressor(
            kernel=_pb2_kernel(),
            optimizer="fmin_l_bfgs_b",
            alpha=1e-10,
        )
        model.fit(X_retry, y)
        return model


def _posterior_std(model: GaussianProcessRegressor, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    _, std = model.predict(x, return_std=True)
    return float(np.asarray(std).reshape(-1)[0])


def _kernel_prior_std(model: GaussianProcessRegressor) -> float:
    kernel = getattr(model, "kernel_", None)
    variance = _safe_float(getattr(kernel, "variance", np.nan), np.nan)
    if np.isfinite(variance) and variance > 0.0:
        return float(np.sqrt(variance))
    return float("nan")


@dataclass
class _O2IDecisionState:
    observed_model: GaussianProcessRegressor
    fixed: np.ndarray
    base_vals: np.ndarray
    hp_names: Tuple[str, ...]
    entropy_index: int
    donor_hp_normalized: np.ndarray
    donor_entropy_normalized: float
    donor_posterior_std: float
    prior_std: float
    n_obs: int
    threshold: float
    warmup_start: int
    warmup_end: int
    base_entropy_coeff: float
    o2i_uncertainty_scale: float
    max_entropy_increment: float
    entropy_guard: float


@dataclass(frozen=True)
class _ProposalDiagnostics:
    raw_u: float = 0.0
    thresholded_u: float = 0.0
    warmup_gain: float = 0.0
    effective_u: float = 0.0
    candidate_posterior_std: float = 0.0
    donor_posterior_std: float = 0.0
    prior_std: float = 0.0
    entropy_increment: float = 0.0
    applied_entropy_coeff: float = 0.0
    acquisition_value: float = np.nan
    gp_data_count: int = 0


def _raw_o2i_signal(
    state: _O2IDecisionState,
    hp_normalized: np.ndarray,
) -> Tuple[float, float]:
    """Compute donor-relative uncertainty using the observed-data PB2 GP."""
    hp = np.asarray(hp_normalized, dtype=np.float64).reshape(-1)
    candidate_query = np.concatenate(
        [state.fixed, hp, [state.donor_entropy_normalized]]
    )
    sigma_c = _posterior_std(state.observed_model, candidate_query)
    sigma_d = float(state.donor_posterior_std)
    additional = max(sigma_c - sigma_d, 0.0)

    if np.isfinite(state.prior_std):
        raw_u = float(np.clip(additional / (state.prior_std + 1e-12), 0.0, 1.0))
    else:
        raw_u = 1.0 if additional > 1e-12 else 0.0
    return raw_u, sigma_c


def _entropy_from_raw_u(
    state: _O2IDecisionState,
    raw_u: float,
) -> Tuple[float, float, float, float, float]:
    u_thr, gain, u_eff = _effective_o2i_signal(
        raw_u,
        state.n_obs,
        threshold=state.threshold,
        warmup_start=state.warmup_start,
        warmup_end=state.warmup_end,
    )
    increment = min(
        float(state.max_entropy_increment),
        float(state.o2i_uncertainty_scale) * float(u_eff),
    )
    entropy = float(
        np.clip(
            state.base_entropy_coeff + increment,
            0.0,
            state.entropy_guard,
        )
    )
    return u_thr, gain, u_eff, float(increment), entropy


def _normalize_entropy(entropy: float, base_vals: np.ndarray, entropy_index: int) -> float:
    lo = float(np.min(base_vals[:, entropy_index]))
    hi = float(np.max(base_vals[:, entropy_index]))
    return float((entropy - lo) / (hi - lo + 1e-8))


def _pb2_ucb_value_with_o2i(
    mean_model: GaussianProcessRegressor,
    variance_model: GaussianProcessRegressor,
    state: _O2IDecisionState,
    hp_normalized: np.ndarray,
) -> Tuple[float, _ProposalDiagnostics]:
    """PB2 UCB evaluated under the entropy induced by gated O2I."""
    hp = np.asarray(hp_normalized, dtype=np.float64).reshape(-1)
    raw_u, sigma_c = _raw_o2i_signal(state, hp)
    u_thr, gain, u_eff, increment, entropy = _entropy_from_raw_u(state, raw_u)
    ent_norm = _normalize_entropy(entropy, state.base_vals, state.entropy_index)

    execution = np.concatenate([hp, [ent_norm]])
    x = np.concatenate([state.fixed, execution]).reshape(1, -1)

    try:
        mean = float(mean_model.predict(x)[0])
    except ValueError:
        mean = -9999.0

    try:
        _, std = variance_model.predict(x, return_std=True)
        variance = float(np.asarray(std).reshape(-1)[0] ** 2)
    except ValueError:
        variance = 0.0

    beta_t = _PB2_UCB_C1 + max(
        0.0,
        np.log(_PB2_UCB_C2 * mean_model.X_train_.shape[0]),
    )
    kappa = float(np.sqrt(beta_t))
    value = float(mean + kappa * variance)

    diagnostics = _ProposalDiagnostics(
        raw_u=float(raw_u),
        thresholded_u=float(u_thr),
        warmup_gain=float(gain),
        effective_u=float(u_eff),
        candidate_posterior_std=float(sigma_c),
        donor_posterior_std=float(state.donor_posterior_std),
        prior_std=float(state.prior_std),
        entropy_increment=float(increment),
        applied_entropy_coeff=float(entropy),
        acquisition_value=value,
        gp_data_count=int(state.n_obs),
    )
    return value, diagnostics


def _optimize_pb2_acquisition_with_o2i(
    mean_model: GaussianProcessRegressor,
    variance_model: GaussianProcessRegressor,
    state: _O2IDecisionState,
) -> Tuple[np.ndarray, float]:
    """Ray-PB2-equivalent UCB optimizer over only the outer HP coordinates."""
    bounds = [(0.0, 1.0) for _ in state.hp_names]
    best_value = -999.0
    best_theta = np.asarray(state.donor_hp_normalized, dtype=np.float64).copy()

    opts = {"maxiter": _PB2_ACQ_MAXITER, "maxfun": _PB2_ACQ_MAXITER, "disp": False}
    for _ in range(_PB2_ACQ_RESTARTS):
        x0 = np.random.uniform(0.0, 1.0, len(state.hp_names))
        res = minimize(
            lambda x: -_pb2_ucb_value_with_o2i(
                mean_model, variance_model, state, x
            )[0],
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=opts,
        )
        theta = np.asarray(res.x, dtype=np.float64)
        if not np.all(np.isfinite(theta)):
            continue
        value, _ = _pb2_ucb_value_with_o2i(
            mean_model, variance_model, state, theta
        )
        if value > best_value:
            best_value = float(value)
            best_theta = theta.copy()

    return np.clip(best_theta, 0.0, 1.0), float(best_value)


def _select_config_o2i(
    Xraw: np.ndarray,
    yraw: np.ndarray,
    current: Optional[np.ndarray],
    newpoint: np.ndarray,
    donor_hyperparams: Dict[str, float],
    model_bounds: Dict[str, Tuple[float, float]],
    hp_names: Sequence[str],
    *,
    num_f: int,
    entropy_param: str,
    base_entropy_coeff: float,
    o2i_uncertainty_scale: float,
    max_entropy_increment: float,
    entropy_guard: float,
    threshold: float,
    warmup_start: int,
    warmup_end: int,
) -> Tuple[np.ndarray, _O2IDecisionState, float]:
    """PB2 selection with one derived execution coordinate (entropy)."""
    # Same adaptive history selection used by PB2.
    length = select_length(Xraw, yraw, model_bounds, num_f)
    Xraw = Xraw[-length:, :]
    yraw = yraw[-length:]

    base_vals = np.array(list(model_bounds.values()), dtype=np.float64).T
    oldpoints = Xraw[:, :num_f]
    old_lims = np.concatenate(
        (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
    ).reshape(2, oldpoints.shape[1])
    limits = np.concatenate((old_lims, base_vals), axis=1)

    X = normalize(Xraw, limits)
    y = standardize(yraw).reshape(yraw.size, 1)
    fixed = normalize(newpoint, oldpoints)

    observed_model = _fit_pb2_gp(X, y)

    if current is None:
        variance_model = deepcopy(observed_model)
    else:
        padding = np.array([fixed for _ in range(current.shape[0])])
        current_norm = normalize(current, base_vals)
        current_aug = np.hstack((padding, current_norm))
        Xnew = np.vstack((X, current_aug))
        ypad = np.zeros(current.shape[0]).reshape(-1, 1)
        ynew = np.vstack((y, ypad))
        variance_model = _fit_pb2_gp(Xnew, ynew)

    hp_names = tuple(hp_names)
    donor_hp_raw = np.asarray(
        [[float(donor_hyperparams[name]) for name in hp_names]], dtype=np.float64
    )
    hp_base_vals = base_vals[:, : len(hp_names)]
    donor_hp_norm = normalize(donor_hp_raw, hp_base_vals).reshape(-1)

    entropy_index = len(hp_names)
    # O2I compares candidate and donor under the donor's currently applied
    # entropy condition, while the next entropy is generated from the baseline.
    donor_entropy = float(
        np.clip(
            donor_hyperparams.get(entropy_param, base_entropy_coeff),
            0.0,
            entropy_guard,
        )
    )
    donor_entropy_norm = _normalize_entropy(donor_entropy, base_vals, entropy_index)
    donor_query = np.concatenate([fixed, donor_hp_norm, [donor_entropy_norm]])
    donor_posterior_std = _posterior_std(observed_model, donor_query)
    prior_std = _kernel_prior_std(observed_model)

    state = _O2IDecisionState(
        observed_model=observed_model,
        fixed=np.asarray(fixed, dtype=np.float64),
        base_vals=np.asarray(base_vals, dtype=np.float64),
        hp_names=hp_names,
        entropy_index=entropy_index,
        donor_hp_normalized=np.asarray(donor_hp_norm, dtype=np.float64),
        donor_entropy_normalized=float(donor_entropy_norm),
        donor_posterior_std=float(donor_posterior_std),
        prior_std=float(prior_std),
        n_obs=int(observed_model.X_train_.shape[0]),
        threshold=float(threshold),
        warmup_start=int(warmup_start),
        warmup_end=int(warmup_end),
        base_entropy_coeff=float(base_entropy_coeff),
        o2i_uncertainty_scale=float(o2i_uncertainty_scale),
        max_entropy_increment=float(max_entropy_increment),
        entropy_guard=float(entropy_guard),
    )

    xt_norm, acquisition_value = _optimize_pb2_acquisition_with_o2i(
        observed_model, variance_model, state
    )

    hp_raw = xt_norm * (
        np.max(hp_base_vals, axis=0) - np.min(hp_base_vals, axis=0)
    ) + np.min(hp_base_vals, axis=0)
    return hp_raw.astype(np.float32), state, float(acquisition_value)


def _final_o2i_for_executable_config(
    state: _O2IDecisionState,
    executable_hp: Dict[str, float],
    acquisition_value: float,
) -> _ProposalDiagnostics:
    hp_raw = np.asarray(
        [[float(executable_hp[name]) for name in state.hp_names]], dtype=np.float64
    )
    hp_base_vals = state.base_vals[:, : len(state.hp_names)]
    hp_norm = normalize(hp_raw, hp_base_vals).reshape(-1)

    raw_u, sigma_c = _raw_o2i_signal(state, hp_norm)
    u_thr, gain, u_eff, increment, entropy = _entropy_from_raw_u(state, raw_u)
    return _ProposalDiagnostics(
        raw_u=float(raw_u),
        thresholded_u=float(u_thr),
        warmup_gain=float(gain),
        effective_u=float(u_eff),
        candidate_posterior_std=float(sigma_c),
        donor_posterior_std=float(state.donor_posterior_std),
        prior_std=float(state.prior_std),
        entropy_increment=float(increment),
        applied_entropy_coeff=float(entropy),
        acquisition_value=float(acquisition_value),
        gp_data_count=int(state.n_obs),
    )


def _prepare_pb2_style_transitions(
    data: pd.DataFrame,
    hp_names: Sequence[str],
    *,
    entropy_param: str,
    use_i2o: bool,
    use_o2i: bool,
) -> Tuple[pd.DataFrame, list[str], list[str]]:
    """Construct PB2-style transition data with only channel-specific additions."""
    df = data.sort_values(by="Time").reset_index(drop=True).copy()

    group_keys = ["Trial"] + list(hp_names)
    model_feature_names = list(hp_names)
    if use_o2i:
        group_keys += [entropy_param]
        model_feature_names += [entropy_param]

    grouped = df.groupby(group_keys, sort=False, dropna=False)
    df["y"] = grouped["Reward"].diff()
    df["t_change"] = grouped["Time"].diff()

    if use_i2o:
        df["S_before"] = grouped["policy_update_state"].shift(1)
        df["S_valid_before"] = grouped["policy_update_state_valid"].shift(1)

    df = df[df["t_change"] > 0].reset_index(drop=True)
    df["R_before"] = df.Reward - df.y
    df["y"] = df.y / df.t_change
    df = df[~df.y.isna()].reset_index(drop=True)

    if use_i2o:
        valid_s = (
            df["S_valid_before"].fillna(0.0).astype(float) >= 0.5
        ) & np.isfinite(df["S_before"].to_numpy(dtype=np.float64))
        df = df.loc[valid_s].reset_index(drop=True)

    df = df.sort_values(by="Time").reset_index(drop=True)
    df = df.iloc[-_PB2_MAX_HISTORY:, :].reset_index(drop=True)

    fixed_columns = ["Time", "R_before"]
    if use_i2o:
        fixed_columns.append("S_before")
    return df, fixed_columns, model_feature_names


class CODAPPOOptimizer(PB2):
    """CODA for PPO with a strict Ray-PB2 optimization core.

    Relative to native PB2, the only algorithmic additions are I2O and/or O2I.
    The O2I channel includes the threshold/warm-up gate by design.
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
        quantile_fraction: float = 0.25,
        variant: str = "full",
        entropy_param: str = "entropy_coeff",
        base_entropy_coeff: float = 0.0,
        o2i_uncertainty_scale: float = 0.004,
        max_entropy_increment: float = 0.004,
        entropy_guard: float = 0.004,
        o2i_threshold: float = 0.10,
        o2i_warmup_start: int = 16,
        o2i_warmup_end: int = 64,
        log_config: bool = True,
        require_attrs: bool = True,
        synch: bool = False,
        custom_explore_fn=None,
    ):
        variant = str(variant).lower()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VALID_VARIANTS)}")

        self.variant = variant
        self.use_i2o = variant in {"full", "i2o"}
        self.use_o2i = variant in {"full", "o2i"}

        self.entropy_param = str(entropy_param)
        self.base_entropy_coeff = float(base_entropy_coeff)
        self.o2i_uncertainty_scale = float(o2i_uncertainty_scale)
        self.max_entropy_increment = float(max_entropy_increment)
        self.entropy_guard = float(entropy_guard)
        self.o2i_threshold = float(o2i_threshold)
        self.o2i_warmup_start = int(o2i_warmup_start)
        self.o2i_warmup_end = int(o2i_warmup_end)

        if not 0.0 <= self.o2i_threshold < 1.0:
            raise ValueError("o2i_threshold must satisfy 0 <= threshold < 1")
        if self.o2i_warmup_start < 0:
            raise ValueError("o2i_warmup_start must be >= 0")
        if self.o2i_warmup_end < self.o2i_warmup_start:
            raise ValueError("o2i_warmup_end must be >= o2i_warmup_start")
        if self.base_entropy_coeff < 0.0:
            raise ValueError("base_entropy_coeff must be >= 0")
        if self.o2i_uncertainty_scale < 0.0:
            raise ValueError("o2i_uncertainty_scale must be >= 0")
        if self.max_entropy_increment < 0.0:
            raise ValueError("max_entropy_increment must be >= 0")
        if self.entropy_guard <= 0.0:
            raise ValueError("entropy_guard must be > 0")
        if self.base_entropy_coeff + self.max_entropy_increment > self.entropy_guard + 1e-12:
            raise ValueError(
                "entropy_guard must be >= base_entropy_coeff + max_entropy_increment"
            )

        hyperparam_bounds = hyperparam_bounds or {}
        if not hyperparam_bounds:
            raise TuneError("`hyperparam_bounds` must be specified for CODA/PB2.")

        # Native PB2 owns the shared population, history, normalization, window
        # selection, GP kernel, UCB constants, and pending-proposal machinery.
        super().__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            hyperparam_bounds=hyperparam_bounds,
            quantile_fraction=quantile_fraction,
            log_config=log_config,
            require_attrs=require_attrs,
            synch=synch,
            custom_explore_fn=custom_explore_fn,
        )

        # Monotone scheduler-side generation token used only to apply the
        # inherited I2O EMA seed once after each checkpoint exploitation.
        self._lineage_generation = 0

        self._model_bounds_flat = deepcopy(self._hyperparam_bounds_flat)
        if self.use_o2i:
            # This coordinate exists only because O2I changes the learner's
            # execution condition.  Turning O2I off removes it from the GP.
            upper = max(
                self.base_entropy_coeff + self.max_entropy_increment,
                1e-12,
            )
            self._model_bounds_flat[self.entropy_param] = [0.0, float(upper)]

    @staticmethod
    def _bridge(config: Dict) -> Dict:
        model = config.setdefault("model", {})
        custom = model.setdefault("custom_model_config", {})
        return custom.setdefault("_coda_bridge", {})

    def _set_trial_identity(self, config: Dict, trial: Trial) -> None:
        bridge = self._bridge(config)
        bridge["trial_id"] = str(getattr(trial, "trial_id", str(trial)))
        bridge["trial_name"] = str(trial)

    def on_trial_add(
        self,
        tune_controller: "TuneController",
        trial: Trial,
    ):
        # Entropy is a fixed PPO setting for PB2/I2O and becomes controlled only
        # when O2I is active.  Initial value remains the baseline in every case.
        trial.config[self.entropy_param] = float(self.base_entropy_coeff)
        self._set_trial_identity(trial.config, trial)
        bridge = self._bridge(trial.config)
        bridge.setdefault("lineage_generation", 0)
        bridge.setdefault("lineage_ema_seed", None)
        bridge.setdefault("guided_update", False)
        bridge.setdefault("o2i_uncertainty_raw", 0.0)
        bridge.setdefault("o2i_uncertainty_thresholded", 0.0)
        bridge.setdefault("o2i_warmup_gain", 0.0)
        bridge.setdefault("o2i_uncertainty_effective", 0.0)
        bridge.setdefault("entropy_increment", 0.0)
        bridge.setdefault("applied_entropy_coeff", float(self.base_entropy_coeff))
        bridge.setdefault("gp_data_count", 0)
        super().on_trial_add(tune_controller, trial)

    def _save_trial_state(
        self,
        state,
        time: int,
        result: Dict,
        trial: Trial,
    ):
        # Native PB2 first stores Trial/Time/HP/Reward exactly as usual.
        out = super()._save_trial_state(state, time, result, trial)
        if self.data.empty:
            return out

        idx = self.data.index[-1]
        policy_state = _metric_from_result(result, "policy_update_state", np.nan)
        policy_valid = _metric_from_result(
            result, "policy_update_state_valid", 0.0
        )
        flat = flatten_dict(trial.config)
        applied_entropy = _safe_float(
            flat.get(self.entropy_param, self.base_entropy_coeff),
            self.base_entropy_coeff,
        )
        bridge = (
            trial.config.get("model", {})
            .get("custom_model_config", {})
            .get("_coda_bridge", {})
        )

        # Extra columns are observational/audit information.  Only S is used by
        # I2O and entropy only by O2I.
        self.data.loc[idx, "policy_update_state"] = policy_state
        self.data.loc[idx, "policy_update_state_valid"] = policy_valid
        self.data.loc[idx, self.entropy_param] = applied_entropy
        self.data.loc[idx, "guided_update"] = float(
            bool(bridge.get("guided_update", False))
        )
        self.data.loc[idx, "o2i_uncertainty_raw"] = _safe_float(
            bridge.get("o2i_uncertainty_raw", 0.0), 0.0
        )
        self.data.loc[idx, "o2i_uncertainty_thresholded"] = _safe_float(
            bridge.get("o2i_uncertainty_thresholded", 0.0), 0.0
        )
        self.data.loc[idx, "o2i_warmup_gain"] = _safe_float(
            bridge.get("o2i_warmup_gain", 0.0), 0.0
        )
        self.data.loc[idx, "o2i_uncertainty_effective"] = _safe_float(
            bridge.get("o2i_uncertainty_effective", 0.0), 0.0
        )
        self.data.loc[idx, "entropy_increment"] = _safe_float(
            bridge.get("entropy_increment", 0.0), 0.0
        )
        self.data.loc[idx, "gp_data_count"] = _safe_float(
            bridge.get("gp_data_count", 0), 0.0
        )
        return out

    def _latest_real_row(self, trial: Trial) -> Optional[pd.Series]:
        if self.data.empty:
            return None
        rows = self.data[self.data["Trial"].eq(str(trial))]
        if rows.empty:
            return None
        return rows.iloc[-1]

    def _set_channel_bridge(
        self,
        config: Dict,
        *,
        receiver: Trial,
        donor_row: Optional[pd.Series],
        diagnostics: _ProposalDiagnostics,
        guided_update: bool,
    ) -> None:
        self._set_trial_identity(config, receiver)
        bridge = self._bridge(config)
        bridge["guided_update"] = bool(guided_update)
        bridge["gp_data_count"] = int(diagnostics.gp_data_count)
        bridge["o2i_uncertainty_raw"] = float(diagnostics.raw_u)
        bridge["o2i_uncertainty_thresholded"] = float(diagnostics.thresholded_u)
        bridge["o2i_warmup_gain"] = float(diagnostics.warmup_gain)
        bridge["o2i_uncertainty_effective"] = float(diagnostics.effective_u)
        bridge["entropy_increment"] = float(diagnostics.entropy_increment)
        bridge["applied_entropy_coeff"] = float(
            diagnostics.applied_entropy_coeff
            if self.use_o2i and guided_update
            else self.base_entropy_coeff
        )

        # The checkpoint itself is still inherited by native PBT/PB2. I2O only
        # transfers the diagnostic EMA seed required by the learner callback.
        # The generation token makes the callback apply that seed exactly once
        # after the cloned configuration/checkpoint becomes active.
        self._lineage_generation += 1
        bridge["lineage_generation"] = int(self._lineage_generation)

        seed = np.nan
        if self.use_i2o and donor_row is not None:
            valid = _safe_float(
                donor_row.get("policy_update_state_valid", 0.0), 0.0
            ) >= 0.5
            if valid:
                seed = _safe_float(
                    donor_row.get("policy_update_state", np.nan), np.nan
                )
        bridge["lineage_ema_seed"] = float(seed) if np.isfinite(seed) else None

    def _coda_explore(
        self,
        data: pd.DataFrame,
        current: Optional[np.ndarray],
        base: Trial,
        old: Trial,
        config: Dict,
    ) -> Tuple[Dict, pd.DataFrame, bool, _ProposalDiagnostics]:
        hp_names = list(self._hyperparam_bounds_flat.keys())
        df, fixed_columns, model_features = _prepare_pb2_style_transitions(
            data,
            hp_names,
            entropy_param=self.entropy_param,
            use_i2o=self.use_i2o,
            use_o2i=self.use_o2i,
        )

        fallback_diag = _ProposalDiagnostics(
            applied_entropy_coeff=float(self.base_entropy_coeff),
            gp_data_count=int(len(df)),
        )
        dfnewpoint = df[df["Trial"] == str(base)]
        if dfnewpoint.empty:
            return config.copy(), data, False, fallback_diag

        y = np.asarray(df["y"].values, dtype=np.float64)
        t_r = df[fixed_columns]
        model_df = df[model_features]
        X = pd.concat([t_r, model_df], axis=1).values
        newpoint = dfnewpoint.iloc[-1][fixed_columns].values.astype(np.float64)

        if self.use_o2i:
            donor_row = self._latest_real_row(base)
            donor_hparams = {
                name: _safe_float(
                    donor_row.get(name, config[name]) if donor_row is not None else config[name],
                    config[name],
                )
                for name in hp_names
            }
            donor_hparams[self.entropy_param] = _safe_float(
                donor_row.get(self.entropy_param, self.base_entropy_coeff)
                if donor_row is not None
                else self.base_entropy_coeff,
                self.base_entropy_coeff,
            )

            new, decision_state, acquisition_value = _select_config_o2i(
                X,
                y,
                current,
                newpoint,
                donor_hparams,
                self._model_bounds_flat,
                hp_names,
                num_f=len(fixed_columns),
                entropy_param=self.entropy_param,
                base_entropy_coeff=self.base_entropy_coeff,
                o2i_uncertainty_scale=self.o2i_uncertainty_scale,
                max_entropy_increment=self.max_entropy_increment,
                entropy_guard=self.entropy_guard,
                threshold=self.o2i_threshold,
                warmup_start=self.o2i_warmup_start,
                warmup_end=self.o2i_warmup_end,
            )
        else:
            # This is Ray PB2's own selection function.  The only difference
            # for I2O is that S_before is an additional fixed coordinate.
            new = _ray_pb2_select_config(
                X,
                y,
                current,
                newpoint,
                self._hyperparam_bounds_flat,
                num_f=len(fixed_columns),
            )
            decision_state = None
            acquisition_value = np.nan

        new_config = config.copy()
        values = []
        for i, name in enumerate(hp_names):
            # Match PB2's type-casting behavior exactly.
            type_ = type(config[name])
            new_config[name] = type_(new[i])
            values.append(type_(new[i]))

        if self.use_o2i:
            executable_hp = {name: new_config[name] for name in hp_names}
            diagnostics = _final_o2i_for_executable_config(
                decision_state,
                executable_hp,
                acquisition_value,
            )
            new_config[self.entropy_param] = float(
                diagnostics.applied_entropy_coeff
            )
        else:
            new_config[self.entropy_param] = float(self.base_entropy_coeff)
            diagnostics = _ProposalDiagnostics(
                applied_entropy_coeff=float(self.base_entropy_coeff),
                acquisition_value=float(acquisition_value),
                gp_data_count=int(len(df)),
            )

        # PB2 itself appends a synthetic observation at the copied agent state so
        # that the next improvement can be associated with the newly selected
        # configuration.  We preserve that mechanism and add S/entropy only as
        # required by the active CODA channels.
        latest = dfnewpoint.iloc[-1]
        new_T = latest["Time"]
        new_reward = latest["Reward"]
        row = {
            "Trial": str(old),
            "Time": new_T,
            **{name: new_config[name] for name in hp_names},
            "Reward": new_reward,
            self.entropy_param: float(new_config[self.entropy_param]),
        }

        donor_real = self._latest_real_row(base)
        if donor_real is not None:
            row["policy_update_state"] = donor_real.get(
                "policy_update_state", np.nan
            )
            row["policy_update_state_valid"] = donor_real.get(
                "policy_update_state_valid", 0.0
            )
        else:
            row["policy_update_state"] = np.nan
            row["policy_update_state_valid"] = 0.0

        data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
        data["Trial"] = data["Trial"].astype(str)
        return new_config, data, True, diagnostics

    def _get_new_config(
        self,
        trial: Trial,
        trial_to_clone: Trial,
    ) -> Tuple[Dict, Dict]:
        """Clone through native PBT/PB2 and apply only active CODA channels."""
        if self.data["Time"].max() > self.last_exploration_time:
            self.current = None

        donor_flat = flatten_dict(trial_to_clone.config)
        donor_row = self._latest_real_row(trial_to_clone)

        new_config_flat, data, guided, diagnostics = self._coda_explore(
            self.data,
            self.current,
            trial_to_clone,
            trial,
            donor_flat,
        )
        self.data = data.copy()

        # Pending proposals use exactly the model-space coordinates.  O2I adds
        # entropy because the GP predicts under the candidate execution condition.
        current_names = list(self._hyperparam_bounds_flat.keys())
        if self.use_o2i:
            current_names += [self.entropy_param]
        new = np.asarray(
            [new_config_flat[name] for name in current_names], dtype=np.float64
        ).reshape(1, -1)

        if self.data["Time"].max() > self.last_exploration_time:
            self.last_exploration_time = self.data["Time"].max()
            self.current = new.copy()
        elif self.current is None:
            self.current = new.copy()
        else:
            self.current = np.concatenate((self.current, new), axis=0)

        new_config = unflatten_dict(new_config_flat)
        self._set_channel_bridge(
            new_config,
            receiver=trial,
            donor_row=donor_row,
            diagnostics=diagnostics,
            guided_update=guided,
        )

        if self._custom_explore_fn:
            new_config = self._custom_explore_fn(new_config)
            assert new_config is not None, (
                "Custom explore function failed to return a new config"
            )

        return new_config, {}

    @staticmethod
    def pb2_equivalence_config() -> Dict[str, object]:
        """Document the behavioral reductions needed for PB2 equivalence.

        This helper is for audit/documentation, not for a manuscript ablation.
        With no I2O coordinate and no O2I execution coordinate, the remaining
        machinery is native Ray PB2.  ARD is intentionally absent.
        """
        return {
            "kernel": "Ray TV_SquaredExp (isotropic)",
            "normalization": "Ray PB2 min-max normalize",
            "response": "Ray PB2 standardize/clip",
            "history_selection": "Ray PB2 select_length + <=1000 rows",
            "acquisition": "Ray PB2 UCB constants and L-BFGS-B settings",
            "log_coordinate_transform": False,
            "robust_reward_normalization": False,
            "ARD": False,
        }


# Public API for the final CODA scheduler.
__all__ = ["CODAPPOOptimizer"]
