#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CODA + SAC runner using the shared repository configuration.

Scientific instantiation
------------------------
Outer-loop coordinates (4-D):
    train_batch_size,
    tau,
    optimization/actor_learning_rate,
    optimization/critic_learning_rate.

I2O (inner -> outer):
    SAC critic TD-error -> bounded learner-state S_t.
    The current TD-error median is compared causally with an EMA reference from
    the same learner lineage. S_t is then smoothed by a second EMA.

O2I (outer -> inner):
    donor-relative incremental observed-GP uncertainty U_delta -> SAC target
    entropy. The baseline is -action_dim and uncertainty may move the target
    entropy toward zero by at most 10% of action_dim. SAC still adapts alpha
    internally; alpha itself is NOT a CODA search coordinate or actuator.

Checkpoint inheritance
----------------------
This runner deliberately uses RLlib's old API stack. SAC checkpoints are
configured with store_buffer_in_checkpoints=True so PBT-style donor cloning can
inherit the replay buffer in addition to policy/critic/optimizer state.

IMPORTANT BEFORE THE FINAL CAMPAIGN
-----------------------------------
1. Run the smoke test first.
2. Confirm custom_metrics/coda_effective_hp_mismatch_count == 0.
3. Confirm sac_td_error_median and policy_update_state are finite after learning
   starts.
4. Confirm target entropy lies in [-action_dim, -0.9*action_dim].
5. Confirm all four workers finish and have checkpoints.
6. Freeze the Ray version after the smoke test; this code touches old-stack
   Policy internals that may change between Ray releases.
"""

from __future__ import annotations

# CPU limits must be set before NumPy / Ray / PyTorch imports.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import json
import logging
import random
import shutil
import sys
import time
import traceback
import warnings
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.backends.cudnn as cudnn


warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


# =============================================================================
# REPOSITORY / SHARED CONFIGURATION
# =============================================================================

# Intended location:
#     <repo>/experiments/sac/train_coda.py
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from configs.environments import (  # noqa: E402
    ENVIRONMENTS,
    action_dimension,
    sac_baseline_target_entropy,
)
from configs.seeds import (  # noqa: E402
    TRAINING_SEEDS,
    SMOKE_TRAINING_SEED,
)
from configs.sac_config import (  # noqa: E402
    POPULATION_SIZE,
    MAX_TIMESTEPS_PER_WORKER,
    PERTURBATION_INTERVAL,
    QUANTILE_FRACTION,
    HYPERPARAM_BOUNDS,
    CODA_LOG10_COORDINATES,
    FIXED_GAMMA,
    FIXED_ENTROPY_LR,
    INITIAL_ALPHA,
    N_STEP,
    TARGET_NETWORK_UPDATE_FREQ,
    LEARNING_STARTS,
    TRAINING_INTENSITY,
    TWIN_Q,
    REPLAY_BUFFER_CAPACITY,
    REPLAY_BUFFER_CONFIG,
    STORE_BUFFER_IN_CHECKPOINTS,
    MODEL_HIDDENS,
    MODEL_ACTIVATION,
    NUM_ENV_RUNNERS,
    NUM_ENVS_PER_RUNNER,
    ROLLOUT_FRAGMENT_LENGTH,
    MIN_SAMPLE_TIMESTEPS_PER_ITERATION,
    OBSERVATION_FILTER,
    NUM_GPUS_PER_TRIAL,
    TD_REFERENCE_EMA_BETA,
    POLICY_STATE_EMA_BETA,
    POLICY_STATE_MIN,
    TD_REFERENCE_EPS,
    TARGET_ENTROPY_MAX_FRACTION,
    MIN_VALID_TRANSITIONS,
    MAX_GP_POINTS,
    REWARD_Z_CLIP,
    target_entropy_bounds,
    build_tunable_sac_config,
)
from coda.schedulers.coda_sac_scheduler import CODASACScheduler  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


# =============================================================================
# CODA VARIANT / METHOD-SPECIFIC CONFIGURATION
# =============================================================================

VARIANT = os.environ.get(
    "CODA_VARIANT",
    "full",
).strip().lower()

if VARIANT not in CODASACScheduler.VALID_VARIANTS:
    raise ValueError(
        "CODA_VARIANT must be one of "
        f"{sorted(CODASACScheduler.VALID_VARIANTS)}, "
        f"got {VARIANT!r}"
    )

ALGO_NAME = {
    "full": "CODA_SAC",
    "i2o": "CODA_SAC_I2O",
    "o2i": "CODA_SAC_O2I",
}[VARIANT]

# Keep current result names compatible with the evaluation pipeline.
RUN_TAG = os.environ.get(
    "CODA_SAC_RUN_TAG",
    "TD_DUQ_DONORREL",
).strip()

OUTPUT_NAME = (
    f"{ALGO_NAME}_{RUN_TAG}"
    if RUN_TAG
    else ALGO_NAME
)

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Compatibility aliases only. The source of truth is configs/sac_config.py.
POBLACION_B = POPULATION_SIZE
TIMESTEPS_MAX = MAX_TIMESTEPS_PER_WORKER

RESULTS_ROOT = REPO_ROOT / "results"

ENVIRONMENT_CONFIG: Dict[str, dict] = {
    env_name: {
        "semillas": list(TRAINING_SEEDS),
        "perturbation_interval": PERTURBATION_INTERVAL,
        "quantile_fraction": QUANTILE_FRACTION,
    }
    for env_name in ENVIRONMENTS
}


# =============================================================================
# SAFE SMOKE TEST
# =============================================================================

# Defaults to a structural smoke test so this file cannot accidentally launch
# 60 x 5M-step population runs. Set CODA_SAC_SMOKE=0 for the final campaign.
SAC_SMOKE_TEST = os.environ.get("CODA_SAC_SMOKE", "1") != "0"
SAC_SMOKE_ENV = os.environ.get("CODA_SAC_SMOKE_ENV", "Hopper-v5")
SAC_SMOKE_SEEDS = [
    int(
        os.environ.get(
            "CODA_SAC_SMOKE_SEED",
            str(SMOKE_TRAINING_SEED),
        )
    )
]
SAC_SMOKE_STEPS = int(os.environ.get("CODA_SAC_SMOKE_STEPS", "500000"))
SAC_SMOKE_INTERVAL = int(os.environ.get("CODA_SAC_SMOKE_INTERVAL", "50000"))


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_num_threads(1)


def _safe_float(value, default=np.nan) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item") and np.asarray(value).size == 1:
            value = value.item()
        value = float(value)
    except (TypeError, ValueError, RuntimeError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite_array(value) -> np.ndarray:
    """Convert an arbitrary tensor/array/list into a finite 1-D float array."""
    if value is None:
        return np.empty((0,), dtype=np.float64)
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return np.empty((0,), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _target_entropy_parameters(
    env_name: str,
) -> Tuple[int, float, float, Tuple[float, float]]:
    """Return environment-specific CODA-SAC actuator parameters."""
    action_dim = action_dimension(env_name)
    baseline = sac_baseline_target_entropy(env_name)
    actuator_bounds = target_entropy_bounds(action_dim)
    increment = float(actuator_bounds[1] - actuator_bounds[0])

    return (
        action_dim,
        baseline,
        increment,
        actuator_bounds,
    )


def _config_attr(config, name: str, default=None):
    try:
        value = getattr(config, name)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        return config[name]
    except Exception:
        return default


def _optimization_dict(config) -> dict:
    value = _config_attr(config, "optimization", {})
    return dict(value or {}) if isinstance(value, dict) else {}


# =============================================================================
# CALLBACK: I2O, O2I ACTUATOR SYNC, AND AUDIT
# =============================================================================


class CODASACTDCallback(DefaultCallbacks):
    """SAC-specific CODA bridge.

    Responsibilities:
    - compute TD-error based I2O state;
    - transfer lineage EMA state after checkpoint inheritance;
    - re-apply Tune-selected SAC hyperparameters after donor restore;
    - apply the target-entropy O2I actuator to the live SAC torch model;
    - audit desired vs effective hyperparameters.
    """

    def __init__(self):
        super().__init__()
        self._policy_state_ema: Optional[float] = None
        self._td_reference_ema: Optional[float] = None
        self._last_lineage_generation: Optional[int] = None
        self._iter_count = 0

    @staticmethod
    def _get_default_policy(algorithm):
        try:
            return algorithm.get_policy("default_policy")
        except Exception:
            try:
                return algorithm.get_policy()
            except Exception:
                return None

    @staticmethod
    def _bridge_from_algorithm(algorithm) -> dict:
        try:
            cfg = algorithm.config
            model_cfg = _config_attr(cfg, "model", {}) or {}
            custom_cfg = model_cfg.get("custom_model_config", {}) or {}
            return custom_cfg.get("_coda_bridge", {}) or {}
        except Exception:
            return {}

    @staticmethod
    def _extract_policy_and_learner_stats(result: dict) -> Tuple[dict, dict]:
        info = result.get("info", {}) or {}
        learner = info.get("learner", {}) or {}
        policy_data = learner.get("default_policy", {}) or {}
        if not isinstance(policy_data, dict):
            policy_data = {}
        learner_stats = policy_data.get("learner_stats", {}) or {}
        if not isinstance(learner_stats, dict):
            learner_stats = {}
        return policy_data, learner_stats

    @staticmethod
    def _set_target_entropy_tensor(policy, value: float) -> None:
        """Update target_entropy on the live model and GPU towers."""
        models = []
        model = getattr(policy, "model", None)
        if model is not None:
            models.append(model)

        towers = getattr(policy, "model_gpu_towers", None)
        if towers:
            models.extend(list(towers))

        with torch.no_grad():
            for tower in models:
                target = getattr(tower, "target_entropy", None)
                if target is None:
                    continue
                try:
                    target.fill_(float(value))
                except Exception:
                    try:
                        target.data.fill_(float(value))
                    except Exception:
                        pass

    def _sync_effective_hyperparams(self, algorithm) -> None:
        """Re-apply Tune values after a donor checkpoint restore."""
        policy = self._get_default_policy(algorithm)
        if policy is None:
            return

        cfg = algorithm.config
        optimization = _optimization_dict(cfg)

        desired_batch = _safe_float(
            _config_attr(cfg, "train_batch_size", np.nan), np.nan
        )
        desired_tau = _safe_float(_config_attr(cfg, "tau", np.nan), np.nan)
        desired_target_entropy = _safe_float(
            _config_attr(cfg, "target_entropy", np.nan), np.nan
        )
        desired_actor_lr = _safe_float(
            optimization.get("actor_learning_rate", np.nan), np.nan
        )
        desired_critic_lr = _safe_float(
            optimization.get("critic_learning_rate", np.nan), np.nan
        )
        desired_alpha_lr = _safe_float(
            optimization.get("entropy_learning_rate", FIXED_ENTROPY_LR),
            FIXED_ENTROPY_LR,
        )

        policy_config = getattr(policy, "config", None)
        if not isinstance(policy_config, dict):
            policy_config = None

        if policy_config is not None:
            if np.isfinite(desired_batch):
                policy_config["train_batch_size"] = int(round(desired_batch))
            if np.isfinite(desired_tau):
                policy_config["tau"] = float(desired_tau)
            if np.isfinite(desired_target_entropy):
                policy_config["target_entropy"] = float(desired_target_entropy)
            policy_opt = dict(policy_config.get("optimization", {}) or {})
            if np.isfinite(desired_actor_lr):
                policy_opt["actor_learning_rate"] = float(desired_actor_lr)
            if np.isfinite(desired_critic_lr):
                policy_opt["critic_learning_rate"] = float(desired_critic_lr)
            if np.isfinite(desired_alpha_lr):
                policy_opt["entropy_learning_rate"] = float(desired_alpha_lr)
            policy_config["optimization"] = policy_opt

        actor_optim = getattr(policy, "actor_optim", None)
        if actor_optim is not None and np.isfinite(desired_actor_lr):
            try:
                for group in actor_optim.param_groups:
                    group["lr"] = float(desired_actor_lr)
            except Exception:
                pass

        critic_optims = getattr(policy, "critic_optims", None) or []
        if np.isfinite(desired_critic_lr):
            for optimizer in critic_optims:
                try:
                    for group in optimizer.param_groups:
                        group["lr"] = float(desired_critic_lr)
                except Exception:
                    pass

        alpha_optim = getattr(policy, "alpha_optim", None)
        if alpha_optim is not None and np.isfinite(desired_alpha_lr):
            try:
                for group in alpha_optim.param_groups:
                    group["lr"] = float(desired_alpha_lr)
            except Exception:
                pass

        # Fallback for versions exposing only the combined optimizer list.
        all_optims = getattr(policy, "_optimizers", None) or []
        if all_optims:
            if actor_optim is None and np.isfinite(desired_actor_lr):
                try:
                    for group in all_optims[0].param_groups:
                        group["lr"] = float(desired_actor_lr)
                except Exception:
                    pass
            if alpha_optim is None and np.isfinite(desired_alpha_lr):
                try:
                    for group in all_optims[-1].param_groups:
                        group["lr"] = float(desired_alpha_lr)
                except Exception:
                    pass

        if np.isfinite(desired_target_entropy):
            self._set_target_entropy_tensor(policy, desired_target_entropy)

    def _effective_values(self, algorithm) -> dict:
        policy = self._get_default_policy(algorithm)
        if policy is None:
            return {}

        policy_config = getattr(policy, "config", {}) or {}
        if not isinstance(policy_config, dict):
            policy_config = {}

        actor_lr = np.nan
        actor_optim = getattr(policy, "actor_optim", None)
        if actor_optim is not None:
            try:
                actor_lr = _safe_float(actor_optim.param_groups[0]["lr"], np.nan)
            except Exception:
                pass

        critic_lr = np.nan
        critic_optims = getattr(policy, "critic_optims", None) or []
        if critic_optims:
            try:
                critic_lr = _safe_float(
                    critic_optims[0].param_groups[0]["lr"], np.nan
                )
            except Exception:
                pass

        alpha_lr = np.nan
        alpha_optim = getattr(policy, "alpha_optim", None)
        if alpha_optim is not None:
            try:
                alpha_lr = _safe_float(alpha_optim.param_groups[0]["lr"], np.nan)
            except Exception:
                pass

        target_entropy = np.nan
        model = getattr(policy, "model", None)
        if model is not None:
            target_entropy = _safe_float(
                getattr(model, "target_entropy", np.nan), np.nan
            )

        return {
            "train_batch_size": _safe_float(
                policy_config.get("train_batch_size", np.nan), np.nan
            ),
            "tau": _safe_float(policy_config.get("tau", np.nan), np.nan),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "alpha_lr": alpha_lr,
            "target_entropy": target_entropy,
        }

    def _audit_effective_hyperparams(self, algorithm, custom: dict) -> None:
        cfg = algorithm.config
        optimization = _optimization_dict(cfg)
        desired = {
            "train_batch_size": _safe_float(
                _config_attr(cfg, "train_batch_size", np.nan), np.nan
            ),
            "tau": _safe_float(_config_attr(cfg, "tau", np.nan), np.nan),
            "actor_lr": _safe_float(
                optimization.get("actor_learning_rate", np.nan), np.nan
            ),
            "critic_lr": _safe_float(
                optimization.get("critic_learning_rate", np.nan), np.nan
            ),
            "alpha_lr": _safe_float(
                optimization.get("entropy_learning_rate", FIXED_ENTROPY_LR),
                FIXED_ENTROPY_LR,
            ),
            "target_entropy": _safe_float(
                _config_attr(cfg, "target_entropy", np.nan), np.nan
            ),
        }
        effective = self._effective_values(algorithm)

        mismatch_count = 0
        # alpha_lr is fixed and audited, but not counted among the four outer HPs
        # plus actuator mismatch count unless finite on both sides.
        for name in (
            "train_batch_size",
            "tau",
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "target_entropy",
        ):
            d = desired.get(name, np.nan)
            e = effective.get(name, np.nan)
            custom[f"coda_desired_{name}"] = d
            custom[f"coda_effective_{name}"] = e

            if name == "train_batch_size" and np.isfinite(d) and np.isfinite(e):
                mismatch = int(round(d)) != int(round(e))
            else:
                mismatch = bool(
                    np.isfinite(d)
                    and np.isfinite(e)
                    and not np.isclose(d, e, rtol=1e-7, atol=1e-12)
                )
            custom[f"coda_mismatch_{name}"] = float(mismatch)
            mismatch_count += int(mismatch)

        custom["coda_effective_hp_mismatch_count"] = float(mismatch_count)

    def _apply_lineage_seed_if_needed(self, algorithm) -> None:
        bridge = self._bridge_from_algorithm(algorithm)
        generation = bridge.get("lineage_generation", None)
        if generation is None:
            return
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            return
        if generation == self._last_lineage_generation:
            return

        state_seed = _safe_float(bridge.get("lineage_ema_seed", np.nan), np.nan)
        td_seed = _safe_float(
            bridge.get("lineage_td_reference_seed", np.nan), np.nan
        )
        self._policy_state_ema = float(state_seed) if np.isfinite(state_seed) else None
        self._td_reference_ema = float(td_seed) if np.isfinite(td_seed) else None
        self._last_lineage_generation = generation

    def on_checkpoint_loaded(self, *, algorithm, **kwargs):
        self._sync_effective_hyperparams(algorithm)
        self._apply_lineage_seed_if_needed(algorithm)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        self._sync_effective_hyperparams(algorithm)
        self._apply_lineage_seed_if_needed(algorithm)

        custom = result.setdefault("custom_metrics", {})
        self._audit_effective_hyperparams(algorithm, custom)
        bridge = self._bridge_from_algorithm(algorithm)

        # Scheduler / GP / O2I diagnostics.
        custom["coda_guided_update"] = float(bool(bridge.get("guided_update", False)))
        custom["coda_gp_data_count"] = _safe_float(
            bridge.get("gp_data_count", 0.0), 0.0
        )
        for source_name, output_name in (
            ("o2i_uncertainty_raw_std", "coda_o2i_uncertainty_raw_std"),
            ("o2i_uncertainty_prior_std", "coda_o2i_uncertainty_prior_std"),
            ("o2i_uncertainty_normalized", "coda_o2i_uncertainty_normalized"),
            ("o2i_kernel_variance", "coda_o2i_kernel_variance"),
            ("o2i_candidate_posterior_std", "coda_o2i_candidate_posterior_std"),
            ("o2i_donor_posterior_std", "coda_o2i_donor_posterior_std"),
            ("o2i_candidate_prior_std", "coda_o2i_candidate_prior_std"),
            ("o2i_donor_prior_std", "coda_o2i_donor_prior_std"),
            ("o2i_incremental_uncertainty", "coda_o2i_incremental_uncertainty"),
            ("o2i_donor_kernel_variance", "coda_o2i_donor_kernel_variance"),
            ("gp_kernel_variance_parameter", "coda_gp_kernel_variance_parameter"),
            ("gp_lengthscale", "coda_gp_lengthscale"),
            ("gp_temporal_epsilon", "coda_gp_temporal_epsilon"),
            ("gp_alpha_used", "coda_gp_alpha_used"),
            ("o2i_acquisition_value", "coda_o2i_acquisition_value"),
        ):
            custom[output_name] = _safe_float(
                bridge.get(source_name, np.nan if "gp_" in output_name else 0.0),
                np.nan if "gp_" in output_name else 0.0,
            )

        # Old bridge field names are kept by the scheduler for compatibility;
        # the SAC outputs below give them unambiguous semantics.
        custom["coda_o2i_reference_target_entropy"] = _safe_float(
            bridge.get("o2i_reference_entropy_coeff", np.nan), np.nan
        )
        custom["coda_target_entropy_increment"] = _safe_float(
            bridge.get("entropy_increment", 0.0), 0.0
        )
        custom["coda_base_target_entropy"] = _safe_float(
            bridge.get("base_entropy_coeff", np.nan), np.nan
        )
        custom["coda_applied_target_entropy"] = _safe_float(
            bridge.get("applied_entropy_coeff", np.nan), np.nan
        )

        policy_data, learner_stats = self._extract_policy_and_learner_stats(result)

        # TD error is an extra learner fetch in old-stack SAC. Be permissive in
        # where we read it because Result formatting changed across Ray releases.
        td_value = None
        for candidate in (
            policy_data.get("td_error"),
            learner_stats.get("td_error"),
            result.get("td_error"),
            result.get("info/learner/default_policy/td_error"),
        ):
            arr = _finite_array(candidate)
            if arr.size:
                td_value = arr
                break

        if td_value is None:
            td_errors = np.empty((0,), dtype=np.float64)
        else:
            td_errors = td_value

        if td_errors.size:
            td_median = float(np.median(td_errors))
            td_p95 = float(np.percentile(td_errors, 95.0))
            td_reference_before = (
                float(self._td_reference_ema)
                if self._td_reference_ema is not None
                else np.nan
            )

            if self._td_reference_ema is None:
                state_health = 1.0
                state_raw = 1.0
                self._td_reference_ema = td_median
            else:
                ratio = td_median / (float(self._td_reference_ema) + TD_REFERENCE_EPS)
                state_health = float(np.exp(-max(0.0, ratio - 1.0)))
                state_raw = float(
                    np.clip(state_health, POLICY_STATE_MIN, 1.0)
                )

            if self._policy_state_ema is None:
                self._policy_state_ema = state_raw
            else:
                self._policy_state_ema = float(
                    POLICY_STATE_EMA_BETA * self._policy_state_ema
                    + (1.0 - POLICY_STATE_EMA_BETA) * state_raw
                )

            state = float(
                np.clip(self._policy_state_ema, POLICY_STATE_MIN, 1.0)
            )

            # Update the TD reference AFTER evaluating the current state so the
            # current observation does not normalize itself.
            if np.isfinite(td_reference_before):
                self._td_reference_ema = float(
                    TD_REFERENCE_EMA_BETA * float(self._td_reference_ema)
                    + (1.0 - TD_REFERENCE_EMA_BETA) * td_median
                )

            valid = True
        else:
            td_median = np.nan
            td_p95 = np.nan
            td_reference_before = np.nan
            state_health = np.nan
            state_raw = np.nan
            state = np.nan
            valid = False

        custom["sac_td_error_median"] = td_median
        custom["sac_td_error_p95"] = td_p95
        custom["sac_td_error_reference_before"] = td_reference_before
        custom["sac_td_error_reference_ema"] = (
            float(self._td_reference_ema)
            if self._td_reference_ema is not None
            else np.nan
        )
        custom["policy_update_state"] = state
        custom["policy_update_state_raw"] = state_raw
        custom["policy_update_state_valid"] = float(valid)
        custom["policy_update_health"] = state_health

        # SAC learner diagnostics (not used directly by the scheduler).
        for key in (
            "actor_loss",
            "critic_loss",
            "alpha_loss",
            "alpha_value",
            "log_alpha_value",
            "target_entropy",
            "mean_q",
            "max_q",
            "min_q",
        ):
            custom[f"sac_{key}"] = _safe_float(
                learner_stats.get(key, np.nan), np.nan
            )

        self._iter_count += 1
        if self._iter_count % 10 == 0:
            logger.info(
                "CODA-SAC | TDmed=%.5g TDref=%.5g S=%.4f U_delta=%.4f "
                "Htarget=%.4f alpha=%.4g mismatch=%d",
                td_median,
                custom["sac_td_error_reference_ema"],
                state,
                custom["coda_o2i_incremental_uncertainty"],
                custom.get("coda_effective_target_entropy", np.nan),
                custom.get("sac_alpha_value", np.nan),
                int(custom.get("coda_effective_hp_mismatch_count", 0.0)),
            )


# =============================================================================
# SAC CONFIGURATION
# =============================================================================

# The shared SAC learner and Tune initialization distributions are built by
# configs.sac_config.build_tunable_sac_config(). CODA-specific I2O/O2I logic,
# lineage transfer, restore synchronization, and scheduler wiring remain here.


# =============================================================================
# OUTPUT / AUDIT HELPERS
# =============================================================================


def _extract_metrics(
    result_grid,
    env_name: str,
    seed: int,
    output_path: Path,
) -> None:
    frames = []

    columns_wanted = [
        "time_total_s",
        "training_iteration",
        "timesteps_total",
        "episodes_total",
        "env_runners/episode_return_mean",
        "env_runners/episode_return_max",
        "env_runners/episode_return_min",
        "env_runners/episode_len_mean",
        "config/train_batch_size",
        "config/tau",
        "config/target_entropy",
        "config/gamma",
        "config/optimization/actor_learning_rate",
        "config/optimization/critic_learning_rate",
        "config/optimization/entropy_learning_rate",
        "info/learner/default_policy/learner_stats/actor_loss",
        "info/learner/default_policy/learner_stats/critic_loss",
        "info/learner/default_policy/learner_stats/alpha_loss",
        "info/learner/default_policy/learner_stats/alpha_value",
        "info/learner/default_policy/learner_stats/log_alpha_value",
        "info/learner/default_policy/learner_stats/target_entropy",
        "info/learner/default_policy/learner_stats/mean_q",
        "info/learner/default_policy/learner_stats/max_q",
        "info/learner/default_policy/learner_stats/min_q",
        "custom_metrics/sac_td_error_median",
        "custom_metrics/sac_td_error_p95",
        "custom_metrics/sac_td_error_reference_before",
        "custom_metrics/sac_td_error_reference_ema",
        "custom_metrics/policy_update_state",
        "custom_metrics/policy_update_state_raw",
        "custom_metrics/policy_update_state_valid",
        "custom_metrics/policy_update_health",
        "custom_metrics/sac_actor_loss",
        "custom_metrics/sac_critic_loss",
        "custom_metrics/sac_alpha_loss",
        "custom_metrics/sac_alpha_value",
        "custom_metrics/sac_log_alpha_value",
        "custom_metrics/sac_target_entropy",
        "custom_metrics/sac_mean_q",
        "custom_metrics/sac_max_q",
        "custom_metrics/sac_min_q",
        "custom_metrics/coda_guided_update",
        "custom_metrics/coda_gp_data_count",
        "custom_metrics/coda_o2i_uncertainty_raw_std",
        "custom_metrics/coda_o2i_uncertainty_prior_std",
        "custom_metrics/coda_o2i_uncertainty_normalized",
        "custom_metrics/coda_o2i_kernel_variance",
        "custom_metrics/coda_o2i_candidate_posterior_std",
        "custom_metrics/coda_o2i_donor_posterior_std",
        "custom_metrics/coda_o2i_candidate_prior_std",
        "custom_metrics/coda_o2i_donor_prior_std",
        "custom_metrics/coda_o2i_incremental_uncertainty",
        "custom_metrics/coda_o2i_donor_kernel_variance",
        "custom_metrics/coda_gp_kernel_variance_parameter",
        "custom_metrics/coda_gp_lengthscale",
        "custom_metrics/coda_gp_temporal_epsilon",
        "custom_metrics/coda_gp_alpha_used",
        "custom_metrics/coda_o2i_acquisition_value",
        "custom_metrics/coda_o2i_reference_target_entropy",
        "custom_metrics/coda_target_entropy_increment",
        "custom_metrics/coda_base_target_entropy",
        "custom_metrics/coda_applied_target_entropy",
        "custom_metrics/coda_desired_train_batch_size",
        "custom_metrics/coda_effective_train_batch_size",
        "custom_metrics/coda_mismatch_train_batch_size",
        "custom_metrics/coda_desired_tau",
        "custom_metrics/coda_effective_tau",
        "custom_metrics/coda_mismatch_tau",
        "custom_metrics/coda_desired_actor_lr",
        "custom_metrics/coda_effective_actor_lr",
        "custom_metrics/coda_mismatch_actor_lr",
        "custom_metrics/coda_desired_critic_lr",
        "custom_metrics/coda_effective_critic_lr",
        "custom_metrics/coda_mismatch_critic_lr",
        "custom_metrics/coda_desired_alpha_lr",
        "custom_metrics/coda_effective_alpha_lr",
        "custom_metrics/coda_mismatch_alpha_lr",
        "custom_metrics/coda_desired_target_entropy",
        "custom_metrics/coda_effective_target_entropy",
        "custom_metrics/coda_mismatch_target_entropy",
        "custom_metrics/coda_effective_hp_mismatch_count",
        "perf/gpu_util_percent0",
        "perf/cpu_util_percent",
        "perf/ram_util_percent",
        "timers/training_iteration_time_ms",
    ]

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe
        if df is None or df.empty:
            continue
        present = [column for column in columns_wanted if column in df.columns]
        out = df[present].copy()
        out["causal_order"] = np.arange(len(out), dtype=np.int64)
        out["entorno"] = env_name
        out["semilla"] = int(seed)
        out["agente_id"] = f"Agente_{idx + 1}"
        frames.append(out)

    if frames:
        final = pd.concat(frames, ignore_index=True)
        final = final.sort_values(["agente_id", "causal_order"], kind="stable")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_path, index=False)


def _save_trial_status_summary(
    result_grid,
    env_name: str,
    seed: int,
    target_steps: int,
    output_path: Path,
) -> None:
    rows = []
    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe
        n_rows = int(len(df)) if df is not None else 0
        final_steps = np.nan
        last_finite_reward = np.nan

        if df is not None and not df.empty:
            if TIME_ATTR in df.columns:
                steps = pd.to_numeric(df[TIME_ATTR], errors="coerce")
                finite_steps = steps[np.isfinite(steps)]
                if not finite_steps.empty:
                    final_steps = float(finite_steps.iloc[-1])
            if METRIC in df.columns:
                rewards = pd.to_numeric(df[METRIC], errors="coerce")
                finite_rewards = rewards[np.isfinite(rewards)]
                if not finite_rewards.empty:
                    last_finite_reward = float(finite_rewards.iloc[-1])

        error = getattr(result, "error", None)
        rows.append(
            {
                "environment": env_name,
                "training_seed": int(seed),
                "agent_id": f"Agente_{idx + 1}",
                "n_metric_rows": n_rows,
                "final_timesteps_total": final_steps,
                "reached_target_steps": int(
                    np.isfinite(final_steps) and final_steps >= target_steps
                ),
                "last_finite_training_return": last_finite_reward,
                "has_checkpoint": int(bool(result.checkpoint)),
                "checkpoint_path": (
                    str(result.checkpoint.path) if result.checkpoint else ""
                ),
                "result_path": str(getattr(result, "path", "")),
                "error": repr(error) if error is not None else "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _validate_result_grid(result_grid, *, target_steps: int) -> None:
    """Reject incomplete P=4 population runs immediately."""
    results = list(result_grid)
    problems = []
    if len(results) != POBLACION_B:
        problems.append(
            f"expected {POBLACION_B} trials, ResultGrid contains {len(results)}"
        )

    for idx, result in enumerate(results):
        agent = f"Agente_{idx + 1}"
        error = getattr(result, "error", None)
        if error is not None:
            problems.append(f"{agent}: Tune/RLlib error={error!r}")

        df = result.metrics_dataframe
        if df is None or df.empty:
            problems.append(f"{agent}: empty metrics dataframe")
            continue

        if TIME_ATTR not in df.columns:
            problems.append(f"{agent}: missing {TIME_ATTR}")
        else:
            steps = pd.to_numeric(df[TIME_ATTR], errors="coerce")
            finite = steps[np.isfinite(steps)]
            terminal = float(finite.iloc[-1]) if not finite.empty else np.nan
            if not np.isfinite(terminal) or terminal < target_steps:
                problems.append(
                    f"{agent}: terminal {TIME_ATTR}={terminal}, expected >= {target_steps}"
                )

        if not result.checkpoint:
            problems.append(f"{agent}: missing final checkpoint")

        mismatch_col = (
            "custom_metrics/"
            "coda_effective_hp_mismatch_count"
        )
        if mismatch_col in df.columns:
            mismatch = pd.to_numeric(
                df[mismatch_col],
                errors="coerce",
            )
            finite_mismatch = mismatch[
                np.isfinite(mismatch)
            ]
            if (
                not finite_mismatch.empty
                and (finite_mismatch != 0).any()
            ):
                problems.append(
                    f"{agent}: non-zero restore-time "
                    "CODA-SAC hyperparameter mismatch"
                )

    if problems:
        raise RuntimeError(
            "Incomplete CODA-SAC population run:\n  - " + "\n  - ".join(problems)
        )


def _copy_final_checkpoints(result_grid, env_name: str, seed: int) -> None:
    root = (RESULTS_ROOT / "champions") / env_name / f"{OUTPUT_NAME}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(result_grid):
        if not result.checkpoint:
            logger.warning("No final checkpoint for Agente_%s", idx + 1)
            continue
        source = Path(result.checkpoint.path)
        target = root / f"Agente_{idx + 1}"
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        except Exception as exc:
            logger.warning("Could not copy checkpoint %s -> %s: %s", source, target, exc)


def _save_metadata(
    path: Path,
    *,
    env_name: str,
    seed: int,
    target_steps: int,
    perturbation_interval: int,
    action_dim: int,
    base_target_entropy: float,
    max_target_entropy_increment: float,
    actuator_bounds: Tuple[float, float],
) -> None:
    payload = {
        "algorithm": ALGO_NAME,
        "run_name": OUTPUT_NAME,
        "variant": VARIANT,
        "environment": env_name,
        "seed": int(seed),
        "population": POBLACION_B,
        "max_timesteps_per_worker": int(target_steps),
        "perturbation_interval": int(perturbation_interval),
        "quantile_fraction": QUANTILE_FRACTION,
        "optimized_hyperparameters": [
            "train_batch_size",
            "tau",
            "optimization/actor_learning_rate",
            "optimization/critic_learning_rate",
        ],
        "hyperparameter_bounds": HYPERPARAM_BOUNDS,
        "configuration_source": {
            "environments": "configs/environments.py",
            "seeds": "configs/seeds.py",
            "sac": "configs/sac_config.py",
            "scheduler": "src/coda/schedulers/coda_sac_scheduler.py",
        },
        "log10_model_coordinates": list(CODA_LOG10_COORDINATES),
        "legacy_log10_model_coordinates_reference": [
            "tau",
            "optimization/actor_learning_rate",
            "optimization/critic_learning_rate",
        ],
        "i2o": {
            "source": "median absolute SAC TD error from old-stack learner fetch",
            "td_reference_ema_beta": TD_REFERENCE_EMA_BETA,
            "state_mapping": "exp(-max(0, TDmedian/(TDref+eps)-1))",
            "state_ema_beta": POLICY_STATE_EMA_BETA,
            "state_lower_bound": POLICY_STATE_MIN,
            "td_eps": TD_REFERENCE_EPS,
            "lineage_reference_transfer": True,
        },
        "o2i": {
            "signal": "donor-relative incremental observed-GP uncertainty",
            "actuator": "SAC target_entropy",
            "action_dim": int(action_dim),
            "base_target_entropy": float(base_target_entropy),
            "max_increment": float(max_target_entropy_increment),
            "actuator_bounds": list(map(float, actuator_bounds)),
            "mapping": "base + max_increment * U_delta, clipped to actuator bounds",
            "alpha_remains_internally_optimized": True,
            "entropy_learning_rate": FIXED_ENTROPY_LR,
        },
        "sac": {
            "gamma": FIXED_GAMMA,
            "initial_alpha": INITIAL_ALPHA,
            "n_step": N_STEP,
            "target_network_update_freq": TARGET_NETWORK_UPDATE_FREQ,
            "learning_starts": LEARNING_STARTS,
            "training_intensity": TRAINING_INTENSITY,
            "twin_q": TWIN_Q,
            "replay_buffer_type": "MultiAgentPrioritizedReplayBuffer",
            "replay_buffer_capacity": REPLAY_BUFFER_CAPACITY,
            "store_buffer_in_checkpoints": STORE_BUFFER_IN_CHECKPOINTS,
            "model_hiddens": MODEL_HIDDENS,
            "model_activation": MODEL_ACTIVATION,
            "num_env_runners": NUM_ENV_RUNNERS,
            "num_envs_per_env_runner": NUM_ENVS_PER_RUNNER,
            "rollout_fragment_length": ROLLOUT_FRAGMENT_LENGTH,
            "observation_filter": OBSERVATION_FILTER,
            "num_gpus_per_trial": NUM_GPUS_PER_TRIAL,
            "old_api_stack": True,
        },
        "surrogate": {
            "min_valid_transitions": MIN_VALID_TRANSITIONS,
            "max_gp_points": MAX_GP_POINTS,
            "reward_z_clip": REWARD_Z_CLIP,
            "ucb": "Ray/PB2-compatible mean + kappa * predictive_variance",
        },
        "checkpointing": {
            "checkpoint_at_end": True,
            "store_buffer_in_checkpoints": STORE_BUFFER_IN_CHECKPOINTS,
            "restore_hyperparameter_resynchronization": True,
        },
        "versions": {
            "python": sys.version,
            "ray": ray.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": _package_version("scikit-learn"),
            "scipy": _package_version("scipy"),
            "gymnasium": _package_version("gymnasium"),
            "mujoco": _package_version("mujoco"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_experiment(
    env_name: str,
    seed: int,
    *,
    target_steps: int,
    perturbation_interval: int,
    quantile_fraction: float,
) -> bool:
    print("\n" + "=" * 78)
    print(
        f"Starting {OUTPUT_NAME} | env={env_name} | seed={seed} | "
        f"population={POBLACION_B} | steps/worker={target_steps:,}"
    )
    print("=" * 78)

    _seed_everything(seed)

    action_dim, base_target_entropy, max_increment, actuator_bounds = (
        _target_entropy_parameters(env_name)
    )

    context_bounds = {
        "T_before": (0.0, float(target_steps)),
        "S_before": (0.0, 1.0),
    }

    scheduler = CODASACScheduler(
        time_attr=TIME_ATTR,
        perturbation_interval=int(perturbation_interval),
        quantile_fraction=float(quantile_fraction),
        hyperparam_bounds=HYPERPARAM_BOUNDS,
        context_bounds=context_bounds,
        variant=VARIANT,
        min_valid_transitions=MIN_VALID_TRANSITIONS,
        max_gp_points=MAX_GP_POINTS,
        entropy_param="target_entropy",
        base_entropy_coeff=base_target_entropy,
        o2i_uncertainty_scale=max_increment,
        max_entropy_increment=max_increment,
        actuator_bounds=actuator_bounds,
        reward_z_clip=REWARD_Z_CLIP,
        synch=False,
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required because each SAC trial requests "
                f"num_gpus={NUM_GPUS_PER_TRIAL}."
            )

        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
            include_dashboard=False,
        )

        sac_config = build_tunable_sac_config(
            env_name,
            seed,
            target_entropy=base_target_entropy,
            callbacks=CODASACTDCallback,
        )

        storage_root = (RESULTS_ROOT / "ray_tune_logs" / OUTPUT_NAME).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)

        tuner = tune.Tuner(
            "SAC",
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=POBLACION_B,
                metric=METRIC,
                mode="max",
                trial_name_creator=lambda trial: (
                    f"{OUTPUT_NAME}_{env_name}_{seed}_{trial.trial_id}"
                ),
                trial_dirname_creator=lambda trial: (
                    f"{OUTPUT_NAME}_{env_name}_{seed}_{trial.trial_id}"
                ),
            ),
            param_space=sac_config.to_dict(),
            run_config=tune.RunConfig(
                name=f"{OUTPUT_NAME}_{env_name}_Seed{seed}",
                verbose=0,
                storage_path=str(storage_root),
                stop={TIME_ATTR: int(target_steps)},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_at_end=True,
                ),
            ),
        )

        results = tuner.fit()

        # Save everything first, even if validation subsequently rejects the
        # population. This preserves diagnostics for failed runs.
        metrics_path = (
            (RESULTS_ROOT / "metrics")
            / env_name
            / f"metrics_{OUTPUT_NAME}_seed{seed}.csv"
        )
        _extract_metrics(results, env_name, seed, metrics_path)

        scheduler_path = (
            (RESULTS_ROOT / "scheduler")
            / env_name
            / f"scheduler_{OUTPUT_NAME}_seed{seed}.csv"
        )
        scheduler_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler.data.to_csv(scheduler_path, index=False)

        status_path = (
            (RESULTS_ROOT / "status")
            / env_name
            / f"status_{OUTPUT_NAME}_seed{seed}.csv"
        )
        _save_trial_status_summary(
            results, env_name, seed, target_steps, status_path
        )

        metadata_path = (
            (RESULTS_ROOT / "metadata")
            / env_name
            / f"metadata_{OUTPUT_NAME}_seed{seed}.json"
        )
        _save_metadata(
            metadata_path,
            env_name=env_name,
            seed=seed,
            target_steps=target_steps,
            perturbation_interval=perturbation_interval,
            action_dim=action_dim,
            base_target_entropy=base_target_entropy,
            max_target_entropy_increment=max_increment,
            actuator_bounds=actuator_bounds,
        )

        _copy_final_checkpoints(results, env_name, seed)

        # Strong run-integrity gate. Failed population seeds must be rerun as a
        # whole population and may not be reduced to surviving workers.
        _validate_result_grid(results, target_steps=target_steps)

        print(
            f"Completed {OUTPUT_NAME}: {env_name}, seed={seed} | "
            f"target_entropy in [{actuator_bounds[0]:.3f}, {actuator_bounds[1]:.3f}]"
        )
        return True

    except Exception as exc:
        print(f"\nCRITICAL ERROR in {env_name}, seed={seed}: {exc}")
        traceback.print_exc()
        return False

    finally:
        if ray.is_initialized():
            ray.shutdown()
        time.sleep(2)


def main() -> None:
    started = time.time()
    failures = []

    if SAC_SMOKE_TEST:
        if SAC_SMOKE_ENV not in ENVIRONMENT_CONFIG:
            raise KeyError(f"Unknown smoke environment: {SAC_SMOKE_ENV}")
        experiment_items = [(SAC_SMOKE_ENV, ENVIRONMENT_CONFIG[SAC_SMOKE_ENV])]
        total = len(SAC_SMOKE_SEEDS)
    else:
        experiment_items = list(ENVIRONMENT_CONFIG.items())
        total = sum(len(cfg["semillas"]) for _, cfg in experiment_items)

    done = 0
    for env_name, env_cfg in experiment_items:
        if SAC_SMOKE_TEST:
            seeds: Sequence[int] = SAC_SMOKE_SEEDS
            target_steps = SAC_SMOKE_STEPS
            interval = SAC_SMOKE_INTERVAL
        else:
            seeds = env_cfg["semillas"]
            target_steps = TIMESTEPS_MAX
            interval = int(env_cfg.get("perturbation_interval", PERTURBATION_INTERVAL))

        quantile = float(env_cfg.get("quantile_fraction", QUANTILE_FRACTION))

        for seed in seeds:
            ok = run_experiment(
                env_name,
                int(seed),
                target_steps=int(target_steps),
                perturbation_interval=int(interval),
                quantile_fraction=quantile,
            )
            if not ok:
                failures.append(f"{env_name} - seed {seed}")
            done += 1
            print(f"Global progress: {done}/{total}")

    hours = (time.time() - started) / 3600.0
    print("\n" + "-" * 78)
    print(f"Experiments finished in {hours:.2f} hours")
    if failures:
        print(f"Failures ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("All requested experiments completed successfully.")
    print("-" * 78)


if __name__ == "__main__":
    main()
