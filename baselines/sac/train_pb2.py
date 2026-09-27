#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PB2 + SAC baseline using the shared repository configuration.

This baseline is designed for a fair comparison against the current
CODA+SAC implementation. It keeps the SAC learner, raw search domain,
initial sampling distributions, population size, interaction horizon,
adaptation interval, replay buffer, model architecture, sampling setup,
training intensity, resource allocation, checkpoint semantics, and
restore-time hyperparameter synchronization aligned with CODA+SAC.

The intended algorithmic differences are only:

1. PB2 retains population-based checkpoint inheritance but selects new
   hyperparameters with Ray Tune's native time-varying GP/PB2 scheduler.
2. PB2 does NOT use CODA's TD-error I2O diagnostic state as GP context.
3. PB2 does NOT use CODA's donor-relative uncertainty O2I actuator.
4. SAC target entropy remains fixed at its baseline value -action_dim.
5. PB2 uses its native raw-coordinate GP geometry; CODA's log10 surrogate
   representation is not injected into this external baseline.

Old RLlib API stack is used intentionally to match CODA+SAC.

All common learner/search/budget settings are imported from ``configs/``.
PB2-specific scheduling, executable-proposal conversion, restore-time
synchronization, and auditing remain local to this script.
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
from ray.tune.schedulers.pb2 import PB2
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.backends.cudnn as cudnn


warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


# =============================================================================
# REPOSITORY / SHARED CONFIGURATION
# =============================================================================

# Intended location:
#     <repo>/experiments/sac/train_pb2.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    build_tunable_sac_config,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


# =============================================================================
# METHOD-SPECIFIC CONFIGURATION
# =============================================================================

ALGO_NAME = "PB2_SAC"

# Kept for backward-compatible names used by the current evaluation scripts.
RUN_TAG = os.environ.get("PB2_SAC_RUN_TAG", "PB2SPACE").strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Compatibility aliases only; all values come from configs/sac_config.py.
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

# Default to smoke mode. Set PB2_SAC_SMOKE=0 for the full configured campaign.
PB2_SAC_SMOKE_TEST = os.environ.get("PB2_SAC_SMOKE", "1") != "0"
PB2_SAC_SMOKE_ENV = os.environ.get("PB2_SAC_SMOKE_ENV", "Hopper-v5")
PB2_SAC_SMOKE_SEEDS = [
    int(os.environ.get("PB2_SAC_SMOKE_SEED", "1042"))
]
PB2_SAC_SMOKE_STEPS = int(
    os.environ.get("PB2_SAC_SMOKE_STEPS", "500000")
)
PB2_SAC_SMOKE_INTERVAL = int(
    os.environ.get("PB2_SAC_SMOKE_INTERVAL", "50000")
)


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
# PB2-SAC CALLBACK: RESTORE SYNC + DIAGNOSTICS ONLY
# =============================================================================

class PB2SACDiagnosticsCallback(DefaultCallbacks):
    """Diagnostics and restore-time synchronization for the PB2-SAC baseline.

    No diagnostic quantity is fed back into PB2.

    The synchronization is necessary because, on RLlib's old SAC stack,
    restoring a donor checkpoint may restore donor-side policy config and
    optimizer state after Tune/PB2 has already installed the receiver's
    GP-selected configuration.
    """

    def __init__(self):
        super().__init__()
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
    def _extract_policy_and_learner_stats(
        result: dict,
    ) -> Tuple[dict, dict]:
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
        """Restore the fixed SAC target entropy on the live model/towers."""
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
        """Re-apply current PB2/Tune values after donor restoration."""
        policy = self._get_default_policy(algorithm)
        if policy is None:
            return

        cfg = algorithm.config
        optimization = _optimization_dict(cfg)

        desired_batch = _safe_float(
            _config_attr(cfg, "train_batch_size", np.nan),
            np.nan,
        )
        desired_tau = _safe_float(
            _config_attr(cfg, "tau", np.nan),
            np.nan,
        )
        desired_target_entropy = _safe_float(
            _config_attr(cfg, "target_entropy", np.nan),
            np.nan,
        )
        desired_actor_lr = _safe_float(
            optimization.get("actor_learning_rate", np.nan),
            np.nan,
        )
        desired_critic_lr = _safe_float(
            optimization.get("critic_learning_rate", np.nan),
            np.nan,
        )
        desired_alpha_lr = _safe_float(
            optimization.get(
                "entropy_learning_rate",
                FIXED_ENTROPY_LR,
            ),
            FIXED_ENTROPY_LR,
        )

        policy_config = getattr(policy, "config", None)
        if not isinstance(policy_config, dict):
            policy_config = None

        if policy_config is not None:
            if np.isfinite(desired_batch):
                policy_config["train_batch_size"] = int(
                    round(desired_batch)
                )

            if np.isfinite(desired_tau):
                policy_config["tau"] = float(desired_tau)

            if np.isfinite(desired_target_entropy):
                policy_config["target_entropy"] = float(
                    desired_target_entropy
                )

            policy_opt = dict(
                policy_config.get("optimization", {}) or {}
            )

            if np.isfinite(desired_actor_lr):
                policy_opt["actor_learning_rate"] = float(
                    desired_actor_lr
                )

            if np.isfinite(desired_critic_lr):
                policy_opt["critic_learning_rate"] = float(
                    desired_critic_lr
                )

            if np.isfinite(desired_alpha_lr):
                policy_opt["entropy_learning_rate"] = float(
                    desired_alpha_lr
                )

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

        # Fallback for Ray versions exposing only a combined optimizer list.
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
            self._set_target_entropy_tensor(
                policy,
                desired_target_entropy,
            )

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
                actor_lr = _safe_float(
                    actor_optim.param_groups[0]["lr"],
                    np.nan,
                )
            except Exception:
                pass

        critic_lr = np.nan
        critic_optims = getattr(policy, "critic_optims", None) or []
        if critic_optims:
            try:
                critic_lr = _safe_float(
                    critic_optims[0].param_groups[0]["lr"],
                    np.nan,
                )
            except Exception:
                pass

        alpha_lr = np.nan
        alpha_optim = getattr(policy, "alpha_optim", None)
        if alpha_optim is not None:
            try:
                alpha_lr = _safe_float(
                    alpha_optim.param_groups[0]["lr"],
                    np.nan,
                )
            except Exception:
                pass

        target_entropy = np.nan
        model = getattr(policy, "model", None)
        if model is not None:
            target_entropy = _safe_float(
                getattr(model, "target_entropy", np.nan),
                np.nan,
            )

        return {
            "train_batch_size": _safe_float(
                policy_config.get("train_batch_size", np.nan),
                np.nan,
            ),
            "tau": _safe_float(
                policy_config.get("tau", np.nan),
                np.nan,
            ),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "alpha_lr": alpha_lr,
            "target_entropy": target_entropy,
        }

    def _audit_effective_hyperparams(
        self,
        algorithm,
        custom: dict,
    ) -> None:
        cfg = algorithm.config
        optimization = _optimization_dict(cfg)

        desired = {
            "train_batch_size": _safe_float(
                _config_attr(cfg, "train_batch_size", np.nan),
                np.nan,
            ),
            "tau": _safe_float(
                _config_attr(cfg, "tau", np.nan),
                np.nan,
            ),
            "actor_lr": _safe_float(
                optimization.get(
                    "actor_learning_rate",
                    np.nan,
                ),
                np.nan,
            ),
            "critic_lr": _safe_float(
                optimization.get(
                    "critic_learning_rate",
                    np.nan,
                ),
                np.nan,
            ),
            "alpha_lr": _safe_float(
                optimization.get(
                    "entropy_learning_rate",
                    FIXED_ENTROPY_LR,
                ),
                FIXED_ENTROPY_LR,
            ),
            "target_entropy": _safe_float(
                _config_attr(
                    cfg,
                    "target_entropy",
                    np.nan,
                ),
                np.nan,
            ),
        }

        effective = self._effective_values(algorithm)

        mismatch_count = 0

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

            custom[f"pb2_sac_desired_{name}"] = d
            custom[f"pb2_sac_effective_{name}"] = e

            if (
                name == "train_batch_size"
                and np.isfinite(d)
                and np.isfinite(e)
            ):
                mismatch = int(round(d)) != int(round(e))
            else:
                mismatch = bool(
                    np.isfinite(d)
                    and np.isfinite(e)
                    and not np.isclose(
                        d,
                        e,
                        rtol=1e-7,
                        atol=1e-12,
                    )
                )

            custom[f"pb2_sac_mismatch_{name}"] = float(
                mismatch
            )
            mismatch_count += int(mismatch)

        custom[
            "pb2_sac_effective_hp_mismatch_count"
        ] = float(mismatch_count)

    def on_checkpoint_loaded(
        self,
        *,
        algorithm,
        **kwargs,
    ):
        # Donor training state is restored first; then the PB2-selected
        # receiver configuration is re-installed on the live SAC policy.
        self._sync_effective_hyperparams(algorithm)

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs,
    ):
        # Defensive re-sync also covers Ray versions with different
        # callback ordering around PB2 checkpoint restoration.
        self._sync_effective_hyperparams(algorithm)

        custom = result.setdefault("custom_metrics", {})
        self._audit_effective_hyperparams(
            algorithm,
            custom,
        )

        policy_data, learner_stats = (
            self._extract_policy_and_learner_stats(result)
        )

        # TD error is logged for offline comparability only.
        td_value = None
        for candidate in (
            policy_data.get("td_error"),
            learner_stats.get("td_error"),
            result.get("td_error"),
            result.get(
                "info/learner/default_policy/td_error"
            ),
        ):
            arr = _finite_array(candidate)
            if arr.size:
                td_value = arr
                break

        if td_value is None:
            td_errors = np.empty(
                (0,),
                dtype=np.float64,
            )
        else:
            td_errors = td_value

        if td_errors.size:
            custom["sac_td_error_median"] = float(
                np.median(td_errors)
            )
            custom["sac_td_error_p95"] = float(
                np.percentile(td_errors, 95.0)
            )
            custom["sac_td_diagnostics_valid"] = 1.0
        else:
            custom["sac_td_error_median"] = np.nan
            custom["sac_td_error_p95"] = np.nan
            custom["sac_td_diagnostics_valid"] = 0.0

        # SAC learner diagnostics: logged but NEVER used by PB2.
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
                learner_stats.get(key, np.nan),
                np.nan,
            )

        self._iter_count += 1

        if self._iter_count % 10 == 0:
            try:
                cfg = algorithm.config
                opt = _optimization_dict(cfg)
                logger.info(
                    "PB2-SAC | batch=%s tau=%.5g "
                    "actor_lr=%.5g critic_lr=%.5g "
                    "target_entropy=%.4f mismatch=%d",
                    int(
                        round(
                            _safe_float(
                                _config_attr(
                                    cfg,
                                    "train_batch_size",
                                    np.nan,
                                )
                            )
                        )
                    ),
                    _safe_float(
                        _config_attr(
                            cfg,
                            "tau",
                            np.nan,
                        )
                    ),
                    _safe_float(
                        opt.get(
                            "actor_learning_rate",
                            np.nan,
                        )
                    ),
                    _safe_float(
                        opt.get(
                            "critic_learning_rate",
                            np.nan,
                        )
                    ),
                    _safe_float(
                        _config_attr(
                            cfg,
                            "target_entropy",
                            np.nan,
                        )
                    ),
                    int(
                        custom.get(
                            "pb2_sac_effective_hp_mismatch_count",
                            0.0,
                        )
                    ),
                )
            except Exception:
                pass


# =============================================================================
# PB2 EXECUTABLE-CONFIG GUARD
# =============================================================================

def _make_pb2_explore_guard(base_target_entropy: float):
    """Return a native-PB2 post-processing guard aligned with CODA+SAC.

    Ray PB2 proposes the four search coordinates using its own TV-GP model.
    This function does not alter PB2's acquisition rule or surrogate geometry;
    it only converts the continuous proposal into a valid executable SAC
    configuration and reasserts fixed controls after checkpoint inheritance.
    """

    batch_lo, batch_hi = HYPERPARAM_BOUNDS["train_batch_size"]
    tau_lo, tau_hi = HYPERPARAM_BOUNDS["tau"]
    actor_lo, actor_hi = HYPERPARAM_BOUNDS["optimization"]["actor_learning_rate"]
    critic_lo, critic_hi = HYPERPARAM_BOUNDS["optimization"]["critic_learning_rate"]

    def _pb2_explore_guard(config: dict) -> dict:
        out = dict(config)

        if "train_batch_size" in out:
            out["train_batch_size"] = int(
                np.clip(
                    int(round(float(out["train_batch_size"]))),
                    int(batch_lo),
                    int(batch_hi),
                )
            )

        if "tau" in out:
            out["tau"] = float(
                np.clip(float(out["tau"]), float(tau_lo), float(tau_hi))
            )

        opt = dict(out.get("optimization", {}) or {})

        if "actor_learning_rate" in opt:
            opt["actor_learning_rate"] = float(
                np.clip(
                    float(opt["actor_learning_rate"]),
                    float(actor_lo),
                    float(actor_hi),
                )
            )

        if "critic_learning_rate" in opt:
            opt["critic_learning_rate"] = float(
                np.clip(
                    float(opt["critic_learning_rate"]),
                    float(critic_lo),
                    float(critic_hi),
                )
            )

        # Fixed SAC control: PB2 does not optimize alpha's learning rate.
        opt["entropy_learning_rate"] = float(FIXED_ENTROPY_LR)
        out["optimization"] = opt

        # No O2I in PB2: target entropy stays at the native baseline.
        out["target_entropy"] = float(base_target_entropy)

        return out

    return _pb2_explore_guard


# =============================================================================
# SAC CONFIGURATION
# =============================================================================

# The common SAC learner and Tune initialization distributions are built by
# configs.sac_config.build_tunable_sac_config(). PB2-specific scheduling and
# restore synchronization remain in this file.


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
        "custom_metrics/sac_td_diagnostics_valid",

        "custom_metrics/sac_actor_loss",
        "custom_metrics/sac_critic_loss",
        "custom_metrics/sac_alpha_loss",
        "custom_metrics/sac_alpha_value",
        "custom_metrics/sac_log_alpha_value",
        "custom_metrics/sac_target_entropy",
        "custom_metrics/sac_mean_q",
        "custom_metrics/sac_max_q",
        "custom_metrics/sac_min_q",

        "custom_metrics/pb2_sac_desired_train_batch_size",
        "custom_metrics/pb2_sac_effective_train_batch_size",
        "custom_metrics/pb2_sac_mismatch_train_batch_size",

        "custom_metrics/pb2_sac_desired_tau",
        "custom_metrics/pb2_sac_effective_tau",
        "custom_metrics/pb2_sac_mismatch_tau",

        "custom_metrics/pb2_sac_desired_actor_lr",
        "custom_metrics/pb2_sac_effective_actor_lr",
        "custom_metrics/pb2_sac_mismatch_actor_lr",

        "custom_metrics/pb2_sac_desired_critic_lr",
        "custom_metrics/pb2_sac_effective_critic_lr",
        "custom_metrics/pb2_sac_mismatch_critic_lr",

        "custom_metrics/pb2_sac_desired_alpha_lr",
        "custom_metrics/pb2_sac_effective_alpha_lr",
        "custom_metrics/pb2_sac_mismatch_alpha_lr",

        "custom_metrics/pb2_sac_desired_target_entropy",
        "custom_metrics/pb2_sac_effective_target_entropy",
        "custom_metrics/pb2_sac_mismatch_target_entropy",

        "custom_metrics/pb2_sac_effective_hp_mismatch_count",

        "perf/gpu_util_percent0",
        "perf/cpu_util_percent",
        "perf/ram_util_percent",
        "timers/training_iteration_time_ms",
    ]

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe

        if df is None or df.empty:
            continue

        present = [
            column
            for column in columns_wanted
            if column in df.columns
        ]

        out = df[present].copy()

        # Preserve Tune-emitted causal order. After checkpoint inheritance,
        # timesteps_total may move backward relative to the discarded branch.
        out["causal_order"] = np.arange(
            len(out),
            dtype=np.int64,
        )
        out["entorno"] = env_name
        out["semilla"] = int(seed)
        out["agente_id"] = (
            f"Agente_{idx + 1}"
        )

        frames.append(out)

    if not frames:
        return

    final = pd.concat(
        frames,
        ignore_index=True,
    )

    final = final.sort_values(
        ["agente_id", "causal_order"],
        kind="stable",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    final.to_csv(
        output_path,
        index=False,
    )


def _extract_pb2_trial_history(
    result_grid,
    env_name: str,
    seed: int,
    output_path: Path,
) -> None:
    """Save the per-trial PB2-SAC executable schedule for downstream audit."""
    frames = []

    wanted = [
        "training_iteration",
        "timesteps_total",
        METRIC,
        "config/train_batch_size",
        "config/tau",
        "config/target_entropy",
        "config/optimization/actor_learning_rate",
        "config/optimization/critic_learning_rate",
        "config/optimization/entropy_learning_rate",
    ]

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe

        if df is None or df.empty:
            continue

        present = [
            c
            for c in wanted
            if c in df.columns
        ]

        out = df[present].copy()
        out["causal_order"] = np.arange(
            len(out),
            dtype=np.int64,
        )
        out["entorno"] = env_name
        out["semilla"] = int(seed)
        out["agente_id"] = (
            f"Agente_{idx + 1}"
        )

        frames.append(out)

    if not frames:
        return

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    history = history.sort_values(
        ["agente_id", "causal_order"],
        kind="stable",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        output_path,
        index=False,
    )


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
                steps = pd.to_numeric(
                    df[TIME_ATTR],
                    errors="coerce",
                )
                finite_steps = steps[
                    np.isfinite(steps)
                ]

                if not finite_steps.empty:
                    final_steps = float(
                        finite_steps.iloc[-1]
                    )

            if METRIC in df.columns:
                rewards = pd.to_numeric(
                    df[METRIC],
                    errors="coerce",
                )
                finite_rewards = rewards[
                    np.isfinite(rewards)
                ]

                if not finite_rewards.empty:
                    last_finite_reward = float(
                        finite_rewards.iloc[-1]
                    )

        error = getattr(
            result,
            "error",
            None,
        )

        rows.append(
            {
                "environment": env_name,
                "training_seed": int(seed),
                "agent_id": f"Agente_{idx + 1}",
                "n_metric_rows": n_rows,
                "final_timesteps_total": final_steps,
                "reached_target_steps": int(
                    np.isfinite(final_steps)
                    and final_steps >= target_steps
                ),
                "last_finite_training_return": (
                    last_finite_reward
                ),
                "has_checkpoint": int(
                    bool(result.checkpoint)
                ),
                "checkpoint_path": (
                    str(result.checkpoint.path)
                    if result.checkpoint
                    else ""
                ),
                "result_path": str(
                    getattr(
                        result,
                        "path",
                        "",
                    )
                ),
                "error": (
                    repr(error)
                    if error is not None
                    else ""
                ),
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


def _validate_result_grid(
    result_grid,
    *,
    target_steps: int,
) -> None:
    """Reject incomplete P=4 PB2-SAC population runs."""
    results = list(result_grid)
    problems = []

    if len(results) != POBLACION_B:
        problems.append(
            f"expected {POBLACION_B} trials, "
            f"ResultGrid contains {len(results)}"
        )

    for idx, result in enumerate(results):
        agent = f"Agente_{idx + 1}"

        error = getattr(
            result,
            "error",
            None,
        )

        if error is not None:
            problems.append(
                f"{agent}: Tune/RLlib error={error!r}"
            )

        df = result.metrics_dataframe

        if df is None or df.empty:
            problems.append(
                f"{agent}: empty metrics dataframe"
            )
            continue

        if TIME_ATTR not in df.columns:
            problems.append(
                f"{agent}: missing {TIME_ATTR}"
            )
        else:
            steps = pd.to_numeric(
                df[TIME_ATTR],
                errors="coerce",
            )
            finite = steps[
                np.isfinite(steps)
            ]
            terminal = (
                float(finite.iloc[-1])
                if not finite.empty
                else np.nan
            )

            if (
                not np.isfinite(terminal)
                or terminal < target_steps
            ):
                problems.append(
                    f"{agent}: terminal "
                    f"{TIME_ATTR}={terminal}, "
                    f"expected >= {target_steps}"
                )

        if not result.checkpoint:
            problems.append(
                f"{agent}: missing final checkpoint"
            )

        # If the mismatch metric exists, require it to be zero at all
        # finite audited rows.
        mismatch_col = (
            "custom_metrics/"
            "pb2_sac_effective_hp_mismatch_count"
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
                    "hyperparameter mismatch detected"
                )

    if problems:
        raise RuntimeError(
            "Incomplete/invalid PB2-SAC population run:\n  - "
            + "\n  - ".join(problems)
        )


def _copy_final_checkpoints(
    result_grid,
    env_name: str,
    seed: int,
) -> None:
    root = (
        (RESULTS_ROOT / "champions")
        / env_name
        / f"{OUTPUT_NAME}_seed{seed}"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for idx, result in enumerate(result_grid):
        if not result.checkpoint:
            logger.warning(
                "No final checkpoint for Agente_%s",
                idx + 1,
            )
            continue

        source = Path(result.checkpoint.path)
        target = root / f"Agente_{idx + 1}"

        try:
            if target.exists():
                shutil.rmtree(target)

            shutil.copytree(
                source,
                target,
            )
        except Exception as exc:
            logger.warning(
                "Could not copy checkpoint %s -> %s: %s",
                source,
                target,
                exc,
            )


def _save_metadata(
    path: Path,
    *,
    env_name: str,
    seed: int,
    target_steps: int,
    perturbation_interval: int,
    quantile_fraction: float,
    action_dim: int,
    base_target_entropy: float,
) -> None:
    payload = {
        "algorithm": ALGO_NAME,
        "run_name": OUTPUT_NAME,
        "environment": env_name,
        "seed": int(seed),

        "population": POBLACION_B,
        "max_timesteps_per_worker": int(
            target_steps
        ),
        "perturbation_interval": int(
            perturbation_interval
        ),
        "quantile_fraction": float(
            quantile_fraction
        ),

        "optimized_hyperparameters": [
            "train_batch_size",
            "tau",
            "optimization/actor_learning_rate",
            "optimization/critic_learning_rate",
        ],

        "hyperparameter_bounds": (
            HYPERPARAM_BOUNDS
        ),

        "configuration_source": {
            "environments": "configs/environments.py",
            "seeds": "configs/seeds.py",
            "sac": "configs/sac_config.py",
        },

        "initial_sampling": {
            "train_batch_size": (
                "randint[256,2048]"
            ),
            "tau": (
                "loguniform[1e-3,2e-2]"
            ),
            "actor_learning_rate": (
                "loguniform[1e-5,3e-4]"
            ),
            "critic_learning_rate": (
                "loguniform[1e-4,1e-3]"
            ),
        },

        "pb2": {
            "scheduler": "ray.tune.schedulers.pb2.PB2",
            "synch": False,
            "native_tv_gp": True,
            "native_raw_coordinate_geometry": True,
            "custom_explore_guard": True,
            "custom_explore_role": (
                "integer/executable conversion and fixed-control enforcement only"
            ),
        },

        "sac": {
            "gamma": FIXED_GAMMA,
            "initial_alpha": INITIAL_ALPHA,
            "n_step": N_STEP,
            "target_network_update_freq": (
                TARGET_NETWORK_UPDATE_FREQ
            ),
            "learning_starts": LEARNING_STARTS,
            "training_intensity": (
                TRAINING_INTENSITY
            ),
            "twin_q": TWIN_Q,

            "entropy_learning_rate": (
                FIXED_ENTROPY_LR
            ),

            "action_dim": int(action_dim),
            "target_entropy": float(
                base_target_entropy
            ),
            "target_entropy_policy": (
                "fixed baseline -action_dim; "
                "not optimized and no O2I modulation"
            ),

            "replay_buffer_type": (
                "MultiAgentPrioritizedReplayBuffer"
            ),
            "replay_buffer_capacity": (
                REPLAY_BUFFER_CAPACITY
            ),
            "store_buffer_in_checkpoints": STORE_BUFFER_IN_CHECKPOINTS,

            "model_hiddens": MODEL_HIDDENS,
            "model_activation": (
                MODEL_ACTIVATION
            ),

            "num_env_runners": (
                NUM_ENV_RUNNERS
            ),
            "num_envs_per_env_runner": (
                NUM_ENVS_PER_RUNNER
            ),
            "rollout_fragment_length": (
                ROLLOUT_FRAGMENT_LENGTH
            ),
            "min_sample_timesteps_per_iteration": (
                MIN_SAMPLE_TIMESTEPS_PER_ITERATION
            ),
            "observation_filter": (
                OBSERVATION_FILTER
            ),
            "num_gpus_per_trial": (
                NUM_GPUS_PER_TRIAL
            ),
            "old_api_stack": True,
        },

        "checkpointing": {
            "checkpoint_at_end": True,
            "store_buffer_in_checkpoints": STORE_BUFFER_IN_CHECKPOINTS,
            "restore_hyperparameter_resynchronization": True,
            "source_of_truth": (
                "AlgorithmConfig/Tune PB2 configuration"
            ),
        },

        "diagnostics": {
            "td_error_logged_only": True,
            "td_error_used_by_scheduler": False,
            "learner_losses_logged_only": True,
        },

        "comparability_to_coda_sac": {
            "same_inner_learner": True,
            "same_search_domain": True,
            "same_initial_sampling": True,
            "same_population": True,
            "same_horizon": True,
            "same_adaptation_interval": True,
            "same_quantile": True,
            "same_training_intensity": True,
            "same_replay_buffer": True,
            "same_model": True,
            "same_sampling_resources": True,
            "same_checkpoint_buffer_semantics": True,
            "same_restore_hp_sync": True,
            "intentional_differences": [
                "native PB2 TV-GP selection instead of CODA contextual TV-GP",
                "no I2O learner diagnostic context",
                "no O2I uncertainty actuator",
                "fixed target entropy at -action_dim",
            ],
        },

        "versions": {
            "python": sys.version,
            "ray": ray.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": _package_version(
                "scikit-learn"
            ),
            "scipy": _package_version(
                "scipy"
            ),
            "gymnasium": _package_version(
                "gymnasium"
            ),
            "mujoco": _package_version(
                "mujoco"
            ),
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


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
        f"Starting {OUTPUT_NAME} | "
        f"env={env_name} | "
        f"seed={seed} | "
        f"population={POBLACION_B} | "
        f"steps/worker={target_steps:,}"
    )
    print("=" * 78)

    _seed_everything(seed)

    action_dim = action_dimension(env_name)
    base_target_entropy = sac_baseline_target_entropy(env_name)

    # Ray PB2 supports nested hyperparameter bounds and internally flattens them.
    # Keep the exact same raw four-dimensional domain as CODA+SAC. PB2 itself
    # operates in its native raw-coordinate GP space (no CODA log10 transform).
    scheduler = PB2(
        time_attr=TIME_ATTR,
        perturbation_interval=int(perturbation_interval),
        hyperparam_bounds=HYPERPARAM_BOUNDS,
        quantile_fraction=float(quantile_fraction),
        log_config=True,
        synch=False,
        custom_explore_fn=_make_pb2_explore_guard(base_target_entropy),
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required because each "
                "SAC trial requests "
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
            callbacks=PB2SACDiagnosticsCallback,
        )

        storage_root = (
            RESULTS_ROOT
            / "ray_tune_logs"
            / OUTPUT_NAME
        ).resolve()

        storage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        tuner = tune.Tuner(
            "SAC",
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=POBLACION_B,
                metric=METRIC,
                mode="max",
                trial_name_creator=lambda trial: (
                    f"{OUTPUT_NAME}_{env_name}_"
                    f"{seed}_{trial.trial_id}"
                ),
                trial_dirname_creator=lambda trial: (
                    f"{OUTPUT_NAME}_{env_name}_"
                    f"{seed}_{trial.trial_id}"
                ),
                # Intentionally do not enable reuse_actors here:
                # CODA+SAC also runs without actor reuse, preserving
                # comparable scheduling/checkpoint semantics.
            ),
            param_space=sac_config.to_dict(),
            run_config=tune.RunConfig(
                name=(
                    f"{OUTPUT_NAME}_{env_name}_"
                    f"Seed{seed}"
                ),
                verbose=0,
                storage_path=str(storage_root),
                stop={
                    TIME_ATTR: int(
                        target_steps
                    )
                },
                checkpoint_config=(
                    tune.CheckpointConfig(
                        checkpoint_at_end=True,
                    )
                ),
            ),
        )

        results = tuner.fit()

        # Save all diagnostics before validating, so failed runs remain auditable.
        metrics_path = (
            (RESULTS_ROOT / "metrics")
            / env_name
            / f"metrics_{OUTPUT_NAME}_seed{seed}.csv"
        )
        _extract_metrics(
            results,
            env_name,
            seed,
            metrics_path,
        )

        # Native PB2 surrogate dataset/history.
        scheduler_path = (
            (RESULTS_ROOT / "scheduler")
            / env_name
            / f"scheduler_{OUTPUT_NAME}_seed{seed}.csv"
        )
        scheduler_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler_data = getattr(scheduler, "data", None)
        if isinstance(scheduler_data, pd.DataFrame):
            scheduler_data.to_csv(scheduler_path, index=False)
        else:
            logger.warning(
                "PB2 scheduler.data is unavailable; native PB2 text logs remain "
                "inside the Tune experiment directory."
            )

        # Per-worker executable config history (separate from PB2's GP dataset).
        history_path = (
            (RESULTS_ROOT / "scheduler")
            / env_name
            / f"history_{OUTPUT_NAME}_seed{seed}.csv"
        )
        _extract_pb2_trial_history(
            results,
            env_name,
            seed,
            history_path,
        )

        status_path = (
            (RESULTS_ROOT / "status")
            / env_name
            / f"status_{OUTPUT_NAME}_seed{seed}.csv"
        )
        _save_trial_status_summary(
            results,
            env_name,
            seed,
            target_steps,
            status_path,
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
            perturbation_interval=(
                perturbation_interval
            ),
            quantile_fraction=(
                quantile_fraction
            ),
            action_dim=action_dim,
            base_target_entropy=(
                base_target_entropy
            ),
        )

        _copy_final_checkpoints(
            results,
            env_name,
            seed,
        )

        _validate_result_grid(
            results,
            target_steps=target_steps,
        )

        print(
            f"Completed {OUTPUT_NAME}: "
            f"{env_name}, seed={seed} | "
            f"fixed_target_entropy="
            f"{base_target_entropy:.3f}"
        )

        return True

    except Exception as exc:
        print(
            f"\nCRITICAL ERROR in {env_name}, "
            f"seed={seed}: {exc}"
        )
        traceback.print_exc()
        return False

    finally:
        if ray.is_initialized():
            ray.shutdown()
        time.sleep(2)


def main() -> None:
    started = time.time()
    failures = []

    if PB2_SAC_SMOKE_TEST:
        if PB2_SAC_SMOKE_ENV not in ENVIRONMENT_CONFIG:
            raise KeyError(
                "Unknown smoke environment: "
                f"{PB2_SAC_SMOKE_ENV}"
            )

        experiment_items = [
            (
                PB2_SAC_SMOKE_ENV,
                ENVIRONMENT_CONFIG[
                    PB2_SAC_SMOKE_ENV
                ],
            )
        ]
        total = len(
            PB2_SAC_SMOKE_SEEDS
        )

    else:
        experiment_items = list(
            ENVIRONMENT_CONFIG.items()
        )
        total = sum(
            len(cfg["semillas"])
            for _, cfg in experiment_items
        )

    done = 0

    for env_name, env_cfg in experiment_items:
        if PB2_SAC_SMOKE_TEST:
            seeds: Sequence[int] = (
                PB2_SAC_SMOKE_SEEDS
            )
            target_steps = (
                PB2_SAC_SMOKE_STEPS
            )
            interval = (
                PB2_SAC_SMOKE_INTERVAL
            )
        else:
            seeds = env_cfg["semillas"]
            target_steps = TIMESTEPS_MAX
            interval = int(
                env_cfg.get(
                    "perturbation_interval",
                    PERTURBATION_INTERVAL,
                )
            )

        quantile = float(
            env_cfg.get(
                "quantile_fraction",
                QUANTILE_FRACTION,
            )
        )

        for seed in seeds:
            ok = run_experiment(
                env_name,
                int(seed),
                target_steps=int(
                    target_steps
                ),
                perturbation_interval=int(
                    interval
                ),
                quantile_fraction=quantile,
            )

            if not ok:
                failures.append(
                    f"{env_name} - seed {seed}"
                )

            done += 1
            print(
                f"Global progress: "
                f"{done}/{total}"
            )

    hours = (
        time.time() - started
    ) / 3600.0

    print("\n" + "-" * 78)
    print(
        f"Experiments finished in "
        f"{hours:.2f} hours"
    )

    if failures:
        print(
            f"Failures ({len(failures)}):"
        )
        for failure in failures:
            print(
                f"  - {failure}"
            )
    else:
        print(
            "All requested experiments "
            "completed successfully."
        )

    print("-" * 78)


if __name__ == "__main__":
    main()
