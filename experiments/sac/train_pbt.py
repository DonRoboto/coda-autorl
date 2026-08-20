#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PBT + SAC baseline using the shared repository configuration.

This is the refactored GitHub version of the validated PBT+SAC baseline.
All learner/search/budget constants are imported from ``configs/`` so the
SAC protocol has a single source of truth shared with PB2, ASHA, and CODA.

Scientific differences relative to CODA+SAC are intentionally limited to:

1. PBT uses standard checkpoint exploitation plus stochastic perturbation/
   resampling instead of CODA's diagnostic-aware TV-GP selection.
2. PBT has no I2O learner-state feedback.
3. PBT has no O2I uncertainty actuator.
4. SAC target entropy remains fixed at the baseline value ``-action_dim``.

The restore-time synchronization/audit remains in this file because it is
specific to the PBT execution path on RLlib's legacy SAC stack.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CPU limits must be set before NumPy / Ray / PyTorch imports.
# ---------------------------------------------------------------------------
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
from importlib import metadata
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.schedulers import PopulationBasedTraining
import torch
import torch.backends.cudnn as cudnn


# =============================================================================
# REPOSITORY IMPORTS
# =============================================================================

# Intended location:
#     <repo>/experiments/sac/train_pbt.py
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
    tune_search_space,
    build_tunable_sac_config,
)


logger = logging.getLogger(__name__)


# =============================================================================
# METHOD-SPECIFIC CONFIGURATION
# =============================================================================

ALGO_NAME = "PBT_SAC"

# Kept for backward-compatible result names used by the current evaluator.
RUN_TAG = os.environ.get("PBT_SAC_RUN_TAG", "PB2SPACE").strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Standard PBT exploration settings.
RESAMPLE_PROBABILITY = 0.25
PERTURBATION_FACTORS = (1.2, 0.8)

RESULTS_ROOT = REPO_ROOT / "results"


# =============================================================================
# SAFE SMOKE TEST
# =============================================================================

# Smoke test enabled by default.
# Full campaign:
#     PBT_SAC_SMOKE=0 python experiments/sac/train_pbt.py
PBT_SAC_SMOKE_TEST = os.environ.get("PBT_SAC_SMOKE", "1") != "0"
PBT_SAC_SMOKE_ENV = os.environ.get(
    "PBT_SAC_SMOKE_ENV",
    "Hopper-v5",
)
PBT_SAC_SMOKE_SEEDS = [
    int(
        os.environ.get(
            "PBT_SAC_SMOKE_SEED",
            str(SMOKE_TRAINING_SEED),
        )
    )
]
PBT_SAC_SMOKE_STEPS = int(
    os.environ.get(
        "PBT_SAC_SMOKE_STEPS",
        "500000",
    )
)
PBT_SAC_SMOKE_INTERVAL = int(
    os.environ.get(
        "PBT_SAC_SMOKE_INTERVAL",
        str(PERTURBATION_INTERVAL),
    )
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
    """Seed Python, NumPy, and PyTorch consistently."""
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
    """Convert tensors/lists/arrays to a finite 1-D float array."""
    if value is None:
        return np.empty((0,), dtype=np.float64)

    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()

        arr = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(-1)
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
    value = _config_attr(
        config,
        "optimization",
        {},
    )
    return (
        dict(value or {})
        if isinstance(value, dict)
        else {}
    )


# =============================================================================
# CALLBACK: RESTORE SYNCHRONIZATION + AUDIT
# =============================================================================

class PBTSACDiagnosticsCallback(DefaultCallbacks):
    """PBT-SAC restore synchronization and observational diagnostics.

    Diagnostics are never fed into PBT.

    The synchronization addresses the legacy RLlib restore path where a donor
    checkpoint may overwrite the receiver's PBT-mutated SAC configuration or
    optimizer learning rates. AlgorithmConfig/Tune remains the source of truth.
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
        policy_data = learner.get(
            "default_policy",
            {},
        ) or {}

        if not isinstance(policy_data, dict):
            policy_data = {}

        learner_stats = policy_data.get(
            "learner_stats",
            {},
        ) or {}

        if not isinstance(learner_stats, dict):
            learner_stats = {}

        return policy_data, learner_stats

    @staticmethod
    def _set_target_entropy_tensor(
        policy,
        value: float,
    ) -> None:
        """Set fixed target entropy on the live SAC model and GPU towers."""
        models = []

        model = getattr(
            policy,
            "model",
            None,
        )
        if model is not None:
            models.append(model)

        towers = getattr(
            policy,
            "model_gpu_towers",
            None,
        )
        if towers:
            models.extend(list(towers))

        with torch.no_grad():
            for tower in models:
                target = getattr(
                    tower,
                    "target_entropy",
                    None,
                )
                if target is None:
                    continue

                try:
                    target.fill_(float(value))
                except Exception:
                    try:
                        target.data.fill_(float(value))
                    except Exception:
                        pass

    def _sync_effective_hyperparams(
        self,
        algorithm,
    ) -> None:
        """Re-apply the PBT/Tune configuration after donor restore."""
        policy = self._get_default_policy(
            algorithm
        )
        if policy is None:
            return

        cfg = algorithm.config
        optimization = _optimization_dict(cfg)

        desired_batch = _safe_float(
            _config_attr(
                cfg,
                "train_batch_size",
                np.nan,
            ),
            np.nan,
        )
        desired_tau = _safe_float(
            _config_attr(
                cfg,
                "tau",
                np.nan,
            ),
            np.nan,
        )
        desired_target_entropy = _safe_float(
            _config_attr(
                cfg,
                "target_entropy",
                np.nan,
            ),
            np.nan,
        )

        desired_actor_lr = _safe_float(
            optimization.get(
                "actor_learning_rate",
                np.nan,
            ),
            np.nan,
        )
        desired_critic_lr = _safe_float(
            optimization.get(
                "critic_learning_rate",
                np.nan,
            ),
            np.nan,
        )
        desired_alpha_lr = _safe_float(
            optimization.get(
                "entropy_learning_rate",
                FIXED_ENTROPY_LR,
            ),
            FIXED_ENTROPY_LR,
        )

        # ------------------------------------------------------------------
        # Policy config
        # ------------------------------------------------------------------
        policy_config = getattr(
            policy,
            "config",
            None,
        )
        if not isinstance(policy_config, dict):
            policy_config = None

        if policy_config is not None:
            if np.isfinite(desired_batch):
                policy_config[
                    "train_batch_size"
                ] = int(round(desired_batch))

            if np.isfinite(desired_tau):
                policy_config["tau"] = float(
                    desired_tau
                )

            if np.isfinite(
                desired_target_entropy
            ):
                policy_config[
                    "target_entropy"
                ] = float(
                    desired_target_entropy
                )

            policy_opt = dict(
                policy_config.get(
                    "optimization",
                    {},
                ) or {}
            )

            if np.isfinite(desired_actor_lr):
                policy_opt[
                    "actor_learning_rate"
                ] = float(desired_actor_lr)

            if np.isfinite(desired_critic_lr):
                policy_opt[
                    "critic_learning_rate"
                ] = float(desired_critic_lr)

            if np.isfinite(desired_alpha_lr):
                policy_opt[
                    "entropy_learning_rate"
                ] = float(desired_alpha_lr)

            policy_config[
                "optimization"
            ] = policy_opt

        # ------------------------------------------------------------------
        # Actor optimizer
        # ------------------------------------------------------------------
        actor_optim = getattr(
            policy,
            "actor_optim",
            None,
        )

        if (
            actor_optim is not None
            and np.isfinite(desired_actor_lr)
        ):
            try:
                for group in actor_optim.param_groups:
                    group["lr"] = float(
                        desired_actor_lr
                    )
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Critic optimizers
        # ------------------------------------------------------------------
        critic_optims = (
            getattr(
                policy,
                "critic_optims",
                None,
            )
            or []
        )

        if np.isfinite(desired_critic_lr):
            for optimizer in critic_optims:
                try:
                    for group in optimizer.param_groups:
                        group["lr"] = float(
                            desired_critic_lr
                        )
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Temperature/alpha optimizer
        # ------------------------------------------------------------------
        alpha_optim = getattr(
            policy,
            "alpha_optim",
            None,
        )

        if (
            alpha_optim is not None
            and np.isfinite(desired_alpha_lr)
        ):
            try:
                for group in alpha_optim.param_groups:
                    group["lr"] = float(
                        desired_alpha_lr
                    )
            except Exception:
                pass

        # Fallback for Ray versions exposing only a combined optimizer list.
        all_optims = (
            getattr(
                policy,
                "_optimizers",
                None,
            )
            or []
        )

        if all_optims:
            if (
                actor_optim is None
                and np.isfinite(desired_actor_lr)
            ):
                try:
                    for group in all_optims[
                        0
                    ].param_groups:
                        group["lr"] = float(
                            desired_actor_lr
                        )
                except Exception:
                    pass

            if (
                alpha_optim is None
                and np.isfinite(desired_alpha_lr)
            ):
                try:
                    for group in all_optims[
                        -1
                    ].param_groups:
                        group["lr"] = float(
                            desired_alpha_lr
                        )
                except Exception:
                    pass

        # Fixed baseline target entropy: no O2I in PBT.
        if np.isfinite(desired_target_entropy):
            self._set_target_entropy_tensor(
                policy,
                desired_target_entropy,
            )

    def _effective_values(
        self,
        algorithm,
    ) -> dict:
        """Read live SAC execution values for the restore audit."""
        policy = self._get_default_policy(
            algorithm
        )
        if policy is None:
            return {}

        policy_config = getattr(
            policy,
            "config",
            {},
        ) or {}

        if not isinstance(
            policy_config,
            dict,
        ):
            policy_config = {}

        actor_lr = np.nan
        actor_optim = getattr(
            policy,
            "actor_optim",
            None,
        )
        if actor_optim is not None:
            try:
                actor_lr = _safe_float(
                    actor_optim.param_groups[
                        0
                    ]["lr"],
                    np.nan,
                )
            except Exception:
                pass

        critic_lr = np.nan
        critic_optims = (
            getattr(
                policy,
                "critic_optims",
                None,
            )
            or []
        )
        if critic_optims:
            try:
                critic_lr = _safe_float(
                    critic_optims[
                        0
                    ].param_groups[0]["lr"],
                    np.nan,
                )
            except Exception:
                pass

        alpha_lr = np.nan
        alpha_optim = getattr(
            policy,
            "alpha_optim",
            None,
        )
        if alpha_optim is not None:
            try:
                alpha_lr = _safe_float(
                    alpha_optim.param_groups[
                        0
                    ]["lr"],
                    np.nan,
                )
            except Exception:
                pass

        target_entropy = np.nan
        model = getattr(
            policy,
            "model",
            None,
        )
        if model is not None:
            target_entropy = _safe_float(
                getattr(
                    model,
                    "target_entropy",
                    np.nan,
                ),
                np.nan,
            )

        return {
            "train_batch_size": _safe_float(
                policy_config.get(
                    "train_batch_size",
                    np.nan,
                ),
                np.nan,
            ),
            "tau": _safe_float(
                policy_config.get(
                    "tau",
                    np.nan,
                ),
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
        """Write desired/effective SAC parameters and mismatch flags."""
        cfg = algorithm.config
        optimization = _optimization_dict(cfg)

        desired = {
            "train_batch_size": _safe_float(
                _config_attr(
                    cfg,
                    "train_batch_size",
                    np.nan,
                ),
                np.nan,
            ),
            "tau": _safe_float(
                _config_attr(
                    cfg,
                    "tau",
                    np.nan,
                ),
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

        effective = self._effective_values(
            algorithm
        )

        mismatch_count = 0

        for name in (
            "train_batch_size",
            "tau",
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "target_entropy",
        ):
            desired_value = desired.get(
                name,
                np.nan,
            )
            effective_value = effective.get(
                name,
                np.nan,
            )

            custom[
                f"pbt_sac_desired_{name}"
            ] = desired_value
            custom[
                f"pbt_sac_effective_{name}"
            ] = effective_value

            if (
                name == "train_batch_size"
                and np.isfinite(desired_value)
                and np.isfinite(effective_value)
            ):
                mismatch = (
                    int(round(desired_value))
                    != int(round(effective_value))
                )
            else:
                mismatch = bool(
                    np.isfinite(desired_value)
                    and np.isfinite(
                        effective_value
                    )
                    and not np.isclose(
                        desired_value,
                        effective_value,
                        rtol=1e-7,
                        atol=1e-12,
                    )
                )

            custom[
                f"pbt_sac_mismatch_{name}"
            ] = float(mismatch)

            mismatch_count += int(
                mismatch
            )

        custom[
            "pbt_sac_effective_hp_mismatch_count"
        ] = float(mismatch_count)

    def on_checkpoint_loaded(
        self,
        *,
        algorithm,
        **kwargs,
    ):
        # Checkpoint is restored first, then the new PBT config is enforced.
        self._sync_effective_hyperparams(
            algorithm
        )

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs,
    ):
        # Defensive synchronization also handles Ray-version callback ordering.
        self._sync_effective_hyperparams(
            algorithm
        )

        custom = result.setdefault(
            "custom_metrics",
            {},
        )

        self._audit_effective_hyperparams(
            algorithm,
            custom,
        )

        policy_data, learner_stats = (
            self._extract_policy_and_learner_stats(
                result
            )
        )

        # TD-error is logged only for offline comparability.
        td_value = None
        for candidate in (
            policy_data.get("td_error"),
            learner_stats.get("td_error"),
            result.get("td_error"),
            result.get(
                "info/learner/default_policy/"
                "td_error"
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
            custom[
                "sac_td_error_median"
            ] = float(
                np.median(td_errors)
            )
            custom[
                "sac_td_error_p95"
            ] = float(
                np.percentile(
                    td_errors,
                    95.0,
                )
            )
            custom[
                "sac_td_diagnostics_valid"
            ] = 1.0
        else:
            custom[
                "sac_td_error_median"
            ] = np.nan
            custom[
                "sac_td_error_p95"
            ] = np.nan
            custom[
                "sac_td_diagnostics_valid"
            ] = 0.0

        # Learner diagnostics are observational only.
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
            custom[
                f"sac_{key}"
            ] = _safe_float(
                learner_stats.get(
                    key,
                    np.nan,
                ),
                np.nan,
            )

        self._iter_count += 1

        if self._iter_count % 10 == 0:
            try:
                cfg = algorithm.config
                opt = _optimization_dict(cfg)

                logger.info(
                    "PBT-SAC | batch=%s "
                    "tau=%.5g actor_lr=%.5g "
                    "critic_lr=%.5g "
                    "target_entropy=%.4f "
                    "mismatch=%d",
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
                            "pbt_sac_effective_hp_mismatch_count",
                            0.0,
                        )
                    ),
                )
            except Exception:
                pass


# =============================================================================
# PBT EXPLORATION
# =============================================================================

def _make_bounded_pbt_explore(
    base_target_entropy: float,
):
    """Create the PBT post-exploration executable-config guard.

    PBT still performs its native resampling/multiplicative perturbation.
    This function only clips the proposal to the common SAC search domain,
    converts batch size to integer, and reasserts fixed SAC controls.
    """

    batch_lo, batch_hi = (
        HYPERPARAM_BOUNDS[
            "train_batch_size"
        ]
    )
    tau_lo, tau_hi = (
        HYPERPARAM_BOUNDS["tau"]
    )

    actor_lo, actor_hi = (
        HYPERPARAM_BOUNDS[
            "optimization"
        ]["actor_learning_rate"]
    )
    critic_lo, critic_hi = (
        HYPERPARAM_BOUNDS[
            "optimization"
        ]["critic_learning_rate"]
    )

    def _bounded_pbt_explore(
        config: dict,
    ) -> dict:
        config[
            "train_batch_size"
        ] = int(
            np.clip(
                int(
                    round(
                        float(
                            config[
                                "train_batch_size"
                            ]
                        )
                    )
                ),
                int(batch_lo),
                int(batch_hi),
            )
        )

        config["tau"] = float(
            np.clip(
                float(config["tau"]),
                float(tau_lo),
                float(tau_hi),
            )
        )

        optimization = dict(
            config.get(
                "optimization",
                {},
            )
            or {}
        )

        optimization[
            "actor_learning_rate"
        ] = float(
            np.clip(
                float(
                    optimization[
                        "actor_learning_rate"
                    ]
                ),
                float(actor_lo),
                float(actor_hi),
            )
        )

        optimization[
            "critic_learning_rate"
        ] = float(
            np.clip(
                float(
                    optimization[
                        "critic_learning_rate"
                    ]
                ),
                float(critic_lo),
                float(critic_hi),
            )
        )

        # Fixed, common SAC control.
        optimization[
            "entropy_learning_rate"
        ] = float(FIXED_ENTROPY_LR)

        config[
            "optimization"
        ] = optimization

        # External baselines have no O2I actuator.
        config[
            "target_entropy"
        ] = float(base_target_entropy)

        return config

    return _bounded_pbt_explore


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

        "custom_metrics/pbt_sac_desired_train_batch_size",
        "custom_metrics/pbt_sac_effective_train_batch_size",
        "custom_metrics/pbt_sac_mismatch_train_batch_size",

        "custom_metrics/pbt_sac_desired_tau",
        "custom_metrics/pbt_sac_effective_tau",
        "custom_metrics/pbt_sac_mismatch_tau",

        "custom_metrics/pbt_sac_desired_actor_lr",
        "custom_metrics/pbt_sac_effective_actor_lr",
        "custom_metrics/pbt_sac_mismatch_actor_lr",

        "custom_metrics/pbt_sac_desired_critic_lr",
        "custom_metrics/pbt_sac_effective_critic_lr",
        "custom_metrics/pbt_sac_mismatch_critic_lr",

        "custom_metrics/pbt_sac_desired_alpha_lr",
        "custom_metrics/pbt_sac_effective_alpha_lr",
        "custom_metrics/pbt_sac_mismatch_alpha_lr",

        "custom_metrics/pbt_sac_desired_target_entropy",
        "custom_metrics/pbt_sac_effective_target_entropy",
        "custom_metrics/pbt_sac_mismatch_target_entropy",

        "custom_metrics/pbt_sac_effective_hp_mismatch_count",

        "perf/gpu_util_percent0",
        "perf/cpu_util_percent",
        "perf/ram_util_percent",
        "timers/training_iteration_time_ms",
    ]

    for idx, result in enumerate(
        result_grid
    ):
        df = result.metrics_dataframe

        if df is None or df.empty:
            continue

        present = [
            column
            for column in columns_wanted
            if column in df.columns
        ]

        out = df[present].copy()

        # Preserve the execution order emitted by Tune.
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
        [
            "agente_id",
            "causal_order",
        ],
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


def _extract_pbt_history(
    result_grid,
    env_name: str,
    seed: int,
    output_path: Path,
) -> None:
    """Save the executed PBT hyperparameter schedule."""
    frames = []

    wanted = [
        "training_iteration",
        TIME_ATTR,
        METRIC,
        "config/train_batch_size",
        "config/tau",
        "config/target_entropy",
        "config/optimization/actor_learning_rate",
        "config/optimization/critic_learning_rate",
        "config/optimization/entropy_learning_rate",
    ]

    for idx, result in enumerate(
        result_grid
    ):
        df = result.metrics_dataframe

        if df is None or df.empty:
            continue

        present = [
            column
            for column in wanted
            if column in df.columns
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
        [
            "agente_id",
            "causal_order",
        ],
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

    for idx, result in enumerate(
        result_grid
    ):
        df = result.metrics_dataframe

        n_rows = (
            int(len(df))
            if df is not None
            else 0
        )

        final_steps = np.nan
        last_finite_reward = np.nan

        if df is not None and not df.empty:
            if TIME_ATTR in df.columns:
                steps = pd.to_numeric(
                    df[TIME_ATTR],
                    errors="coerce",
                )
                finite = steps[
                    np.isfinite(steps)
                ]

                if not finite.empty:
                    final_steps = float(
                        finite.iloc[-1]
                    )

            if METRIC in df.columns:
                rewards = pd.to_numeric(
                    df[METRIC],
                    errors="coerce",
                )
                finite = rewards[
                    np.isfinite(rewards)
                ]

                if not finite.empty:
                    last_finite_reward = float(
                        finite.iloc[-1]
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
                "agent_id": (
                    f"Agente_{idx + 1}"
                ),
                "n_metric_rows": n_rows,
                "final_timesteps_total": (
                    final_steps
                ),
                "reached_target_steps": int(
                    np.isfinite(final_steps)
                    and final_steps
                    >= target_steps
                ),
                "last_finite_training_return": (
                    last_finite_reward
                ),
                "has_checkpoint": int(
                    bool(result.checkpoint)
                ),
                "checkpoint_path": (
                    str(
                        result.checkpoint.path
                    )
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
    """Reject incomplete or restore-invalid PBT-SAC population runs."""
    results = list(result_grid)
    problems = []

    if len(results) != POPULATION_SIZE:
        problems.append(
            f"expected {POPULATION_SIZE} "
            f"trials, ResultGrid contains "
            f"{len(results)}"
        )

    mismatch_col = (
        "custom_metrics/"
        "pbt_sac_effective_hp_mismatch_count"
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
                f"{agent}: "
                f"Tune/RLlib error={error!r}"
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
            finite_steps = steps[
                np.isfinite(steps)
            ]

            terminal = (
                float(
                    finite_steps.iloc[-1]
                )
                if not finite_steps.empty
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
                and (
                    finite_mismatch != 0
                ).any()
            ):
                problems.append(
                    f"{agent}: non-zero "
                    "restore-time HP mismatch"
                )

    if problems:
        raise RuntimeError(
            "Invalid PBT-SAC population run:\n  - "
            + "\n  - ".join(problems)
        )


def _copy_final_checkpoints(
    result_grid,
    env_name: str,
    seed: int,
) -> None:
    root = (
        RESULTS_ROOT
        / "champions"
        / env_name
        / f"{OUTPUT_NAME}_seed{seed}"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for idx, result in enumerate(
        result_grid
    ):
        if not result.checkpoint:
            logger.warning(
                "No final checkpoint for "
                "Agente_%s",
                idx + 1,
            )
            continue

        source = Path(
            result.checkpoint.path
        )
        target = (
            root
            / f"Agente_{idx + 1}"
        )

        try:
            if target.exists():
                shutil.rmtree(target)

            shutil.copytree(
                source,
                target,
            )
        except Exception as exc:
            logger.warning(
                "Could not copy checkpoint "
                "%s -> %s: %s",
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
    """Write an explicit record of the shared SAC and PBT settings."""

    payload = {
        "algorithm": ALGO_NAME,
        "run_name": OUTPUT_NAME,
        "environment": env_name,
        "seed": int(seed),

        "population": int(
            POPULATION_SIZE
        ),
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
            "environments": (
                "configs/environments.py"
            ),
            "seeds": (
                "configs/seeds.py"
            ),
            "sac": (
                "configs/sac_config.py"
            ),
        },

        "pbt": {
            "resample_probability": (
                RESAMPLE_PROBABILITY
            ),
            "perturbation_factors": list(
                PERTURBATION_FACTORS
            ),
            "synch": False,
            "bounded_custom_explore": True,
            "mutation_rule": (
                "standard Ray PBT "
                "resampling/multiplicative "
                "perturbation followed by "
                "clipping to the common "
                "SAC search domain"
            ),
        },

        "sac": {
            "gamma": FIXED_GAMMA,
            "initial_alpha": INITIAL_ALPHA,
            "n_step": N_STEP,
            "target_network_update_freq": (
                TARGET_NETWORK_UPDATE_FREQ
            ),
            "learning_starts": (
                LEARNING_STARTS
            ),
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
                "fixed at -action_dim; "
                "no O2I actuator"
            ),

            "replay_buffer_capacity": (
                REPLAY_BUFFER_CAPACITY
            ),
            "replay_buffer_config": (
                REPLAY_BUFFER_CONFIG
            ),
            "store_buffer_in_checkpoints": (
                STORE_BUFFER_IN_CHECKPOINTS
            ),

            "model_hiddens": (
                MODEL_HIDDENS
            ),
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
            "store_buffer_in_checkpoints": (
                STORE_BUFFER_IN_CHECKPOINTS
            ),
            "restore_hyperparameter_resynchronization": True,
            "audit_metric": (
                "pbt_sac_effective_hp_mismatch_count"
            ),
        },

        "versions": {
            "python": sys.version,
            "ray": ray.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
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
        f"population={POPULATION_SIZE} | "
        f"steps/worker={target_steps:,}"
    )
    print("=" * 78)

    _seed_everything(seed)

    # Environment-specific fixed SAC baseline.
    action_dim = action_dimension(
        env_name
    )
    base_target_entropy = (
        sac_baseline_target_entropy(
            env_name
        )
    )

    # Same initial sampling distributions as every SAC method.
    hyperparam_mutations = (
        tune_search_space()
    )

    scheduler = PopulationBasedTraining(
        time_attr=TIME_ATTR,
        metric=METRIC,
        mode="max",
        perturbation_interval=int(
            perturbation_interval
        ),
        hyperparam_mutations=(
            hyperparam_mutations
        ),
        quantile_fraction=float(
            quantile_fraction
        ),
        resample_probability=(
            RESAMPLE_PROBABILITY
        ),
        perturbation_factors=(
            PERTURBATION_FACTORS
        ),
        custom_explore_fn=(
            _make_bounded_pbt_explore(
                base_target_entropy
            )
        ),
        log_config=True,
        synch=False,
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required because "
                "configs/sac_config.py sets "
                f"NUM_GPUS_PER_TRIAL="
                f"{NUM_GPUS_PER_TRIAL}."
            )

        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
            include_dashboard=False,
        )

        # The common SAC learner/search initialization now comes entirely
        # from configs/sac_config.py.
        sac_config = build_tunable_sac_config(
            env_name,
            seed,
            target_entropy=(
                base_target_entropy
            ),
            callbacks=(
                PBTSACDiagnosticsCallback
            ),
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
                num_samples=(
                    POPULATION_SIZE
                ),
                metric=METRIC,
                mode="max",
                trial_name_creator=(
                    lambda trial: (
                        f"{OUTPUT_NAME}_"
                        f"{env_name}_"
                        f"{seed}_"
                        f"{trial.trial_id}"
                    )
                ),
                trial_dirname_creator=(
                    lambda trial: (
                        f"{OUTPUT_NAME}_"
                        f"{env_name}_"
                        f"{seed}_"
                        f"{trial.trial_id}"
                    )
                ),
            ),
            param_space=(
                sac_config.to_dict()
            ),
            run_config=tune.RunConfig(
                name=(
                    f"{OUTPUT_NAME}_"
                    f"{env_name}_"
                    f"Seed{seed}"
                ),
                verbose=0,
                storage_path=str(
                    storage_root
                ),
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

        # Save first, validate second: failed runs remain auditable.
        metrics_path = (
            RESULTS_ROOT
            / "metrics"
            / env_name
            / (
                f"metrics_{OUTPUT_NAME}"
                f"_seed{seed}.csv"
            )
        )
        _extract_metrics(
            results,
            env_name,
            seed,
            metrics_path,
        )

        scheduler_path = (
            RESULTS_ROOT
            / "scheduler"
            / env_name
            / (
                f"scheduler_{OUTPUT_NAME}"
                f"_seed{seed}.csv"
            )
        )
        _extract_pbt_history(
            results,
            env_name,
            seed,
            scheduler_path,
        )

        status_path = (
            RESULTS_ROOT
            / "status"
            / env_name
            / (
                f"status_{OUTPUT_NAME}"
                f"_seed{seed}.csv"
            )
        )
        _save_trial_status_summary(
            results,
            env_name,
            seed,
            target_steps,
            status_path,
        )

        metadata_path = (
            RESULTS_ROOT
            / "metadata"
            / env_name
            / (
                f"metadata_{OUTPUT_NAME}"
                f"_seed{seed}.json"
            )
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
            f"\nCRITICAL ERROR in "
            f"{env_name}, seed={seed}: "
            f"{exc}"
        )
        traceback.print_exc()
        return False

    finally:
        if ray.is_initialized():
            ray.shutdown()

        time.sleep(2)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    started = time.time()
    failures = []

    if PBT_SAC_SMOKE_TEST:
        if PBT_SAC_SMOKE_ENV not in ENVIRONMENTS:
            raise KeyError(
                f"Unknown smoke environment "
                f"{PBT_SAC_SMOKE_ENV!r}. "
                f"Expected one of "
                f"{ENVIRONMENTS}."
            )

        experiment_items = [
            (
                PBT_SAC_SMOKE_ENV,
                PBT_SAC_SMOKE_SEEDS,
                PBT_SAC_SMOKE_STEPS,
                PBT_SAC_SMOKE_INTERVAL,
            )
        ]

    else:
        experiment_items = [
            (
                env_name,
                TRAINING_SEEDS,
                MAX_TIMESTEPS_PER_WORKER,
                PERTURBATION_INTERVAL,
            )
            for env_name in ENVIRONMENTS
        ]

    total = sum(
        len(seeds)
        for _, seeds, _, _
        in experiment_items
    )

    done = 0

    for (
        env_name,
        seeds,
        target_steps,
        interval,
    ) in experiment_items:
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
                quantile_fraction=float(
                    QUANTILE_FRACTION
                ),
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

    elapsed_hours = (
        time.time() - started
    ) / 3600.0

    print("\n" + "-" * 78)
    print(
        f"Experiments finished in "
        f"{elapsed_hours:.2f} hours"
    )

    if failures:
        print(
            f"Failures ({len(failures)}):"
        )
        for failure in failures:
            print(f"  - {failure}")
    else:
        print(
            "All requested experiments "
            "completed successfully."
        )

    print("-" * 78)


if __name__ == "__main__":
    main()
