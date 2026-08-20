#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASHA + SAC baseline using the shared repository configuration.

Scientific role
---------------
ASHA is used as a resource-allocation baseline. Each trial samples one
static SAC configuration at initialization and keeps it unchanged for its
entire lifetime. ASHA may stop underperforming trials early, but it does
not inherit donor checkpoints and does not perform online hyperparameter
adaptation.

For comparability with CODA+SAC, PBT+SAC, and PB2+SAC, this runner keeps
the following components identical whenever applicable:

- SAC old API stack and PyTorch backend;
- four-dimensional raw hyperparameter search domain;
- initial sampling distributions;
- network architecture;
- replay-buffer configuration;
- training intensity;
- rollout fragment length and EnvRunner setup;
- fixed entropy-learning rate;
- baseline target entropy = -action_dim;
- per-trial maximum interaction budget;
- deterministic/reproducibility controls;
- held-out-compatible final checkpointing.

Intentional differences from CODA+SAC
-------------------------------------
1. No population checkpoint inheritance.
2. No online hyperparameter changes.
3. No TV-GP.
4. No I2O diagnostic feedback.
5. No O2I uncertainty actuator.
6. ASHA allocates resources by asynchronous successive halving.

Budget matching
---------------
The current CODA/PBT/PB2 SAC campaign uses P=4 workers x 1,000,000
steps/worker = 4,000,000 nominal interactions per method/environment/seed.

To target the same idealized aggregate budget with ASHA:
    num_samples = 8
    grace_period = 250,000
    reduction_factor = 2
    max_t = 1,000,000

Idealized allocation:
    8 * 0.25M
  + 4 * (0.50M - 0.25M)
  + 2 * (1.00M - 0.50M)
  = 4.00M interactions.

Because ASHA is asynchronous, realized aggregate interactions are logged
and may differ from the nominal target. The number of ASHA initial trials
must be frozen before the final campaign and must not be selected using
reward/performance.

All common learner/search/budget settings are imported from ``configs/``.
Only ASHA-specific resource allocation, static-configuration validation,
budget accounting, and result handling remain local to this script.
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
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.backends.cudnn as cudnn


logger = logging.getLogger(__name__)


# =============================================================================
# REPOSITORY / SHARED CONFIGURATION
# =============================================================================

# Intended location:
#     <repo>/experiments/sac/train_asha.py
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
    MAX_TIMESTEPS_PER_WORKER,
    NOMINAL_AGGREGATE_BUDGET,
    ASHA_NUM_SAMPLES,
    ASHA_GRACE_PERIOD,
    ASHA_REDUCTION_FACTOR,
    ASHA_BRACKETS,
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

logger = logging.getLogger(__name__)


# =============================================================================
# METHOD-SPECIFIC CONFIGURATION
# =============================================================================

ALGO_NAME = "ASHA_SAC"

# Kept for backward-compatible result names used by the current evaluator.
RUN_TAG = os.environ.get("ASHA_SAC_RUN_TAG", "PB2SPACE").strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Compatibility alias; source of truth is configs/sac_config.py.
TIMESTEPS_MAX = MAX_TIMESTEPS_PER_WORKER

RESULTS_ROOT = REPO_ROOT / "results"

ENVIRONMENT_CONFIG: Dict[str, dict] = {
    env_name: {
        "semillas": list(TRAINING_SEEDS),
    }
    for env_name in ENVIRONMENTS
}


# =============================================================================
# SAFE SMOKE TEST
# =============================================================================

# Default: smoke test enabled.
# Set ASHA_SAC_SMOKE=0 to run the configured campaign.
ASHA_SAC_SMOKE_TEST = os.environ.get("ASHA_SAC_SMOKE", "1") != "0"
ASHA_SAC_SMOKE_ENV = os.environ.get(
    "ASHA_SAC_SMOKE_ENV",
    "Hopper-v5",
)
ASHA_SAC_SMOKE_SEEDS = [
    int(
        os.environ.get(
            "ASHA_SAC_SMOKE_SEED",
            str(SMOKE_TRAINING_SEED),
        )
    )
]
ASHA_SAC_SMOKE_STEPS = int(
    os.environ.get("ASHA_SAC_SMOKE_STEPS", "500000")
)

# Scale the smoke grace period to preserve the same 1/4 max_t ratio.
ASHA_SAC_SMOKE_GRACE = int(
    os.environ.get(
        "ASHA_SAC_SMOKE_GRACE",
        str(max(1, ASHA_SAC_SMOKE_STEPS // 4)),
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
# ASHA-SAC DIAGNOSTICS CALLBACK
# =============================================================================

class ASHASACDiagnosticsCallback(DefaultCallbacks):
    """Audit the static SAC configuration and log learner diagnostics.

    ASHA receives only METRIC from Tune's scheduler interface.
    TD error and learner-loss diagnostics are observational only.
    No diagnostic quantity feeds back into ASHA.
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
                optimization.get("actor_learning_rate", np.nan),
                np.nan,
            ),
            "critic_lr": _safe_float(
                optimization.get("critic_learning_rate", np.nan),
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
                _config_attr(cfg, "target_entropy", np.nan),
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

            custom[f"asha_sac_desired_{name}"] = d
            custom[f"asha_sac_effective_{name}"] = e

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

            custom[f"asha_sac_mismatch_{name}"] = float(mismatch)
            mismatch_count += int(mismatch)

        custom[
            "asha_sac_effective_hp_mismatch_count"
        ] = float(mismatch_count)

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs,
    ):
        custom = result.setdefault("custom_metrics", {})

        # Audit only. Unlike PBT/PB2/CODA, ASHA has no donor restore path.
        self._audit_effective_hyperparams(
            algorithm,
            custom,
        )

        policy_data, learner_stats = (
            self._extract_policy_and_learner_stats(result)
        )

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
            td_errors = np.empty((0,), dtype=np.float64)
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
                    "ASHA-SAC | batch=%s tau=%.5g "
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
                        _config_attr(cfg, "tau", np.nan)
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
                            "asha_sac_effective_hp_mismatch_count",
                            0.0,
                        )
                    ),
                )
            except Exception:
                pass


# =============================================================================
# SAC CONFIGURATION
# =============================================================================

# The shared SAC learner and Tune initialization distributions are built by
# configs.sac_config.build_tunable_sac_config(). ASHA-specific resource
# allocation and static-configuration auditing remain in this file.


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

        "custom_metrics/asha_sac_desired_train_batch_size",
        "custom_metrics/asha_sac_effective_train_batch_size",
        "custom_metrics/asha_sac_mismatch_train_batch_size",

        "custom_metrics/asha_sac_desired_tau",
        "custom_metrics/asha_sac_effective_tau",
        "custom_metrics/asha_sac_mismatch_tau",

        "custom_metrics/asha_sac_desired_actor_lr",
        "custom_metrics/asha_sac_effective_actor_lr",
        "custom_metrics/asha_sac_mismatch_actor_lr",

        "custom_metrics/asha_sac_desired_critic_lr",
        "custom_metrics/asha_sac_effective_critic_lr",
        "custom_metrics/asha_sac_mismatch_critic_lr",

        "custom_metrics/asha_sac_desired_alpha_lr",
        "custom_metrics/asha_sac_effective_alpha_lr",
        "custom_metrics/asha_sac_mismatch_alpha_lr",

        "custom_metrics/asha_sac_desired_target_entropy",
        "custom_metrics/asha_sac_effective_target_entropy",
        "custom_metrics/asha_sac_mismatch_target_entropy",

        "custom_metrics/asha_sac_effective_hp_mismatch_count",

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

        # ASHA has no donor inheritance, but keep causal_order for downstream
        # analysis parity with the other methods.
        out["causal_order"] = np.arange(
            len(out),
            dtype=np.int64,
        )
        out["entorno"] = env_name
        out["semilla"] = int(seed)
        out["agente_id"] = f"Trial_{idx + 1}"

        frames.append(out)

    if not frames:
        return

    final = pd.concat(frames, ignore_index=True)
    final = final.sort_values(
        ["agente_id", "causal_order"],
        kind="stable",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    final.to_csv(output_path, index=False)


def _validate_static_asha_configs(result_grid) -> None:
    """Require every ASHA trial to keep its sampled configuration static."""
    problems = []

    config_columns = [
        "config/train_batch_size",
        "config/tau",
        "config/optimization/actor_learning_rate",
        "config/optimization/critic_learning_rate",
        "config/optimization/entropy_learning_rate",
        "config/target_entropy",
    ]

    for idx, result in enumerate(result_grid):
        trial_name = f"Trial_{idx + 1}"
        df = result.metrics_dataframe

        if df is None or df.empty:
            problems.append(
                f"{trial_name}: empty metrics dataframe"
            )
            continue

        for column in config_columns:
            if column not in df.columns:
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )
            finite = values[np.isfinite(values)]

            if finite.empty:
                continue

            # Integer batch can be tested exactly after rounding.
            if column == "config/train_batch_size":
                unique = np.unique(
                    np.rint(finite.to_numpy()).astype(np.int64)
                )
            else:
                arr = finite.to_numpy(dtype=np.float64)
                ref = arr[0]
                changed = ~np.isclose(
                    arr,
                    ref,
                    rtol=1e-12,
                    atol=1e-15,
                )
                unique = np.array(
                    [ref, arr[changed][0]]
                    if np.any(changed)
                    else [ref]
                )

            if len(unique) > 1:
                problems.append(
                    f"{trial_name}: ASHA config changed in {column}"
                )

        mismatch_col = (
            "custom_metrics/"
            "asha_sac_effective_hp_mismatch_count"
        )

        if mismatch_col in df.columns:
            mismatch = pd.to_numeric(
                df[mismatch_col],
                errors="coerce",
            )
            finite = mismatch[np.isfinite(mismatch)]

            if not finite.empty and (finite != 0).any():
                problems.append(
                    f"{trial_name}: effective SAC config mismatch detected"
                )

    if problems:
        raise RuntimeError(
            "ASHA-SAC static-configuration audit failed:\n  - "
            + "\n  - ".join(problems)
        )


def _save_asha_summary(
    result_grid,
    env_name: str,
    seed: int,
    *,
    target_steps: int,
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe

        final_steps = np.nan
        final_reward = np.nan
        n_rows = 0

        if df is not None and not df.empty:
            n_rows = int(len(df))

            if TIME_ATTR in df.columns:
                values = pd.to_numeric(
                    df[TIME_ATTR],
                    errors="coerce",
                )
                finite = values[np.isfinite(values)]
                if not finite.empty:
                    final_steps = float(finite.iloc[-1])

            if METRIC in df.columns:
                rewards = pd.to_numeric(
                    df[METRIC],
                    errors="coerce",
                )
                finite = rewards[np.isfinite(rewards)]
                if not finite.empty:
                    final_reward = float(finite.iloc[-1])

        reached_max = bool(
            np.isfinite(final_steps)
            and final_steps >= target_steps
        )

        error = getattr(result, "error", None)

        rows.append(
            {
                "environment": env_name,
                "training_seed": int(seed),
                "trial_id": f"Trial_{idx + 1}",
                "n_metric_rows": n_rows,
                "final_timesteps_total": final_steps,
                "final_training_return": final_reward,
                "reached_max_resource": int(reached_max),
                "has_checkpoint": int(bool(result.checkpoint)),
                "checkpoint_path": (
                    str(result.checkpoint.path)
                    if result.checkpoint
                    else ""
                ),
                "result_path": str(
                    getattr(result, "path", "")
                ),
                "error": (
                    repr(error)
                    if error is not None
                    else ""
                ),
            }
        )

    summary = pd.DataFrame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(output_path, index=False)

    return summary


def _realized_budget_from_summary(
    summary: pd.DataFrame,
) -> float:
    """Sum final trial resources; valid because ASHA has no inheritance."""
    if summary is None or summary.empty:
        return 0.0

    values = pd.to_numeric(
        summary["final_timesteps_total"],
        errors="coerce",
    ).fillna(0.0)

    return float(values.sum())


def _copy_full_budget_checkpoints(
    result_grid,
    env_name: str,
    seed: int,
    *,
    target_steps: int,
) -> None:
    """Copy checkpoints and explicitly mark full-budget champion candidates."""
    root = (
        (RESULTS_ROOT / "champions")
        / env_name
        / f"{OUTPUT_NAME}_seed{seed}"
    )
    root.mkdir(parents=True, exist_ok=True)

    candidates = []

    for idx, result in enumerate(result_grid):
        trial_name = f"Trial_{idx + 1}"
        df = result.metrics_dataframe

        final_t = np.nan
        final_reward = np.nan

        if df is not None and not df.empty:
            if TIME_ATTR in df.columns:
                values = pd.to_numeric(
                    df[TIME_ATTR],
                    errors="coerce",
                )
                finite = values[np.isfinite(values)]
                if not finite.empty:
                    final_t = float(finite.iloc[-1])

            if METRIC in df.columns:
                rewards = pd.to_numeric(
                    df[METRIC],
                    errors="coerce",
                )
                finite = rewards[np.isfinite(rewards)]
                if not finite.empty:
                    final_reward = float(finite.iloc[-1])

        reached_max = bool(
            np.isfinite(final_t)
            and final_t >= target_steps
        )

        if reached_max:
            candidates.append(
                {
                    "agente_id": trial_name,
                    "final_timesteps_total": final_t,
                    "final_training_return": final_reward,
                }
            )

        if not result.checkpoint:
            continue

        source = Path(result.checkpoint.path)
        target = root / trial_name

        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        except Exception as exc:
            logger.warning(
                "Could not copy checkpoint %s -> %s: %s",
                source,
                target,
                exc,
            )

    candidates_path = root / "full_budget_candidates.json"
    candidates_path.write_text(
        json.dumps(candidates, indent=2),
        encoding="utf-8",
    )

    if not candidates:
        raise RuntimeError(
            "No ASHA-SAC trial reached the full resource budget. "
            "The seed cannot produce an eligible held-out champion."
        )


def _validate_result_grid(
    result_grid,
    *,
    target_steps: int,
) -> None:
    """ASHA validity gate.

    Early-stopped trials are expected. A valid ASHA seed requires:
    - all ResultGrid entries to be readable;
    - no Tune/RLlib errors;
    - static configurations;
    - at least one full-budget trial;
    - a checkpoint for every full-budget candidate.
    """
    results = list(result_grid)
    problems = []
    n_full = 0

    if len(results) != ASHA_NUM_SAMPLES:
        problems.append(
            f"expected {ASHA_NUM_SAMPLES} ASHA trials, "
            f"ResultGrid contains {len(results)}"
        )

    for idx, result in enumerate(results):
        trial_name = f"Trial_{idx + 1}"
        error = getattr(result, "error", None)

        if error is not None:
            problems.append(
                f"{trial_name}: Tune/RLlib error={error!r}"
            )

        df = result.metrics_dataframe
        if df is None or df.empty:
            problems.append(
                f"{trial_name}: empty metrics dataframe"
            )
            continue

        terminal = np.nan

        if TIME_ATTR not in df.columns:
            problems.append(
                f"{trial_name}: missing {TIME_ATTR}"
            )
        else:
            values = pd.to_numeric(
                df[TIME_ATTR],
                errors="coerce",
            )
            finite = values[np.isfinite(values)]
            if not finite.empty:
                terminal = float(finite.iloc[-1])

        if np.isfinite(terminal) and terminal >= target_steps:
            n_full += 1
            if not result.checkpoint:
                problems.append(
                    f"{trial_name}: full-budget trial missing final checkpoint"
                )

    if n_full < 1:
        problems.append(
            "no ASHA trial reached the full resource budget"
        )

    if problems:
        raise RuntimeError(
            "Invalid ASHA-SAC seed:\n  - "
            + "\n  - ".join(problems)
        )


def _save_metadata(
    path: Path,
    *,
    env_name: str,
    seed: int,
    target_steps: int,
    grace_period: int,
    action_dim: int,
    base_target_entropy: float,
    realized_budget: Optional[float] = None,
) -> None:
    realized_value = (
        float(realized_budget)
        if realized_budget is not None
        and np.isfinite(realized_budget)
        else None
    )

    target_budget = (
        NOMINAL_AGGREGATE_BUDGET
        if target_steps == TIMESTEPS_MAX
        else None
    )

    budget_ratio = (
        realized_value / target_budget
        if realized_value is not None
        and target_budget is not None
        and target_budget > 0
        else None
    )

    payload = {
        "algorithm": ALGO_NAME,
        "run_name": OUTPUT_NAME,
        "environment": env_name,
        "seed": int(seed),

        "num_samples": int(ASHA_NUM_SAMPLES),
        "time_attr": TIME_ATTR,
        "metric": METRIC,
        "mode": "max",

        "max_t": int(target_steps),
        "grace_period": int(grace_period),
        "reduction_factor": ASHA_REDUCTION_FACTOR,
        "brackets": ASHA_BRACKETS,

        "target_aggregate_budget": target_budget,
        "realized_aggregate_budget": realized_value,
        "realized_to_target_budget_ratio": budget_ratio,

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
        },

        "initial_sampling": {
            "train_batch_size": "randint[256,2048]",
            "tau": "loguniform[1e-3,2e-2]",
            "actor_learning_rate": "loguniform[1e-5,3e-4]",
            "critic_learning_rate": "loguniform[1e-4,1e-3]",
        },

        "static_hyperparameters_during_trial": True,

        "sac": {
            "gamma": FIXED_GAMMA,
            "initial_alpha": INITIAL_ALPHA,
            "n_step": N_STEP,
            "target_network_update_freq": TARGET_NETWORK_UPDATE_FREQ,
            "learning_starts": LEARNING_STARTS,
            "training_intensity": TRAINING_INTENSITY,
            "twin_q": TWIN_Q,

            "entropy_learning_rate": FIXED_ENTROPY_LR,
            "action_dim": int(action_dim),
            "target_entropy": float(base_target_entropy),
            "target_entropy_policy": (
                "fixed baseline -action_dim; no CODA O2I modulation"
            ),

            "replay_buffer_type": "MultiAgentPrioritizedReplayBuffer",
            "replay_buffer_capacity": REPLAY_BUFFER_CAPACITY,
            "store_buffer_in_checkpoints": STORE_BUFFER_IN_CHECKPOINTS,

            "model_hiddens": MODEL_HIDDENS,
            "model_activation": MODEL_ACTIVATION,

            "num_env_runners": NUM_ENV_RUNNERS,
            "num_envs_per_env_runner": NUM_ENVS_PER_RUNNER,
            "rollout_fragment_length": ROLLOUT_FRAGMENT_LENGTH,
            "min_sample_timesteps_per_iteration": (
                MIN_SAMPLE_TIMESTEPS_PER_ITERATION
            ),
            "observation_filter": OBSERVATION_FILTER,
            "num_gpus_per_trial": NUM_GPUS_PER_TRIAL,
            "old_api_stack": True,
        },

        "asha": {
            "resource_allocation_only": True,
            "checkpoint_inheritance": False,
            "online_hyperparameter_adaptation": False,
            "i2o": False,
            "o2i": False,
        },

        "checkpointing": {
            "num_to_keep": 1,
            "checkpoint_at_end": True,
            "store_buffer_in_checkpoints": STORE_BUFFER_IN_CHECKPOINTS,
        },

        "audit": {
            "causal_order_exported": True,
            "static_hyperparameter_validation": True,
            "effective_hyperparameter_audit": True,
            "realized_budget_recorded": True,
            "full_budget_candidates_recorded": True,
        },

        "versions": {
            "python": sys.version,
            "ray": ray.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "gymnasium": _package_version("gymnasium"),
            "mujoco": _package_version("mujoco"),
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload, indent=2),
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
    grace_period: int,
) -> bool:
    print("\n" + "=" * 78)
    print(
        f"Starting {OUTPUT_NAME} | "
        f"env={env_name} | "
        f"seed={seed} | "
        f"initial_trials={ASHA_NUM_SAMPLES} | "
        f"max_t={target_steps:,} | "
        f"grace={grace_period:,}"
    )
    print("=" * 78)

    _seed_everything(seed)

    action_dim = action_dimension(env_name)
    base_target_entropy = sac_baseline_target_entropy(env_name)

    scheduler = ASHAScheduler(
        time_attr=TIME_ATTR,
        metric=METRIC,
        mode="max",
        max_t=int(target_steps),
        grace_period=int(grace_period),
        reduction_factor=ASHA_REDUCTION_FACTOR,
        brackets=ASHA_BRACKETS,
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
            callbacks=ASHASACDiagnosticsCallback,
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
                num_samples=ASHA_NUM_SAMPLES,
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
                    TIME_ATTR: int(target_steps)
                },
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
            ),
        )

        results = tuner.fit()

        # ASHA trials must remain static.
        _validate_static_asha_configs(results)

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

        summary_path = (
            (RESULTS_ROOT / "scheduler")
            / env_name
            / f"scheduler_{OUTPUT_NAME}_seed{seed}.csv"
        )
        summary = _save_asha_summary(
            results,
            env_name,
            seed,
            target_steps=target_steps,
            output_path=summary_path,
        )

        realized_budget = _realized_budget_from_summary(
            summary
        )

        _copy_full_budget_checkpoints(
            results,
            env_name,
            seed,
            target_steps=target_steps,
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
            grace_period=grace_period,
            action_dim=action_dim,
            base_target_entropy=base_target_entropy,
            realized_budget=realized_budget,
        )

        _validate_result_grid(
            results,
            target_steps=target_steps,
        )

        n_full = int(
            summary["reached_max_resource"].sum()
        )

        if target_steps == TIMESTEPS_MAX:
            budget_ratio = (
                realized_budget
                / NOMINAL_AGGREGATE_BUDGET
            )
            print(
                "ASHA-SAC realized interaction budget: "
                f"{realized_budget:,.0f} | "
                f"target={NOMINAL_AGGREGATE_BUDGET:,.0f} | "
                f"ratio={budget_ratio:.4f}"
            )
        else:
            print(
                "ASHA-SAC smoke realized interaction budget: "
                f"{realized_budget:,.0f}"
            )

        print(
            f"Completed {OUTPUT_NAME}: "
            f"{env_name}, seed={seed} | "
            f"full-budget trials="
            f"{n_full}/{ASHA_NUM_SAMPLES} | "
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

    if ASHA_SAC_SMOKE_TEST:
        if ASHA_SAC_SMOKE_ENV not in ENVIRONMENT_CONFIG:
            raise KeyError(
                "Unknown smoke environment: "
                f"{ASHA_SAC_SMOKE_ENV}"
            )

        experiment_items = [
            (
                ASHA_SAC_SMOKE_ENV,
                ENVIRONMENT_CONFIG[
                    ASHA_SAC_SMOKE_ENV
                ],
            )
        ]
        total = len(ASHA_SAC_SMOKE_SEEDS)

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
        if ASHA_SAC_SMOKE_TEST:
            seeds: Sequence[int] = (
                ASHA_SAC_SMOKE_SEEDS
            )
            target_steps = ASHA_SAC_SMOKE_STEPS
            grace_period = ASHA_SAC_SMOKE_GRACE
        else:
            seeds = env_cfg["semillas"]
            target_steps = TIMESTEPS_MAX
            grace_period = ASHA_GRACE_PERIOD

        for seed in seeds:
            ok = run_experiment(
                env_name,
                int(seed),
                target_steps=int(target_steps),
                grace_period=int(grace_period),
            )

            if not ok:
                failures.append(
                    f"{env_name} - seed {seed}"
                )

            done += 1
            print(
                f"Global progress: {done}/{total}"
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
            "All requested experiments completed successfully."
        )

    print("-" * 78)


if __name__ == "__main__":
    main()
