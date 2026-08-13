

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
from typing import Optional

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
for candidate in [PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from w_pb2_experimentos_config import (  # noqa: E402
    CONFIG_EXPERIMENTOS,
    TIMESTEPS_MAX,
)

ALGO_NAME = "ASHA"

# Keep output names aligned with the final PBT/PB2/CODA PB2SPACE campaign.
# Set ASHA_RUN_TAG="" for canonical names such as metrics_ASHA_seed1042.csv.
RUN_TAG = os.environ.get("ASHA_RUN_TAG", "PB2SPACE").strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Exact four-dimensional PPO search space used by PBT/PB2/CODA.
HYPERPARAM_BOUNDS = {
    "train_batch_size": [1000, 60000],
    "lambda": [0.90, 0.99],
    "clip_param": [0.10, 0.50],
    "lr": [1e-5, 1e-3],
}

# Fixed across PBT/PB2/ASHA/CODA.
FIXED_VF_LOSS_COEFF = 0.5
FIXED_ENTROPY_COEFF = 0.0
FIXED_GAMMA = 0.99


# -----------------------------------------------------------------------------
# ASHA budget-calibration protocol
# -----------------------------------------------------------------------------
#
# IMPORTANT:
# ASHA is asynchronous. Therefore, the realized interaction budget is not
# guaranteed to equal a simple idealized successive-halving formula.
#
# This configuration is intended as a PILOT calibration:
#   - many initial configurations,
#   - early pruning beginning at 50k interactions,
#   - same maximum per-trial horizon as PBT/PB2/CODA,
#   - realized budget measured explicitly.
#
# After 2-3 pilot runs, inspect the reported realized budget and, if needed,
# adjust ASHA_NUM_SAMPLES ONCE before the final experimental campaign.
#
ASHA_NUM_SAMPLES = 30
ASHA_GRACE_PERIOD = 50_000
ASHA_REDUCTION_FACTOR = 2
ASHA_BRACKETS = 1

# Target aggregate interaction budget used by PBT/PB2/CODA:
# 4 workers x 2M interactions = 8M.
ASHA_TARGET_AGGREGATE_BUDGET = 8_000_000

# Budget-ratio ranges used only for audit/reporting.
ASHA_BUDGET_EXCELLENT_LOW = 0.95
ASHA_BUDGET_EXCELLENT_HIGH = 1.05
ASHA_BUDGET_ACCEPTABLE_LOW = 0.90
ASHA_BUDGET_ACCEPTABLE_HIGH = 1.10


# -----------------------------------------------------------------------------
# Optional smoke-test controls
# -----------------------------------------------------------------------------
HOPPER_SMOKE_TEST = False
HOPPER_TEST_ENV = "Hopper-v5"
HOPPER_TEST_SEEDS = [1042]


# -----------------------------------------------------------------------------
# Diagnostics callback
# -----------------------------------------------------------------------------
class ASHADiagnosticsCallback(DefaultCallbacks):
    """Record PPO diagnostics without feeding them back to ASHA."""

    @staticmethod
    def _safe_float(value, default=np.nan) -> float:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            value = float(value)
        except (TypeError, ValueError, RuntimeError):
            return float(default)

        return value if np.isfinite(value) else float(default)

    @staticmethod
    def _extract_learner_stats(result: dict):
        info = result.get("info", {}) or {}
        learner = info.get("learner", {}) or {}
        policy_data = learner.get("default_policy", {}) or {}

        learner_stats = policy_data.get("learner_stats")
        if isinstance(learner_stats, dict):
            return learner_stats

        if isinstance(policy_data, dict):
            expected = {
                "kl",
                "mean_kl_loss",
                "vf_explained_var",
                "entropy",
                "mean_entropy",
            }
            if expected.intersection(policy_data.keys()):
                return policy_data

        return None

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        custom = result.setdefault("custom_metrics", {})
        learner_stats = self._extract_learner_stats(result)

        if learner_stats is None:
            custom["policy_kl"] = np.nan
            custom["vf_explained_var"] = np.nan
            custom["policy_entropy"] = np.nan
            custom["diagnostics_valid"] = 0.0
            return

        policy_kl = self._safe_float(
            learner_stats.get(
                "kl",
                learner_stats.get("mean_kl_loss", np.nan),
            )
        )
        vf_explained_var = self._safe_float(
            learner_stats.get("vf_explained_var", np.nan)
        )
        policy_entropy = self._safe_float(
            learner_stats.get(
                "entropy",
                learner_stats.get("mean_entropy", np.nan),
            )
        )

        valid = bool(
            np.isfinite(policy_kl)
            and np.isfinite(vf_explained_var)
        )

        custom["policy_kl"] = policy_kl
        custom["vf_explained_var"] = vf_explained_var
        custom["policy_entropy"] = policy_entropy
        custom["diagnostics_valid"] = float(valid)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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


def _budget_status(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "unavailable"

    if ASHA_BUDGET_EXCELLENT_LOW <= ratio <= ASHA_BUDGET_EXCELLENT_HIGH:
        return "excellent"

    if ASHA_BUDGET_ACCEPTABLE_LOW <= ratio <= ASHA_BUDGET_ACCEPTABLE_HIGH:
        return "acceptable"

    if ratio < ASHA_BUDGET_ACCEPTABLE_LOW:
        return "under_target"

    return "over_target"


def _suggest_num_samples_from_budget(
    current_num_samples: int,
    realized_budget: float,
) -> Optional[int]:
    """Budget-only calibration suggestion.

    This does NOT use reward. It simply rescales the number of initial trials
    according to the observed interaction-budget ratio.

    Because ASHA is asynchronous and nonlinear, this is only a pilot heuristic.
    Final ASHA_NUM_SAMPLES should be fixed before the final campaign.
    """
    if not np.isfinite(realized_budget) or realized_budget <= 0:
        return None

    suggested = int(
        round(
            current_num_samples
            * ASHA_TARGET_AGGREGATE_BUDGET
            / realized_budget
        )
    )

    return max(2, suggested)


def _save_metadata(
    path: Path,
    *,
    env_name: str,
    seed: int,
    realized_budget: Optional[float] = None,
) -> None:
    realized_budget_value = (
        float(realized_budget)
        if realized_budget is not None and np.isfinite(realized_budget)
        else None
    )

    realized_budget_ratio = (
        realized_budget_value / ASHA_TARGET_AGGREGATE_BUDGET
        if realized_budget_value is not None
        else None
    )

    suggested_num_samples = (
        _suggest_num_samples_from_budget(
            ASHA_NUM_SAMPLES,
            realized_budget_value,
        )
        if realized_budget_value is not None
        else None
    )

    payload = {
        "algorithm": ALGO_NAME,
        "run_name": OUTPUT_NAME,
        "environment": env_name,
        "seed": int(seed),
        "num_samples": ASHA_NUM_SAMPLES,
        "time_attr": TIME_ATTR,
        "metric": METRIC,
        "mode": "max",
        "max_t": int(TIMESTEPS_MAX),
        "grace_period": ASHA_GRACE_PERIOD,
        "reduction_factor": ASHA_REDUCTION_FACTOR,
        "brackets": ASHA_BRACKETS,
        "target_aggregate_budget": ASHA_TARGET_AGGREGATE_BUDGET,
        "realized_aggregate_budget": realized_budget_value,
        "realized_to_target_budget_ratio": realized_budget_ratio,
        "budget_status": (
            _budget_status(realized_budget_ratio)
            if realized_budget_ratio is not None
            else None
        ),
        "pilot_budget_calibration": {
            "enabled": True,
            "uses_reward": False,
            "suggested_num_samples_from_this_run": suggested_num_samples,
            "note": (
                "Fix ASHA_NUM_SAMPLES before the final campaign. "
                "Do not tune it using performance/reward."
            ),
        },
        "optimized_hyperparameters": [
            "train_batch_size",
            "lambda",
            "clip_param",
            "lr",
        ],
        "hyperparameter_bounds": HYPERPARAM_BOUNDS,
        "initial_sampling": {
            "train_batch_size": "discrete uniform integer",
            "lambda": "uniform",
            "clip_param": "uniform",
            "lr": "log-uniform",
        },
        "static_hyperparameters_during_trial": True,
        "fixed_entropy_coeff": FIXED_ENTROPY_COEFF,
        "fixed_vf_loss_coeff": FIXED_VF_LOSS_COEFF,
        "ppo": {
            "gamma": FIXED_GAMMA,
            "grad_clip": 0.5,
            "minibatch_size": 512,
            "num_sgd_iter": 10,
            "vf_clip_param": 10.0,
            "use_kl_loss": True,
            "kl_coeff": 0.2,
            "kl_target": 0.01,
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
            "num_env_runners": 4,
            "num_envs_per_env_runner": 8,
            "observation_filter": "MeanStdFilter",
            "num_gpus_per_trial": 0.2,
        },
        "checkpointing": {
            "num_to_keep": 1,
            "checkpoint_at_end": True,
        },
        "audit": {
            "causal_order_exported": True,
            "static_hyperparameter_validation": True,
            "gamma_exported_in_metrics": True,
            "realized_budget_recorded": True,
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
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _build_ppo_config(env_name: str, seed: int) -> PPOConfig:
    """PPO config exactly aligned with PBT/PB2/CODA PB2SPACE runners."""
    return (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .callbacks(ASHADiagnosticsCallback)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(num_gpus=0.2)
        .debugging(seed=seed)
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=8,
            observation_filter="MeanStdFilter",
        )
        .training(
            # Same initial distributions as PBT/PB2/CODA.
            train_batch_size=tune.randint(1000, 60001),
            lr=tune.loguniform(1e-5, 1e-3),
            lambda_=tune.uniform(0.90, 0.99),
            clip_param=tune.uniform(0.10, 0.50),

            # Static/fixed controls for ASHA.
            entropy_coeff=FIXED_ENTROPY_COEFF,
            vf_loss_coeff=FIXED_VF_LOSS_COEFF,

            minibatch_size=512,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
            gamma=FIXED_GAMMA,
            grad_clip=0.5,
            num_sgd_iter=10,
            vf_clip_param=10.0,
            model={
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "tanh",
                "vf_share_layers": False,
            },
        )
    )


def _last_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if "causal_order" in df.columns:
        order = pd.to_numeric(df["causal_order"], errors="coerce")
        finite = order.notna()
        if finite.any():
            return df.loc[order[finite].idxmax()]

    if "time_total_s" in df.columns:
        order = pd.to_numeric(df["time_total_s"], errors="coerce")
        finite = order.notna()
        if finite.any():
            return df.loc[order[finite].idxmax()]

    if "training_iteration" in df.columns:
        order = pd.to_numeric(df["training_iteration"], errors="coerce")
        finite = order.notna()
        if finite.any():
            return df.loc[order[finite].idxmax()]

    return df.iloc[-1]


def _safe_numeric(value, default=np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _validate_static_asha_configs(result_grid) -> None:
    """Fail if any ASHA trial changes a hyperparameter during training.

    ASHA is a resource-allocation baseline over static configurations.
    This validation is audit-only and does not alter scheduler behavior.
    """
    hp_columns = [
        "config/train_batch_size",
        "config/lambda",
        "config/clip_param",
        "config/lr",
        "config/entropy_coeff",
        "config/vf_loss_coeff",
        "config/gamma",
    ]

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe

        if df is None or df.empty:
            continue

        for col in hp_columns:
            if col not in df.columns:
                continue

            values = pd.to_numeric(
                df[col],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            ref = float(values.iloc[0])
            array = values.to_numpy(dtype=float)

            if not np.allclose(
                array,
                ref,
                rtol=1e-12,
                atol=1e-15,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"ASHA trial {idx + 1} changed {col}. "
                    f"Observed values: {values.unique()}"
                )


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
        "config/lr",
        "config/entropy_coeff",
        "config/lambda",
        "config/clip_param",
        "config/train_batch_size",
        "config/vf_loss_coeff",
        "config/gamma",
        "info/learner/default_policy/learner_stats/kl",
        "info/learner/default_policy/learner_stats/entropy",
        "info/learner/default_policy/learner_stats/vf_explained_var",
        "info/learner/default_policy/learner_stats/policy_loss",
        "info/learner/default_policy/learner_stats/vf_loss",
        "custom_metrics/policy_kl",
        "custom_metrics/vf_explained_var",
        "custom_metrics/policy_entropy",
        "custom_metrics/diagnostics_valid",
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
            c for c in columns_wanted
            if c in df.columns
        ]

        out = df[present].copy()

        # Explicit execution-order index for a common downstream pipeline.
        out["causal_order"] = np.arange(
            len(out),
            dtype=np.int64,
        )

        out["entorno"] = env_name
        out["semilla"] = seed
        out["agente_id"] = f"Agente_{idx + 1}"
        out["asha_trial_id"] = str(
            getattr(result, "path", "")
        )

        if TIME_ATTR in df.columns:
            max_t = pd.to_numeric(
                df[TIME_ATTR],
                errors="coerce",
            ).max()
            reached = bool(
                np.isfinite(max_t)
                and max_t >= TIMESTEPS_MAX
            )
        else:
            reached = False

        out["asha_reached_max_resource"] = int(reached)
        frames.append(out)

    if frames:
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


def _save_asha_summary(
    result_grid,
    env_name: str,
    seed: int,
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe

        if (
            df is not None
            and not df.empty
            and "causal_order" not in df.columns
        ):
            df = df.copy()
            df["causal_order"] = np.arange(
                len(df),
                dtype=np.int64,
            )

        last = _last_row(df)

        if last is None:
            final_t = np.nan
            final_reward = np.nan
            batch = np.nan
            lambda_ = np.nan
            clip = np.nan
            lr = np.nan
            gamma = np.nan
        else:
            final_t = _safe_numeric(
                last.get(TIME_ATTR, np.nan)
            )
            final_reward = _safe_numeric(
                last.get(METRIC, np.nan)
            )
            batch = _safe_numeric(
                last.get(
                    "config/train_batch_size",
                    np.nan,
                )
            )
            lambda_ = _safe_numeric(
                last.get("config/lambda", np.nan)
            )
            clip = _safe_numeric(
                last.get(
                    "config/clip_param",
                    np.nan,
                )
            )
            lr = _safe_numeric(
                last.get("config/lr", np.nan)
            )
            gamma = _safe_numeric(
                last.get("config/gamma", np.nan)
            )

        rows.append(
            {
                "entorno": env_name,
                "semilla": seed,
                "agente_id": f"Agente_{idx + 1}",
                "final_timesteps_total": final_t,
                "final_training_return": final_reward,
                "reached_max_resource": int(
                    np.isfinite(final_t)
                    and final_t >= TIMESTEPS_MAX
                ),
                "train_batch_size": batch,
                "lambda": lambda_,
                "clip_param": clip,
                "lr": lr,
                "gamma": gamma,
                "has_checkpoint": int(
                    bool(result.checkpoint)
                ),
                "checkpoint_path": (
                    str(result.checkpoint.path)
                    if result.checkpoint
                    else ""
                ),
                "result_path": str(
                    getattr(result, "path", "")
                ),
                "error": (
                    str(result.error)
                    if getattr(result, "error", None)
                    else ""
                ),
            }
        )

    summary = pd.DataFrame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(
        output_path,
        index=False,
    )

    return summary


def _copy_final_checkpoints(
    result_grid,
    env_name: str,
    seed: int,
) -> None:
    """Copy each trial's final checkpoint and identify full-budget candidates.

    Champion selection is intentionally deferred to the common held-out-test
    script, which uses training data only and a pre-specified terminal criterion.
    """
    root = (
        Path("./results/champions")
        / env_name
        / f"{OUTPUT_NAME}_seed{seed}"
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = []

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe

        if (
            df is not None
            and not df.empty
            and "causal_order" not in df.columns
        ):
            df = df.copy()
            df["causal_order"] = np.arange(
                len(df),
                dtype=np.int64,
            )

        last = _last_row(df)

        final_t = (
            _safe_numeric(
                last.get(TIME_ATTR, np.nan)
            )
            if last is not None
            else np.nan
        )

        final_reward = (
            _safe_numeric(
                last.get(METRIC, np.nan)
            )
            if last is not None
            else np.nan
        )

        reached_max = bool(
            np.isfinite(final_t)
            and final_t >= TIMESTEPS_MAX
        )

        agent_name = f"Agente_{idx + 1}"

        if reached_max:
            candidates.append(
                {
                    "agente_id": agent_name,
                    "final_timesteps_total": final_t,
                    "final_training_return": final_reward,
                }
            )

        if not result.checkpoint:
            logger.warning(
                "No final checkpoint available for %s",
                agent_name,
            )
            continue

        source = Path(
            result.checkpoint.path
        )
        target = root / agent_name

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

    candidates_path = (
        root / "full_budget_candidates.json"
    )

    candidates_path.write_text(
        json.dumps(candidates, indent=2),
        encoding="utf-8",
    )

    if candidates:
        finite = [
            x
            for x in candidates
            if np.isfinite(
                x["final_training_return"]
            )
        ]

        if finite:
            best = max(
                finite,
                key=lambda x: x[
                    "final_training_return"
                ],
            )

            print(
                "Best full-budget ASHA trial by last training return "
                "(diagnostic only; not final champion rule): "
                f"{best['agente_id']} | "
                f"return={best['final_training_return']:.3f} | "
                f"T={best['final_timesteps_total']:.0f}"
            )

        else:
            print(
                f"ASHA full-budget candidates: "
                f"{len(candidates)} "
                "(terminal returns unavailable)"
            )

    else:
        logger.warning(
            "No ASHA trial reached TIMESTEPS_MAX=%s "
            "in %s seed=%s",
            TIMESTEPS_MAX,
            env_name,
            seed,
        )


def _realized_budget_from_summary(
    summary: pd.DataFrame,
) -> float:
    """Sum final ASHA trial resources.

    This is valid for ASHA because trials do not inherit checkpoints and
    timesteps_total does not reset to a donor lineage.
    """
    if summary is None or summary.empty:
        return 0.0

    values = pd.to_numeric(
        summary["final_timesteps_total"],
        errors="coerce",
    ).fillna(0.0)

    return float(values.sum())


def _print_budget_report(
    realized_budget: float,
) -> None:
    ratio = (
        realized_budget
        / ASHA_TARGET_AGGREGATE_BUDGET
        if ASHA_TARGET_AGGREGATE_BUDGET > 0
        else np.nan
    )

    status = _budget_status(ratio)

    suggested = _suggest_num_samples_from_budget(
        ASHA_NUM_SAMPLES,
        realized_budget,
    )

    print("\n" + "-" * 72)
    print("ASHA BUDGET AUDIT")
    print(
        f"Current initial trials: {ASHA_NUM_SAMPLES}"
    )
    print(
        f"Grace period: {ASHA_GRACE_PERIOD:,}"
    )
    print(
        f"Realized budget: "
        f"{realized_budget:,.0f}"
    )
    print(
        f"Target budget: "
        f"{ASHA_TARGET_AGGREGATE_BUDGET:,.0f}"
    )
    print(
        f"Budget ratio: {ratio:.4f}"
    )
    print(
        f"Budget status: {status}"
    )

    if suggested is not None:
        print(
            "Budget-only suggested ASHA_NUM_SAMPLES "
            f"for the NEXT pilot: {suggested}"
        )

    print(
        "IMPORTANT: choose the final sample count using budget only, "
        "not reward, and freeze it before the final campaign."
    )
    print("-" * 72)


def run_experiment(
    env_name: str,
    seed: int,
) -> bool:
    print("\n" + "=" * 72)
    print(
        f"Starting {OUTPUT_NAME} | "
        f"env={env_name} | "
        f"seed={seed} | "
        f"initial_trials={ASHA_NUM_SAMPLES} | "
        f"grace={ASHA_GRACE_PERIOD:,}"
    )
    print("=" * 72)

    _seed_everything(seed)

    scheduler = ASHAScheduler(
        time_attr=TIME_ATTR,
        max_t=int(TIMESTEPS_MAX),
        grace_period=ASHA_GRACE_PERIOD,
        reduction_factor=ASHA_REDUCTION_FACTOR,
        brackets=ASHA_BRACKETS,
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required because each PPO trial requests "
                "num_gpus=0.2"
            )

        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
            include_dashboard=False,
        )

        ppo_config = _build_ppo_config(
            env_name,
            seed,
        )

        storage_root = Path(
            f"./results/ray_tune_logs/{OUTPUT_NAME}"
        ).resolve()

        storage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        tuner = tune.Tuner(
            "PPO",
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

            # Legacy dict conversion is kept to expose PPOConfig.lambda_
            # as the Tune-compatible config key "lambda", exactly as in
            # PB2/PBT/CODA.
            param_space=ppo_config.to_dict(),

            run_config=tune.RunConfig(
                name=(
                    f"{OUTPUT_NAME}_{env_name}_"
                    f"Seed{seed}"
                ),
                verbose=0,
                storage_path=str(storage_root),
                stop={
                    TIME_ATTR: TIMESTEPS_MAX
                },
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
            ),
        )

        results = tuner.fit()

        # Audit-only validation:
        # ASHA hyperparameters must remain static.
        _validate_static_asha_configs(
            results
        )

        metrics_path = (
            Path("./results/metrics")
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
            Path("./results/scheduler")
            / env_name
            / f"scheduler_{OUTPUT_NAME}_seed{seed}.csv"
        )

        summary = _save_asha_summary(
            results,
            env_name,
            seed,
            summary_path,
        )

        realized_budget = (
            _realized_budget_from_summary(
                summary
            )
        )

        _print_budget_report(
            realized_budget
        )

        metadata_path = (
            Path("./results/metadata")
            / env_name
            / f"metadata_{OUTPUT_NAME}_seed{seed}.json"
        )

        _save_metadata(
            metadata_path,
            env_name=env_name,
            seed=seed,
            realized_budget=realized_budget,
        )

        _copy_final_checkpoints(
            results,
            env_name,
            seed,
        )

        n_full = (
            int(
                summary[
                    "reached_max_resource"
                ].sum()
            )
            if not summary.empty
            else 0
        )

        print(
            f"Completed {OUTPUT_NAME}: "
            f"{env_name}, seed={seed} | "
            f"full-budget trials="
            f"{n_full}/{ASHA_NUM_SAMPLES}"
        )

        return True

    except Exception as exc:
        print(
            f"\nCRITICAL ERROR in "
            f"{env_name}, seed={seed}: {exc}"
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

    if HOPPER_SMOKE_TEST:
        if HOPPER_TEST_ENV not in CONFIG_EXPERIMENTOS:
            raise KeyError(
                f"{HOPPER_TEST_ENV!r} is not present "
                "in CONFIG_EXPERIMENTOS"
            )

        experiment_items = [
            (
                HOPPER_TEST_ENV,
                CONFIG_EXPERIMENTOS[
                    HOPPER_TEST_ENV
                ],
            )
        ]

        total = len(
            HOPPER_TEST_SEEDS
        )

    else:
        experiment_items = list(
            CONFIG_EXPERIMENTOS.items()
        )

        total = sum(
            len(
                conf.get(
                    "semillas",
                    [],
                )
            )
            for conf
            in CONFIG_EXPERIMENTOS.values()
        )

    done = 0

    for env_name, env_cfg in experiment_items:
        seeds = (
            HOPPER_TEST_SEEDS
            if HOPPER_SMOKE_TEST
            else env_cfg.get(
                "semillas",
                [],
            )
        )

        for seed in seeds:
            ok = run_experiment(
                env_name,
                int(seed),
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

    print("\n" + "-" * 72)
    print(
        f"Experiments finished "
        f"in {hours:.2f} hours"
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
            "All experiments "
            "completed successfully."
        )

    print("-" * 72)


if __name__ == "__main__":
    main()
