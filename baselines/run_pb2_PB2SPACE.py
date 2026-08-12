#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:58:08 2026

@author: yor5
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
from typing import Optional

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers.pb2 import PB2
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
    POBLACION_B,
)

ALGO_NAME = "PB2"

# Keep the outputs aligned with the final CODA/PBT search-space campaign.
# Set PB2_RUN_TAG="" if you later want canonical filenames such as
# metrics_PB2_seed1042.csv.
RUN_TAG = os.environ.get("PB2_RUN_TAG", "PB2SPACE").strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Same four-dimensional PPO search space used by the PB2 paper and by
# the final CODA comparison.
HYPERPARAM_BOUNDS = {
    "train_batch_size": [1000, 60000],
    # PPOConfig uses lambda_ in the builder API, while to_dict() exposes
    # the Tune-compatible config key "lambda".
    "lambda": [0.90, 0.99],
    "clip_param": [0.10, 0.50],
    "lr": [1e-5, 1e-3],
}

# Fixed across PBT, PB2, ASHA, and CODA.
FIXED_VF_LOSS_COEFF = 0.5

# Entropy is excluded from the HPO space in PB2 so that it is reserved
# exclusively as the O2I actuator in CODA.
FIXED_ENTROPY_COEFF = 0.0


# -----------------------------------------------------------------------------
# Hopper smoke-test controls
# -----------------------------------------------------------------------------
# Set True for a single-environment validation before the full campaign.
HOPPER_SMOKE_TEST = False
HOPPER_TEST_ENV = "Hopper-v5"
HOPPER_TEST_SEEDS = [1042]


# -----------------------------------------------------------------------------
# Diagnostics callback
# -----------------------------------------------------------------------------
class PB2DiagnosticsCallback(DefaultCallbacks):
    """Record PPO diagnostics without feeding them back to PB2."""

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
# PB2 executable-config guard
# -----------------------------------------------------------------------------
def _pb2_explore_guard(config: dict) -> dict:
    """
    Convert PB2's continuous GP proposal into an executable PPO config.

    PB2 itself respects hyperparam_bounds. This function mainly ensures that
    train_batch_size is integer-valued and that the fixed coefficients remain
    fixed after checkpoint exploitation.
    """
    out = dict(config)

    if "train_batch_size" in out:
        out["train_batch_size"] = int(
            np.clip(
                int(round(float(out["train_batch_size"]))),
                HYPERPARAM_BOUNDS["train_batch_size"][0],
                HYPERPARAM_BOUNDS["train_batch_size"][1],
            )
        )

    if "lambda" in out:
        out["lambda"] = float(
            np.clip(
                float(out["lambda"]),
                HYPERPARAM_BOUNDS["lambda"][0],
                HYPERPARAM_BOUNDS["lambda"][1],
            )
        )

    if "clip_param" in out:
        out["clip_param"] = float(
            np.clip(
                float(out["clip_param"]),
                HYPERPARAM_BOUNDS["clip_param"][0],
                HYPERPARAM_BOUNDS["clip_param"][1],
            )
        )

    if "lr" in out:
        out["lr"] = float(
            np.clip(
                float(out["lr"]),
                HYPERPARAM_BOUNDS["lr"][0],
                HYPERPARAM_BOUNDS["lr"][1],
            )
        )

    # These are not PB2 search coordinates.
    out["entropy_coeff"] = FIXED_ENTROPY_COEFF
    out["vf_loss_coeff"] = FIXED_VF_LOSS_COEFF

    return out


# -----------------------------------------------------------------------------
# Experiment helpers
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


def _save_metadata(
    path: Path,
    *,
    env_name: str,
    seed: int,
    w_pb2_params: dict,
) -> None:
    payload = {
        "algorithm": ALGO_NAME,
        "run_name": OUTPUT_NAME,
        "environment": env_name,
        "seed": int(seed),
        "population": int(POBLACION_B),
        "max_timesteps_per_worker": int(TIMESTEPS_MAX),
        "optimized_hyperparameters": [
            "train_batch_size",
            "lambda",
            "clip_param",
            "lr",
        ],
        "hyperparameter_bounds": HYPERPARAM_BOUNDS,
        "fixed_entropy_coeff": FIXED_ENTROPY_COEFF,
        "fixed_vf_loss_coeff": FIXED_VF_LOSS_COEFF,
        "perturbation_interval": int(
            w_pb2_params.get("perturbation_interval", 50_000)
        ),
        "quantile_fraction": float(
            w_pb2_params.get("quantile_fraction", 0.25)
        ),
        "population_adaptation": "asynchronous",
        "pb2_custom_explore_guard": {
            "integer_train_batch_size": True,
            "clips_to_search_bounds": True,
            "forces_entropy_coeff": FIXED_ENTROPY_COEFF,
            "forces_vf_loss_coeff": FIXED_VF_LOSS_COEFF,
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
    return (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .callbacks(PB2DiagnosticsCallback)
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
            # Identical initial distributions across PBT/PB2/CODA.
            train_batch_size=tune.randint(1000, 60001),
            lr=tune.loguniform(1e-5, 1e-3),
            lambda_=tune.uniform(0.90, 0.99),
            clip_param=tune.uniform(0.10, 0.50),

            # Fixed, not optimized by PB2.
            entropy_coeff=FIXED_ENTROPY_COEFF,
            vf_loss_coeff=FIXED_VF_LOSS_COEFF,

            minibatch_size=512,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
            gamma=0.999,
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
        out["entorno"] = env_name
        out["semilla"] = seed
        out["agente_id"] = f"Agente_{idx + 1}"
        frames.append(out)

    if frames:
        final = pd.concat(frames, ignore_index=True)

        # Keep causal execution order within each Ray trial. Do not sort by
        # timesteps_total because checkpoint inheritance can decrease it.
        sort_cols = ["agente_id"]
        if "training_iteration" in final.columns:
            sort_cols.append("training_iteration")

        final = final.sort_values(sort_cols)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_path, index=False)


def _copy_final_checkpoints(
    result_grid,
    env_name: str,
    seed: int,
) -> None:
    root = (
        Path("./results/champions")
        / env_name
        / f"{OUTPUT_NAME}_seed{seed}"
    )
    root.mkdir(parents=True, exist_ok=True)

    best_reward = -np.inf
    best_agent = None

    for idx, result in enumerate(result_grid):
        if not result.checkpoint:
            continue

        raw = result.metrics or {}
        reward = np.nan

        if isinstance(raw.get("env_runners"), dict):
            reward = raw["env_runners"].get(
                "episode_return_mean",
                np.nan,
            )

        if not np.isfinite(float(reward)):
            reward = raw.get("episode_reward_mean", np.nan)

        reward = (
            float(reward)
            if reward is not None
            else np.nan
        )

        if np.isfinite(reward) and reward > best_reward:
            best_reward = reward
            best_agent = f"Agente_{idx + 1}"

        source = Path(result.checkpoint.path)
        target = root / f"Agente_{idx + 1}"

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

    if best_agent:
        print(
            f"Best final worker: {best_agent} | "
            f"return={best_reward:.3f}"
        )


def run_experiment(
    env_name: str,
    seed: int,
    w_pb2_params: dict,
) -> bool:
    print("\n" + "=" * 72)
    print(
        f"Starting {OUTPUT_NAME} | env={env_name} | "
        f"seed={seed} | population={POBLACION_B}"
    )
    print("=" * 72)

    _seed_everything(seed)

    perturbation_interval = int(
        w_pb2_params.get("perturbation_interval", 50_000)
    )
    quantile_fraction = float(
        w_pb2_params.get("quantile_fraction", 0.25)
    )

    scheduler = PB2(
        time_attr=TIME_ATTR,
        # metric/mode are supplied by TuneConfig below.
        perturbation_interval=perturbation_interval,
        hyperparam_bounds=HYPERPARAM_BOUNDS,
        quantile_fraction=quantile_fraction,
        log_config=True,
        synch=False,
        custom_explore_fn=_pb2_explore_guard,
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

        ppo_config = _build_ppo_config(env_name, seed)

        storage_root = Path(
            f"./results/ray_tune_logs/{OUTPUT_NAME}"
        ).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)

        tuner = tune.Tuner(
            "PPO",
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
            # Keep legacy dict conversion so PPO's lambda_ builder field is
            # exposed to Tune/PB2 as the config key "lambda".
            param_space=ppo_config.to_dict(),
            run_config=tune.RunConfig(
                name=f"{OUTPUT_NAME}_{env_name}_Seed{seed}",
                verbose=0,
                storage_path=str(storage_root),
                stop={TIME_ATTR: TIMESTEPS_MAX},
            ),
        )

        results = tuner.fit()

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

        scheduler_path = (
            Path("./results/scheduler")
            / env_name
            / f"scheduler_{OUTPUT_NAME}_seed{seed}.csv"
        )
        scheduler_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        scheduler_data = getattr(scheduler, "data", None)
        if isinstance(scheduler_data, pd.DataFrame):
            scheduler_data.to_csv(
                scheduler_path,
                index=False,
            )
        else:
            logger.warning(
                "PB2 scheduler.data is unavailable; native Ray PB2 logs "
                "(pb2_global.txt / pb2_policy_*.txt) remain in the Tune "
                "experiment directory."
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
            w_pb2_params=w_pb2_params,
        )

        _copy_final_checkpoints(
            results,
            env_name,
            seed,
        )

        print(
            f"Completed {OUTPUT_NAME}: "
            f"{env_name}, seed={seed}"
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

    if HOPPER_SMOKE_TEST:
        if HOPPER_TEST_ENV not in CONFIG_EXPERIMENTOS:
            raise KeyError(
                f"{HOPPER_TEST_ENV!r} is not present in CONFIG_EXPERIMENTOS"
            )

        experiment_items = [
            (
                HOPPER_TEST_ENV,
                CONFIG_EXPERIMENTOS[HOPPER_TEST_ENV],
            )
        ]
        total = len(HOPPER_TEST_SEEDS)

    else:
        experiment_items = list(CONFIG_EXPERIMENTOS.items())
        total = sum(
            len(conf.get("semillas", []))
            for conf in CONFIG_EXPERIMENTOS.values()
        )

    done = 0

    for env_name, env_cfg in experiment_items:
        seeds = (
            HOPPER_TEST_SEEDS
            if HOPPER_SMOKE_TEST
            else env_cfg.get("semillas", [])
        )

        params = env_cfg.get("w_pb2_params", {})

        for seed in seeds:
            ok = run_experiment(
                env_name,
                int(seed),
                params,
            )

            if not ok:
                failures.append(
                    f"{env_name} - seed {seed}"
                )

            done += 1
            print(f"Global progress: {done}/{total}")

    hours = (time.time() - started) / 3600.0

    print("\n" + "-" * 72)
    print(f"Experiments finished in {hours:.2f} hours")

    if failures:
        print(f"Failures ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("All experiments completed successfully.")

    print("-" * 72)


if __name__ == "__main__":
    main()
