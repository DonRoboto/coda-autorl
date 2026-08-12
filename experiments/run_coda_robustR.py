#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run CODA / CODA-I2O / CODA-O2I experiments with RLlib PPO.

Expected project dependency (kept compatible with the user's existing project):
    config.w_pb2_experimentos_config
        - CONFIG_EXPERIMENTOS
        - TIMESTEPS_MAX
        - POBLACION_B

Select the directional variant with the environment variable:
    CODA_VARIANT=full python run_coda.py
    CODA_VARIANT=i2o  python run_coda.py
    CODA_VARIANT=o2i  python run_coda.py
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
import re
import shutil
import sys
import time
import traceback
import warnings
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.train import CheckpointConfig, RunConfig
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.backends.cudnn as cudnn

from coda_scheduler_robustR import CODAScheduler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
# Preserve compatibility if this script is kept two levels below project root.
for candidate in [PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from w_pb2_experimentos_config import (  # noqa: E402
    CONFIG_EXPERIMENTOS,
    TIMESTEPS_MAX,
    POBLACION_B,
)

VARIANT = os.environ.get("CODA_VARIANT", "full").strip().lower()
if VARIANT not in CODAScheduler.VALID_VARIANTS:
    raise ValueError(
        f"CODA_VARIANT must be one of {sorted(CODAScheduler.VALID_VARIANTS)}, got {VARIANT!r}"
    )

ALGO_NAME = {
    "full": "CODA",
    "i2o": "CODA_I2O",
    "o2i": "CODA_O2I",
}[VARIANT]

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Reward context is normalized causally inside CODAScheduler; no task-specific
# reward bounds are required.

HYPERPARAM_BOUNDS = {
    "train_batch_size": [1000, 60000],
    "entropy_coeff": [0.0, 0.01],
    "lr": [1e-5, 1e-3],
    "vf_loss_coeff": [0.01, 0.5],
}


# -----------------------------------------------------------------------------
# Reproducibility / diagnostics callback
# -----------------------------------------------------------------------------
class CODACallback(DefaultCallbacks):
    """Extract PPO diagnostics and maintain a lineage-consistent learner-state EMA."""

    def __init__(self):
        super().__init__()
        self._stability_ema: Optional[float] = None
        self._last_lineage_generation: Optional[int] = None
        self._iter_count = 0

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

    @staticmethod
    def _bridge_from_algorithm(algorithm) -> dict:
        try:
            cfg = algorithm.config
            model_cfg = cfg.model if hasattr(cfg, "model") else cfg.get("model", {})
            model_cfg = model_cfg or {}
            custom_cfg = model_cfg.get("custom_model_config", {}) or {}
            return custom_cfg.get("_coda_bridge", {}) or {}
        except Exception:
            return {}

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

        seed = self._safe_float(bridge.get("lineage_ema_seed", np.nan), np.nan)
        # A valid donor S seeds the receiver EMA.  If donor diagnostics were
        # invalid, restart the EMA from the receiver's first future valid raw S.
        self._stability_ema = float(seed) if np.isfinite(seed) else None
        self._last_lineage_generation = generation

    def on_checkpoint_loaded(self, *, algorithm, **kwargs):
        # Depending on Ray version/reset ordering, the new trial config may be
        # visible here or only by the next on_train_result.  Calling in both
        # places is safe because generation IDs make the operation idempotent.
        self._apply_lineage_seed_if_needed(algorithm)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        self._apply_lineage_seed_if_needed(algorithm)

        custom = result.setdefault("custom_metrics", {})
        learner_stats = self._extract_learner_stats(result)
        bridge = self._bridge_from_algorithm(algorithm)

        # Audit metadata from the scheduler's last adaptation event.
        custom["coda_gp_uncertainty"] = self._safe_float(
            bridge.get("gp_uncertainty", 0.0), 0.0
        )
        custom["coda_guided_update"] = float(bool(bridge.get("guided_update", False)))
        custom["coda_nominal_entropy_coeff"] = self._safe_float(
            bridge.get("nominal_entropy_coeff", np.nan), np.nan
        )
        custom["coda_applied_entropy_coeff"] = self._safe_float(
            bridge.get("applied_entropy_coeff", np.nan), np.nan
        )

        if learner_stats is None:
            for key in (
                "stability_index",
                "stability_raw",
                "actor_health",
                "critic_health",
                "policy_kl",
                "vf_explained_var",
                "policy_entropy",
            ):
                custom[key] = np.nan
            custom["stability_valid"] = 0.0
            return

        policy_kl = self._safe_float(
            learner_stats.get("kl", learner_stats.get("mean_kl_loss", np.nan))
        )
        vf_explained_var = self._safe_float(
            learner_stats.get("vf_explained_var", np.nan)
        )
        policy_entropy = self._safe_float(
            learner_stats.get("entropy", learner_stats.get("mean_entropy", np.nan))
        )

        valid = bool(np.isfinite(policy_kl) and np.isfinite(vf_explained_var))

        if valid:
            reference_kl = 0.01  # matches PPO kl_target below
            actor_health = float(
                np.exp(-max(0.0, max(policy_kl, 0.0) / reference_kl - 1.0))
            )
            clipped_vf = float(np.clip(vf_explained_var, -10.0, 1.0))
            critic_health = float(np.exp(-max(0.0, 1.0 - clipped_vf)))
            stability_raw = float(
                np.clip(actor_health * critic_health, 1e-6, 1.0)
            )

            beta = 0.90
            if self._stability_ema is None:
                self._stability_ema = stability_raw
            else:
                self._stability_ema = float(
                    beta * self._stability_ema + (1.0 - beta) * stability_raw
                )
            stability_index = float(np.clip(self._stability_ema, 1e-6, 1.0))
        else:
            actor_health = np.nan
            critic_health = np.nan
            stability_raw = np.nan
            stability_index = np.nan

        custom["stability_index"] = stability_index
        custom["stability_raw"] = stability_raw
        custom["stability_valid"] = float(valid)
        custom["actor_health"] = actor_health
        custom["critic_health"] = critic_health
        custom["policy_kl"] = policy_kl
        custom["vf_explained_var"] = vf_explained_var
        custom["policy_entropy"] = policy_entropy

        self._iter_count += 1
        if self._iter_count % 5 == 0:
            try:
                cfg = algorithm.config
                entropy_coeff = (
                    float(cfg.entropy_coeff)
                    if hasattr(cfg, "entropy_coeff")
                    else float(cfg.get("entropy_coeff", np.nan))
                )
                logger.info(
                    "PPO diagnostics | KL=%.5f entropy=%.4f coeff=%.5f S=%.4f",
                    policy_kl,
                    policy_entropy,
                    entropy_coeff,
                    stability_index,
                )
            except Exception:
                pass


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

    # Favor reproducibility over cuDNN auto-tuning.  RLlib/Ray scheduling can
    # still introduce nondeterminism, so seeds remain the statistical unit.
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
        "variant": VARIANT,
        "environment": env_name,
        "seed": int(seed),
        "population": int(POBLACION_B),
        "max_timesteps_per_worker": int(TIMESTEPS_MAX),
        "reward_context_normalization": {
            "method": "causal_robust_median_iqr",
            "location": "median of past valid R_before values",
            "scale": "IQR/1.349 with MAD/std fallbacks",
            "z_clip": 4.0,
            "mapped_interval": [0.0, 1.0],
            "uses_future_information": False,
        },
        "hyperparameter_bounds": HYPERPARAM_BOUNDS,
        "perturbation_interval": int(w_pb2_params.get("perturbation_interval", 50_000)),
        "quantile_fraction": float(w_pb2_params.get("quantile_fraction", 0.25)),
        "min_valid_transitions": 2,
        "uncertainty_scale": 0.005,
        "max_entropy_increment": 0.015,
        "entropy_guard": 0.05,
        "effective_max_applied_entropy": 0.025 if VARIANT in {"full", "o2i"} else 0.01,
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


def _build_ppo_config(env_name: str, seed: int) -> PPOConfig:
    return (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .callbacks(CODACallback)
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
            # Initial trial distributions.  Keep the same distributions across
            # all compared methods when rerunning the benchmark.
            train_batch_size=tune.randint(1000, 60001),  # inclusive 60,000
            lr=tune.loguniform(1e-5, 1e-3),
            vf_loss_coeff=tune.uniform(0.01, 0.5),
            entropy_coeff=tune.uniform(0.0, 0.01),
            clip_param=0.2,
            minibatch_size=512,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
            gamma=0.999,
            lambda_=0.95,
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


def _extract_metrics(result_grid, env_name: str, seed: int, output_path: Path) -> None:
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
        "config/train_batch_size",
        "config/vf_loss_coeff",
        "info/learner/default_policy/learner_stats/kl",
        "info/learner/default_policy/learner_stats/entropy",
        "info/learner/default_policy/learner_stats/vf_explained_var",
        "info/learner/default_policy/learner_stats/policy_loss",
        "info/learner/default_policy/learner_stats/vf_loss",
        "custom_metrics/stability_index",
        "custom_metrics/stability_raw",
        "custom_metrics/stability_valid",
        "custom_metrics/actor_health",
        "custom_metrics/critic_health",
        "custom_metrics/policy_kl",
        "custom_metrics/vf_explained_var",
        "custom_metrics/policy_entropy",
        "custom_metrics/coda_gp_uncertainty",
        "custom_metrics/coda_gp_uncertainty_raw",
        "custom_metrics/coda_gp_uncertainty_variance",
        "custom_metrics/coda_guided_update",
        "custom_metrics/coda_nominal_entropy_coeff",
        "custom_metrics/coda_applied_entropy_coeff",
        "perf/gpu_util_percent0",
        "perf/cpu_util_percent",
        "perf/ram_util_percent",
        "timers/training_iteration_time_ms",
    ]

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe
        if df is None or df.empty:
            continue
        present = [c for c in columns_wanted if c in df.columns]
        out = df[present].copy()
        out["entorno"] = env_name
        out["semilla"] = seed
        out["agente_id"] = f"Agente_{idx + 1}"
        frames.append(out)

    if frames:
        final = pd.concat(frames, ignore_index=True)
        final = final.sort_values(["agente_id", "training_iteration"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_path, index=False)


def _copy_final_checkpoints(result_grid, env_name: str, seed: int) -> None:
    root = Path("./results/champions") / env_name / f"{ALGO_NAME}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)

    best_reward = -np.inf
    best_agent = None

    for idx, result in enumerate(result_grid):
        if not result.checkpoint:
            continue

        raw = result.metrics or {}
        reward = np.nan
        if isinstance(raw.get("env_runners"), dict):
            reward = raw["env_runners"].get("episode_return_mean", np.nan)
        if not np.isfinite(float(reward)):
            reward = raw.get("episode_reward_mean", np.nan)
        reward = float(reward) if reward is not None else np.nan

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
            logger.warning("Could not copy checkpoint %s -> %s: %s", source, target, exc)

    if best_agent:
        print(f"Best final worker: {best_agent} | return={best_reward:.3f}")


def run_experiment(env_name: str, seed: int, w_pb2_params: dict) -> bool:
    print("\n" + "=" * 72)
    print(
        f"Starting {ALGO_NAME} | env={env_name} | seed={seed} | population={POBLACION_B}"
    )
    print("=" * 72)

    _seed_everything(seed)
    context_bounds = {
        "T_before": (0.0, float(TIMESTEPS_MAX)),
        "S_before": (0.0, 1.0),
    }

    scheduler = CODAScheduler(
        time_attr=TIME_ATTR,
        #metric=METRIC,
        #mode="max",
        perturbation_interval=50_000,
        quantile_fraction=0.25,
        hyperparam_bounds=HYPERPARAM_BOUNDS,
        context_bounds=context_bounds,
        variant=VARIANT,
        min_valid_transitions=2,
        max_gp_points=1000,
        uncertainty_scale=0.005,
        max_entropy_increment=0.015,
        entropy_guard=0.05,
        reward_z_clip=4.0,
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required because each PPO trial requests num_gpus=0.2")

        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
            include_dashboard=False,
        )

        ppo_config = _build_ppo_config(env_name, seed)
        storage_root = Path(f"./results/ray_tune_logs/{ALGO_NAME}").resolve()
        storage_root.mkdir(parents=True, exist_ok=True)

        tuner = tune.Tuner(
            "PPO",
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=POBLACION_B,
                metric=METRIC,
                mode="max",
                trial_name_creator=lambda trial: (
                    f"{ALGO_NAME}_{env_name}_{seed}_{trial.trial_id}"
                ),
                trial_dirname_creator=lambda trial: (
                    f"{ALGO_NAME}_{env_name}_{seed}_{trial.trial_id}"
                ),
            ),
            param_space=ppo_config,
            run_config=tune.RunConfig(
                name=f"{ALGO_NAME}_{env_name}_Seed{seed}",
                verbose=0,
                storage_path=str(storage_root),
                stop={TIME_ATTR: TIMESTEPS_MAX},
                # checkpoint_config=CheckpointConfig(
                #     checkpoint_at_end=True,
                #     # Keep several checkpoints because PBT/PB2 exploitation may
                #     # need a top trial's recent checkpoint while saves overlap.
                #     num_to_keep=4,
                # ),
            ),
        )

        results = tuner.fit()

        metrics_path = (
            Path("./results/metrics")
            / env_name
            / f"metrics_{ALGO_NAME}_seed{seed}.csv"
        )
        _extract_metrics(results, env_name, seed, metrics_path)

        # Save scheduler-side audit data: lineage anchors, uncertainty, and the
        # exact applied configurations used to fit the GP.
        scheduler_path = (
            Path("./results/scheduler")
            / env_name
            / f"scheduler_{ALGO_NAME}_seed{seed}.csv"
        )
        scheduler_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler.data.to_csv(scheduler_path, index=False)

        metadata_path = (
            Path("./results/metadata")
            / env_name
            / f"metadata_{ALGO_NAME}_seed{seed}.json"
        )
        _save_metadata(
            metadata_path,
            env_name=env_name,
            seed=seed,
            w_pb2_params=w_pb2_params,
        )

        _copy_final_checkpoints(results, env_name, seed)
        print(f"Completed {ALGO_NAME}: {env_name}, seed={seed}")
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
    total = sum(len(conf.get("semillas", [])) for conf in CONFIG_EXPERIMENTOS.values())
    done = 0

    for env_name, env_cfg in CONFIG_EXPERIMENTOS.items():
        seeds = env_cfg.get("semillas", [])
        params = env_cfg.get("w_pb2_params", {})

        for seed in seeds:
            ok = run_experiment(env_name, int(seed), params)
            if not ok:
                failures.append(f"{env_name} - seed {seed}")
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
