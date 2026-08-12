#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.backends.cudnn as cudnn

from coda_scheduler_HIM_PB2SPACE import CODAScheduler

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

VARIANT = os.environ.get("CODA_VARIANT", "full").strip().lower()
if VARIANT not in CODAScheduler.VALID_VARIANTS:
    raise ValueError(
        f"CODA_VARIANT must be one of {sorted(CODAScheduler.VALID_VARIANTS)}, "
        f"got {VARIANT!r}"
    )

ALGO_NAME = {
    "full": "CODA",
    "i2o": "CODA_I2O",
    "o2i": "CODA_O2I",
}[VARIANT]

# Keep smoke-test outputs separate from previous O2I prototypes.
# Set CODA_RUN_TAG="" in the environment if you later want the canonical
# filenames (e.g., metrics_CODA_seedXXXX.csv) for the final campaign.
RUN_TAG = os.environ.get("CODA_RUN_TAG", "PB2SPACE").strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Reward context is normalized causally inside CODAScheduler; no task-specific
# reward bounds are required.
HYPERPARAM_BOUNDS = {
    "train_batch_size": [1000, 60000],
    # RLlib's PPOConfig uses `lambda_`, but AlgorithmConfig.to_dict() exposes
    # the Tune-compatible key as `lambda`. The scheduler therefore optimizes
    # `lambda` in trial.config.
    "lambda": [0.90, 0.99],
    "clip_param": [0.10, 0.50],
    "lr": [1e-5, 1e-3],
}

# Value-function loss weighting is no longer part of the HPO search space.
# Keep it fixed and identical across PBT/PB2/ASHA/CODA in the final campaign.
FIXED_VF_LOSS_COEFF = 0.5

# Entropy is NOT optimized. It is reserved exclusively as the O2I actuator.
BASE_ENTROPY_COEFF = 0.0
O2I_ENTROPY_SCALE = 0.005
O2I_MAX_INCREMENT = 0.005
ENTROPY_GUARD = 0.05

# -----------------------------------------------------------------------------
# Hopper smoke-test controls
# -----------------------------------------------------------------------------
# Leave True for the Hopper validation run. Set False before the final campaign.
HOPPER_SMOKE_TEST = False
HOPPER_TEST_ENV = "Hopper-v5"
HOPPER_TEST_SEEDS = [1111]


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

        seed = self._safe_float(
            bridge.get("lineage_ema_seed", np.nan), np.nan
        )
        self._stability_ema = float(seed) if np.isfinite(seed) else None
        self._last_lineage_generation = generation

    def on_checkpoint_loaded(self, *, algorithm, **kwargs):
        self._apply_lineage_seed_if_needed(algorithm)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        self._apply_lineage_seed_if_needed(algorithm)

        custom = result.setdefault("custom_metrics", {})
        learner_stats = self._extract_learner_stats(result)
        bridge = self._bridge_from_algorithm(algorithm)

        # ---------------------------------------------------------------------
        # Scheduler audit metadata
        # ---------------------------------------------------------------------
        custom["coda_o2i_intervention_magnitude"] = self._safe_float(
            bridge.get("o2i_intervention_magnitude", 0.0), 0.0
        )
        custom["coda_o2i_delta_train_batch_size"] = self._safe_float(
            bridge.get("o2i_delta_train_batch_size", 0.0), 0.0
        )
        custom["coda_o2i_delta_lambda"] = self._safe_float(
            bridge.get("o2i_delta_lambda", 0.0), 0.0
        )
        custom["coda_o2i_delta_lr"] = self._safe_float(
            bridge.get("o2i_delta_lr", 0.0), 0.0
        )
        custom["coda_o2i_delta_clip_param"] = self._safe_float(
            bridge.get("o2i_delta_clip_param", 0.0), 0.0
        )
        custom["coda_entropy_increment"] = self._safe_float(
            bridge.get("entropy_increment", 0.0), 0.0
        )
        custom["coda_guided_update"] = float(
            bool(bridge.get("guided_update", False))
        )
        custom["coda_base_entropy_coeff"] = self._safe_float(
            bridge.get("base_entropy_coeff", BASE_ENTROPY_COEFF),
            BASE_ENTROPY_COEFF,
        )
        custom["coda_nominal_entropy_coeff"] = self._safe_float(
            bridge.get("nominal_entropy_coeff", BASE_ENTROPY_COEFF),
            BASE_ENTROPY_COEFF,
        )
        custom["coda_applied_entropy_coeff"] = self._safe_float(
            bridge.get("applied_entropy_coeff", np.nan), np.nan
        )

        # ---------------------------------------------------------------------
        # PPO learner-state diagnostic
        # ---------------------------------------------------------------------
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

        if valid:
            reference_kl = 0.01  # matches PPO kl_target below
            actor_health = float(
                np.exp(
                    -max(
                        0.0,
                        max(policy_kl, 0.0) / reference_kl - 1.0,
                    )
                )
            )

            clipped_vf = float(
                np.clip(vf_explained_var, -10.0, 1.0)
            )
            critic_health = float(
                np.exp(-max(0.0, 1.0 - clipped_vf))
            )

            stability_raw = float(
                np.clip(
                    actor_health * critic_health,
                    1e-6,
                    1.0,
                )
            )

            beta = 0.90
            if self._stability_ema is None:
                self._stability_ema = stability_raw
            else:
                self._stability_ema = float(
                    beta * self._stability_ema
                    + (1.0 - beta) * stability_raw
                )

            stability_index = float(
                np.clip(self._stability_ema, 1e-6, 1.0)
            )
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
                intervention = custom["coda_o2i_intervention_magnitude"]
                logger.info(
                    "PPO diagnostics | KL=%.5f entropy=%.4f coeff=%.5f "
                    "S=%.4f intervention=%.3f",
                    policy_kl,
                    policy_entropy,
                    entropy_coeff,
                    stability_index,
                    intervention,
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
        "optimized_hyperparameters": [
            "train_batch_size", "lambda", "clip_param", "lr"
        ],
        "fixed_vf_loss_coeff": FIXED_VF_LOSS_COEFF,
        "entropy_role": "O2I actuator only; excluded from outer-loop search",
        "base_entropy_coeff": BASE_ENTROPY_COEFF,
        "perturbation_interval": int(
            w_pb2_params.get("perturbation_interval", 50_000)
        ),
        "quantile_fraction": float(
            w_pb2_params.get("quantile_fraction", 0.25)
        ),
        "min_valid_transitions": 2,
        "max_gp_points": 1000,
        "o2i_feedback": {
            "method": "normalized_hyperparameter_intervention_magnitude",
            "reference_configuration":
                "donor_latest_real_applied_configuration",
            "proposal_configuration":
                "nominal_pb2_proposal_over_batch_lambda_clip_lr",
            "normalization_domain":
                "outer_hyperparameter_model_bounds_only",
            "aggregation":
                "mean_absolute_normalized_coordinate_change",
            "formula":
                "mean_j(abs(h_pb2_norm_j-h_current_norm_j))",
            "signal_range": [0.0, 1.0],
            "uses_learner_state_directly": False,
            "uses_gp_uncertainty": False,
            "entropy_is_search_coordinate": False,
            "applied_entropy_is_gp_execution_feature": True,
        },
        "o2i_entropy_scale": O2I_ENTROPY_SCALE,
        "max_entropy_increment": (
            O2I_MAX_INCREMENT
            if VARIANT in {"full", "o2i"}
            else 0.0
        ),
        "entropy_guard": ENTROPY_GUARD,
        "effective_max_applied_entropy": (
            BASE_ENTROPY_COEFF + O2I_MAX_INCREMENT
            if VARIANT in {"full", "o2i"}
            else BASE_ENTROPY_COEFF
        ),
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
            # Keep these initial distributions identical across compared methods.
            train_batch_size=tune.randint(1000, 60001),
            lr=tune.loguniform(1e-5, 1e-3),
            lambda_=tune.uniform(0.90, 0.99),
            clip_param=tune.uniform(0.10, 0.50),
            # Entropy is deliberately excluded from HPO and is controlled only
            # by CODA's O2I intervention signal.
            entropy_coeff=BASE_ENTROPY_COEFF,
            # Keep the value-function loss weight fixed; it is not part of the
            # PB2-matched outer-loop search space.
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
        "custom_metrics/stability_index",
        "custom_metrics/stability_raw",
        "custom_metrics/stability_valid",
        "custom_metrics/actor_health",
        "custom_metrics/critic_health",
        "custom_metrics/policy_kl",
        "custom_metrics/vf_explained_var",
        "custom_metrics/policy_entropy",
        "custom_metrics/coda_o2i_intervention_magnitude",
        "custom_metrics/coda_o2i_delta_train_batch_size",
        "custom_metrics/coda_o2i_delta_lambda",
        "custom_metrics/coda_o2i_delta_clip_param",
        "custom_metrics/coda_o2i_delta_lr",
        "custom_metrics/coda_entropy_increment",
        "custom_metrics/coda_guided_update",
        "custom_metrics/coda_base_entropy_coeff",
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
                "episode_return_mean", np.nan
            )

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

    context_bounds = {
        "T_before": (0.0, float(TIMESTEPS_MAX)),
        "S_before": (0.0, 1.0),
    }

    scheduler = CODAScheduler(
        time_attr=TIME_ATTR,
        # metric/mode are supplied by TuneConfig below.
        perturbation_interval=int(
            w_pb2_params.get("perturbation_interval", 50_000)
        ),
        quantile_fraction=float(
            w_pb2_params.get("quantile_fraction", 0.25)
        ),
        hyperparam_bounds=HYPERPARAM_BOUNDS,
        context_bounds=context_bounds,
        variant=VARIANT,
        min_valid_transitions=2,
        max_gp_points=1000,
        base_entropy_coeff=BASE_ENTROPY_COEFF,
        o2i_entropy_scale=O2I_ENTROPY_SCALE,
        max_entropy_increment=O2I_MAX_INCREMENT,
        entropy_guard=ENTROPY_GUARD,
        reward_z_clip=4.0,
        synch=False,
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
            # Use the legacy dict so RLlib exposes GAE lambda under the
            # Tune-compatible key `lambda` (PPOConfig stores it internally as
            # `lambda_`).
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
        scheduler.data.to_csv(
            scheduler_path,
            index=False,
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
            (HOPPER_TEST_ENV, CONFIG_EXPERIMENTOS[HOPPER_TEST_ENV])
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
