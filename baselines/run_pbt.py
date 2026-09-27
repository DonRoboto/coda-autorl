

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
from ray.tune.schedulers import PopulationBasedTraining
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
    POBLACION_B,
)

ALGO_NAME = "PBT"

# PB2-matched PBT baseline aligned with the current CODA PPO protocol.
# Set PBT_RUN_TAG="" for canonical final-campaign filenames.
RUN_TAG = os.environ.get("PBT_RUN_TAG", "PB2SPACE").strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

# Match the PPO hyperparameter space used in the PB2 experiments.
HYPERPARAM_BOUNDS = {
    "train_batch_size": [1000, 60000],
    "lambda": [0.90, 0.99],
    "clip_param": [0.10, 0.50],
    "lr": [1e-5, 1e-3],
}

# These remain fixed and identical across the external baselines; CODA uses
# the same vf_loss_coeff while entropy is reserved for its O2I actuator.
FIXED_VF_LOSS_COEFF = 0.5
FIXED_ENTROPY_COEFF = 0.0

# Standard PBT exploration defaults. These reproduce Ray Tune's default
# multiplicative PBT behavior unless overridden from w_pb2_params.
DEFAULT_RESAMPLE_PROBABILITY = 0.25
DEFAULT_PERTURBATION_FACTORS = (1.2, 0.8)


# -----------------------------------------------------------------------------
# Optional Hopper smoke-test controls
# -----------------------------------------------------------------------------
HOPPER_SMOKE_TEST = False
HOPPER_TEST_ENV = "Hopper-v5"
HOPPER_TEST_SEEDS = [1042]


# -----------------------------------------------------------------------------
# Reproducibility / diagnostics callback
# -----------------------------------------------------------------------------
class PBTDiagnosticsCallback(DefaultCallbacks):
    """Record PPO diagnostics without feeding them back into PBT."""

    def __init__(self):
        super().__init__()
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
    def _algorithm_config_value(algorithm, key: str, default=np.nan):
        """Read a scalar from AlgorithmConfig or its legacy dict form."""
        cfg = algorithm.config
        attr_name = "lambda_" if key == "lambda" else key

        try:
            value = getattr(cfg, attr_name)
            if value is not None:
                return value
        except (AttributeError, TypeError):
            pass

        if isinstance(cfg, dict):
            if key in cfg:
                return cfg[key]
            if attr_name in cfg:
                return cfg[attr_name]

        try:
            return cfg[key]
        except Exception:
            try:
                return cfg[attr_name]
            except Exception:
                return default

    @staticmethod
    def _get_default_policy(algorithm):
        try:
            return algorithm.get_policy("default_policy")
        except Exception:
            try:
                return algorithm.get_policy()
            except Exception:
                return None

    def _sync_effective_hyperparams(self, algorithm) -> None:
        """Re-apply current Tune/PBT values after checkpoint restoration.

        On RLlib's legacy PPO stack, restoring a donor checkpoint can restore
        donor-side policy configuration and optimizer state after PBT has
        already installed the receiver's mutated configuration.  For a fair
        PBT-vs-CODA comparison, AlgorithmConfig remains the source of truth and
        the scalar PPO values susceptible to restore are re-applied here.
        """
        policy = self._get_default_policy(algorithm)
        if policy is None:
            return

        desired_lr = self._safe_float(
            self._algorithm_config_value(algorithm, "lr", np.nan), np.nan
        )
        desired_clip = self._safe_float(
            self._algorithm_config_value(algorithm, "clip_param", np.nan), np.nan
        )
        desired_lambda = self._safe_float(
            self._algorithm_config_value(algorithm, "lambda", np.nan), np.nan
        )
        desired_entropy = self._safe_float(
            self._algorithm_config_value(algorithm, "entropy_coeff", np.nan), np.nan
        )

        policy_config = getattr(policy, "config", None)
        if isinstance(policy_config, dict):
            if np.isfinite(desired_lr):
                policy_config["lr"] = float(desired_lr)
            if np.isfinite(desired_clip):
                policy_config["clip_param"] = float(desired_clip)
            if np.isfinite(desired_lambda):
                policy_config["lambda"] = float(desired_lambda)
            if np.isfinite(desired_entropy):
                policy_config["entropy_coeff"] = float(desired_entropy)

        # PPO has one main optimizer in this configuration.  Update only that
        # optimizer rather than overwriting arbitrary policy optimizers.
        if np.isfinite(desired_lr):
            if hasattr(policy, "cur_lr"):
                try:
                    policy.cur_lr = float(desired_lr)
                except Exception:
                    pass

            optimizers = getattr(policy, "_optimizers", None) or []
            if optimizers:
                try:
                    for param_group in optimizers[0].param_groups:
                        param_group["lr"] = float(desired_lr)
                except Exception:
                    pass

        if np.isfinite(desired_entropy) and hasattr(policy, "entropy_coeff"):
            try:
                policy.entropy_coeff = float(desired_entropy)
            except Exception:
                pass

    def _audit_effective_hyperparams(self, algorithm, custom: dict) -> None:
        """Record desired and effective PPO values after checkpoint restore."""
        policy = self._get_default_policy(algorithm)
        if policy is None:
            return

        policy_config = getattr(policy, "config", {}) or {}
        if not isinstance(policy_config, dict):
            policy_config = {}

        desired = {
            "lr": self._safe_float(
                self._algorithm_config_value(algorithm, "lr", np.nan), np.nan
            ),
            "clip_param": self._safe_float(
                self._algorithm_config_value(algorithm, "clip_param", np.nan), np.nan
            ),
            "lambda": self._safe_float(
                self._algorithm_config_value(algorithm, "lambda", np.nan), np.nan
            ),
            "entropy_coeff": self._safe_float(
                self._algorithm_config_value(algorithm, "entropy_coeff", np.nan), np.nan
            ),
        }

        optimizers = getattr(policy, "_optimizers", None) or []
        effective_lr = np.nan
        if optimizers:
            try:
                effective_lr = self._safe_float(
                    optimizers[0].param_groups[0].get("lr", np.nan), np.nan
                )
            except Exception:
                pass

        effective = {
            "lr": effective_lr,
            "clip_param": self._safe_float(
                policy_config.get("clip_param", np.nan), np.nan
            ),
            "lambda": self._safe_float(
                policy_config.get("lambda", np.nan), np.nan
            ),
            "entropy_coeff": self._safe_float(
                getattr(
                    policy,
                    "entropy_coeff",
                    policy_config.get("entropy_coeff", np.nan),
                ),
                np.nan,
            ),
        }

        mismatch_count = 0
        for name in ("lr", "clip_param", "lambda", "entropy_coeff"):
            custom[f"pbt_desired_{name}"] = desired[name]
            custom[f"pbt_effective_{name}"] = effective[name]
            mismatch = float(
                np.isfinite(desired[name])
                and np.isfinite(effective[name])
                and not np.isclose(
                    desired[name], effective[name], rtol=1e-7, atol=1e-12
                )
            )
            custom[f"pbt_mismatch_{name}"] = mismatch
            mismatch_count += int(mismatch)

        custom["pbt_effective_hp_mismatch_count"] = float(mismatch_count)

    def on_checkpoint_loaded(self, *, algorithm, **kwargs):
        # Restore donor state first, then enforce the PBT-mutated configuration.
        self._sync_effective_hyperparams(algorithm)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Defensive synchronization also covers Ray versions where callback
        # ordering around checkpoint restoration differs.
        self._sync_effective_hyperparams(algorithm)

        custom = result.setdefault("custom_metrics", {})
        self._audit_effective_hyperparams(algorithm, custom)
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

        self._iter_count += 1
        if self._iter_count % 5 == 0:
            try:
                cfg = algorithm.config
                logger.info(
                    "PBT PPO diagnostics | KL=%.5f entropy=%.4f "
                    "batch=%s lambda=%.4f clip=%.4f lr=%.6g",
                    policy_kl,
                    policy_entropy,
                    getattr(cfg, "train_batch_size", np.nan),
                    float(getattr(cfg, "lambda_", np.nan)),
                    float(getattr(cfg, "clip_param", np.nan)),
                    float(getattr(cfg, "lr", np.nan)),
                )
            except Exception:
                pass


# -----------------------------------------------------------------------------
# PBT exploration helper
# -----------------------------------------------------------------------------
def _bounded_pbt_explore(config: dict) -> dict:
    """Keep PBT mutations inside the exact PB2-matched search domain.

    Ray PBT normally multiplies continuous values by 0.8 or 1.2 when it does
    not resample. That can move a value outside the declared sampling domain.
    We retain the standard PBT mutation rule but clip the resulting values so
    every method operates over the same admissible hyperparameter bounds.
    """

    batch_lo, batch_hi = HYPERPARAM_BOUNDS["train_batch_size"]
    lambda_lo, lambda_hi = HYPERPARAM_BOUNDS["lambda"]
    clip_lo, clip_hi = HYPERPARAM_BOUNDS["clip_param"]
    lr_lo, lr_hi = HYPERPARAM_BOUNDS["lr"]

    config["train_batch_size"] = int(
        np.clip(
            int(round(float(config["train_batch_size"]))),
            batch_lo,
            batch_hi,
        )
    )
    config["lambda"] = float(
        np.clip(float(config["lambda"]), lambda_lo, lambda_hi)
    )
    config["clip_param"] = float(
        np.clip(float(config["clip_param"]), clip_lo, clip_hi)
    )
    config["lr"] = float(
        np.clip(float(config["lr"]), lr_lo, lr_hi)
    )

    # These are fixed experimental controls and must never be mutated by PBT.
    config["entropy_coeff"] = FIXED_ENTROPY_COEFF
    config["vf_loss_coeff"] = FIXED_VF_LOSS_COEFF

    return config


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
    perturbation_interval = int(
        w_pb2_params.get("perturbation_interval", 50_000)
    )
    quantile_fraction = float(
        w_pb2_params.get("quantile_fraction", 0.25)
    )
    resample_probability = float(
        w_pb2_params.get(
            "resample_probability", DEFAULT_RESAMPLE_PROBABILITY
        )
    )
    perturbation_factors = tuple(
        w_pb2_params.get(
            "perturbation_factors", DEFAULT_PERTURBATION_FACTORS
        )
    )

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
        "fixed_ppo": {
            "gamma": 0.99,
            "minibatch_size": 512,
            "num_sgd_iter": 10,
            "grad_clip": 0.5,
            "vf_clip_param": 10.0,
            "kl_coeff": 0.2,
            "kl_target": 0.01,
            "network": [512, 512],
            "activation": "tanh",
            "vf_share_layers": False,
            "env_runners": 4,
            "num_envs_per_runner": 8,
            "observation_filter": "MeanStdFilter",
        },
        "restore_hyperparameter_sync": {
            "enabled": True,
            "source_of_truth": "AlgorithmConfig/Tune PBT configuration",
            "synchronized": [
                "lr", "clip_param", "lambda", "entropy_coeff"
            ],
            "audit_prefix": "pbt_",
        },
        "pbt": {
            "perturbation_interval": perturbation_interval,
            "quantile_fraction": quantile_fraction,
            "resample_probability": resample_probability,
            "perturbation_factors": list(perturbation_factors),
            "synch": False,
            "bounded_custom_explore": True,
            "mutation_rule": (
                "standard Ray PBT resampling or multiplicative perturbation, "
                "followed by clipping to the PB2-matched domain"
            ),
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
        .callbacks(PBTDiagnosticsCallback)
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
            # Initial distributions are exactly the same as CODA/PB2SPACE.
            train_batch_size=tune.randint(1000, 60001),
            lr=tune.loguniform(1e-5, 1e-3),
            lambda_=tune.uniform(0.90, 0.99),
            clip_param=tune.uniform(0.10, 0.50),

            # Fixed controls: entropy is not optimized in the baseline.
            entropy_coeff=FIXED_ENTROPY_COEFF,
            vf_loss_coeff=FIXED_VF_LOSS_COEFF,

            minibatch_size=512,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
            gamma=0.99,
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
        "custom_metrics/pbt_desired_lr",
        "custom_metrics/pbt_effective_lr",
        "custom_metrics/pbt_mismatch_lr",
        "custom_metrics/pbt_desired_clip_param",
        "custom_metrics/pbt_effective_clip_param",
        "custom_metrics/pbt_mismatch_clip_param",
        "custom_metrics/pbt_desired_lambda",
        "custom_metrics/pbt_effective_lambda",
        "custom_metrics/pbt_mismatch_lambda",
        "custom_metrics/pbt_desired_entropy_coeff",
        "custom_metrics/pbt_effective_entropy_coeff",
        "custom_metrics/pbt_mismatch_entropy_coeff",
        "custom_metrics/pbt_effective_hp_mismatch_count",
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
        # Preserve the causal row order emitted by Tune.  After checkpoint
        # inheritance, training_iteration/timesteps_total may move backward, so
        # sorting by those counters can interleave different lineages.
        out["causal_order"] = np.arange(len(out), dtype=int)
        out["entorno"] = env_name
        out["semilla"] = seed
        out["agente_id"] = f"Agente_{idx + 1}"
        frames.append(out)

    if frames:
        final = pd.concat(frames, ignore_index=True)
        final = final.sort_values(
            ["agente_id", "causal_order"],
            kind="mergesort",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_path, index=False)


def _extract_pbt_history(
    result_grid,
    env_name: str,
    seed: int,
    output_path: Path,
) -> None:
    """Save the observed PBT hyperparameter schedule for downstream analysis."""

    frames = []
    wanted = [
        "training_iteration",
        "timesteps_total",
        METRIC,
        "config/train_batch_size",
        "config/lambda",
        "config/clip_param",
        "config/lr",
        "config/entropy_coeff",
        "config/vf_loss_coeff",
    ]

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe
        if df is None or df.empty:
            continue

        present = [c for c in wanted if c in df.columns]
        out = df[present].copy()
        out["causal_order"] = np.arange(len(out), dtype=int)
        out["entorno"] = env_name
        out["semilla"] = seed
        out["agente_id"] = f"Agente_{idx + 1}"
        frames.append(out)

    if not frames:
        return

    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(
        ["agente_id", "causal_order"],
        kind="mergesort",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_path, index=False)


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

    perturbation_interval = int(
        w_pb2_params.get("perturbation_interval", 50_000)
    )
    quantile_fraction = float(
        w_pb2_params.get("quantile_fraction", 0.25)
    )
    resample_probability = float(
        w_pb2_params.get(
            "resample_probability", DEFAULT_RESAMPLE_PROBABILITY
        )
    )
    perturbation_factors = tuple(
        w_pb2_params.get(
            "perturbation_factors", DEFAULT_PERTURBATION_FACTORS
        )
    )

    # Same four-dimensional domain as CODA/PB2SPACE.
    hyperparam_mutations = {
        "train_batch_size": tune.randint(1000, 60001),
        "lambda": tune.uniform(0.90, 0.99),
        "clip_param": tune.uniform(0.10, 0.50),
        "lr": tune.loguniform(1e-5, 1e-3),
    }

    scheduler = PopulationBasedTraining(
        time_attr=TIME_ATTR,
        #metric=METRIC,
        #mode="max",
        perturbation_interval=perturbation_interval,
        hyperparam_mutations=hyperparam_mutations,
        quantile_fraction=quantile_fraction,
        resample_probability=resample_probability,
        perturbation_factors=perturbation_factors,
        custom_explore_fn=_bounded_pbt_explore,
        log_config=True,
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
            # Tune-compatible key `lambda`.
            param_space=ppo_config.to_dict(),
            run_config=tune.RunConfig(
                name=f"{OUTPUT_NAME}_{env_name}_Seed{seed}",
                verbose=0,
                storage_path=str(storage_root),
                stop={TIME_ATTR: TIMESTEPS_MAX},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_at_end=True,
                ),
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

        # This gives a CSV with the configuration observed at each training
        # iteration. Ray also writes the native pbt_global.txt mutation log
        # because log_config=True.
        scheduler_path = (
            Path("./results/scheduler")
            / env_name
            / f"scheduler_{OUTPUT_NAME}_seed{seed}.csv"
        )
        _extract_pbt_history(
            results,
            env_name,
            seed,
            scheduler_path,
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
