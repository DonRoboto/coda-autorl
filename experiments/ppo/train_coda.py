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
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.backends.cudnn as cudnn


warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


# =============================================================================
# REPOSITORY / SHARED CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from configs.environments import ENVIRONMENTS  # noqa: E402
from configs.seeds import TRAINING_SEEDS, SMOKE_TRAINING_SEED  # noqa: E402
from configs.ppo_config import (  # noqa: E402
    POPULATION_SIZE,
    MAX_TIMESTEPS_PER_WORKER,
    PERTURBATION_INTERVAL,
    QUANTILE_FRACTION,
    HYPERPARAM_BOUNDS,
    CODA_LOG10_COORDINATES,
    FIXED_GAMMA,
    FIXED_VF_LOSS_COEFF,
    BASE_ENTROPY_COEFF,
    POLICY_KL_REFERENCE,
    POLICY_STATE_MIN,
    POLICY_STATE_EMA_BETA,
    O2I_UNCERTAINTY_SCALE,
    O2I_MAX_INCREMENT,
    ENTROPY_GUARD,
    MIN_VALID_TRANSITIONS,
    MAX_GP_POINTS,
    REWARD_Z_CLIP,
    NUM_GPUS_PER_TRIAL,
    build_tunable_ppo_config,
)
from coda.schedulers.coda_ppo_scheduler import CODAPPOOptimizer  # noqa: E402

# Backward-compatible local alias.
CODAScheduler = CODAPPOOptimizer

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)

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

RUN_TAG = os.environ.get(
    "CODA_RUN_TAG",
    "KL_DUQ_DONORREL_PB2SPACE",
).strip()
OUTPUT_NAME = f"{ALGO_NAME}_{RUN_TAG}" if RUN_TAG else ALGO_NAME

METRIC = "env_runners/episode_return_mean"
TIME_ATTR = "timesteps_total"

POBLACION_B = POPULATION_SIZE
TIMESTEPS_MAX = MAX_TIMESTEPS_PER_WORKER

RESULTS_ROOT = REPO_ROOT / "results"

ENVIRONMENT_CONFIG = {
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

CODA_PPO_SMOKE_TEST = os.environ.get("CODA_PPO_SMOKE", "1") != "0"
CODA_PPO_SMOKE_ENV = os.environ.get("CODA_PPO_SMOKE_ENV", "Hopper-v5")
CODA_PPO_SMOKE_SEEDS = [
    int(os.environ.get("CODA_PPO_SMOKE_SEED", str(SMOKE_TRAINING_SEED)))
]
CODA_PPO_SMOKE_STEPS = int(
    os.environ.get("CODA_PPO_SMOKE_STEPS", "500000")
)


# -----------------------------------------------------------------------------
# Callback: restore synchronization, KL-only I2O, and diagnostics
# -----------------------------------------------------------------------------# -----------------------------------------------------------------------------
# Callback: restore synchronization, KL-only I2O, and diagnostics
# -----------------------------------------------------------------------------
class CODAKLUQCallback(DefaultCallbacks):
    """Maintain KL-only I2O state and donor-reference O2I audits."""

    def __init__(self):
        super().__init__()
        self._policy_state_ema: Optional[float] = None
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

    @staticmethod
    def _algorithm_config_value(algorithm, key: str, default=np.nan):
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
        """Re-apply Tune values after a donor checkpoint restore."""
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
            self._algorithm_config_value(algorithm, "entropy_coeff", np.nan),
            np.nan,
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
                self._algorithm_config_value(algorithm, "clip_param", np.nan),
                np.nan,
            ),
            "lambda": self._safe_float(
                self._algorithm_config_value(algorithm, "lambda", np.nan), np.nan
            ),
            "entropy_coeff": self._safe_float(
                self._algorithm_config_value(
                    algorithm, "entropy_coeff", np.nan
                ),
                np.nan,
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
            custom[f"coda_desired_{name}"] = desired[name]
            custom[f"coda_effective_{name}"] = effective[name]
            mismatch = float(
                np.isfinite(desired[name])
                and np.isfinite(effective[name])
                and not np.isclose(
                    desired[name], effective[name], rtol=1e-7, atol=1e-12
                )
            )
            custom[f"coda_mismatch_{name}"] = mismatch
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

        seed = self._safe_float(
            bridge.get("lineage_ema_seed", np.nan), np.nan
        )
        self._policy_state_ema = float(seed) if np.isfinite(seed) else None
        self._last_lineage_generation = generation

    def on_checkpoint_loaded(self, *, algorithm, **kwargs):
        self._sync_effective_hyperparams(algorithm)
        self._apply_lineage_seed_if_needed(algorithm)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        self._sync_effective_hyperparams(algorithm)
        self._apply_lineage_seed_if_needed(algorithm)

        custom = result.setdefault("custom_metrics", {})
        self._audit_effective_hyperparams(algorithm, custom)
        learner_stats = self._extract_learner_stats(result)
        bridge = self._bridge_from_algorithm(algorithm)

        # Scheduler/O2I audit metadata.
        custom["coda_guided_update"] = float(
            bool(bridge.get("guided_update", False))
        )
        custom["coda_gp_data_count"] = self._safe_float(
            bridge.get("gp_data_count", 0.0), 0.0
        )
        # Backward-compatible aliases.
        custom["coda_o2i_uncertainty_raw_std"] = self._safe_float(
            bridge.get("o2i_uncertainty_raw_std", 0.0), 0.0
        )
        custom["coda_o2i_uncertainty_prior_std"] = self._safe_float(
            bridge.get("o2i_uncertainty_prior_std", 0.0), 0.0
        )
        custom["coda_o2i_uncertainty_normalized"] = self._safe_float(
            bridge.get("o2i_uncertainty_normalized", 0.0), 0.0
        )
        custom["coda_o2i_kernel_variance"] = self._safe_float(
            bridge.get("o2i_kernel_variance", 0.0), 0.0
        )

        # Explicit donor-relative incremental uncertainty diagnostics.
        custom["coda_o2i_candidate_posterior_std"] = self._safe_float(
            bridge.get(
                "o2i_candidate_posterior_std",
                bridge.get("o2i_uncertainty_raw_std", 0.0),
            ),
            0.0,
        )
        custom["coda_o2i_donor_posterior_std"] = self._safe_float(
            bridge.get(
                "o2i_donor_posterior_std",
                bridge.get("o2i_donor_uncertainty_raw_std", 0.0),
            ),
            0.0,
        )
        custom["coda_o2i_candidate_prior_std"] = self._safe_float(
            bridge.get(
                "o2i_candidate_prior_std",
                bridge.get("o2i_uncertainty_prior_std", 0.0),
            ),
            0.0,
        )
        custom["coda_o2i_donor_prior_std"] = self._safe_float(
            bridge.get(
                "o2i_donor_prior_std",
                bridge.get("o2i_donor_uncertainty_prior_std", 0.0),
            ),
            0.0,
        )
        custom["coda_o2i_incremental_uncertainty"] = self._safe_float(
            bridge.get(
                "o2i_incremental_uncertainty",
                bridge.get("o2i_uncertainty_normalized", 0.0),
            ),
            0.0,
        )
        custom["coda_o2i_donor_kernel_variance"] = self._safe_float(
            bridge.get("o2i_donor_kernel_variance", 0.0), 0.0
        )
        custom["coda_gp_kernel_variance_parameter"] = self._safe_float(
            bridge.get("gp_kernel_variance_parameter", np.nan), np.nan
        )
        custom["coda_gp_lengthscale"] = self._safe_float(
            bridge.get("gp_lengthscale", np.nan), np.nan
        )
        custom["coda_gp_temporal_epsilon"] = self._safe_float(
            bridge.get("gp_temporal_epsilon", np.nan), np.nan
        )
        custom["coda_gp_alpha_used"] = self._safe_float(
            bridge.get("gp_alpha_used", np.nan), np.nan
        )
        custom["coda_o2i_acquisition_value"] = self._safe_float(
            bridge.get("o2i_acquisition_value", np.nan), np.nan
        )
        custom["coda_o2i_reference_entropy_coeff"] = self._safe_float(
            bridge.get("o2i_reference_entropy_coeff", BASE_ENTROPY_COEFF),
            BASE_ENTROPY_COEFF,
        )
        custom["coda_entropy_increment"] = self._safe_float(
            bridge.get("entropy_increment", 0.0), 0.0
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

        # KL-only I2O state. EV is logged only as an independent diagnostic.
        if learner_stats is None:
            custom["policy_update_state"] = np.nan
            custom["policy_update_state_raw"] = np.nan
            custom["policy_update_state_valid"] = 0.0
            custom["policy_update_health"] = np.nan
            custom["policy_kl"] = np.nan
            custom["vf_explained_var"] = np.nan
            custom["policy_entropy"] = np.nan
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

        valid = bool(np.isfinite(policy_kl))

        if valid:
            policy_update_health = float(
                np.exp(
                    -max(
                        0.0,
                        max(policy_kl, 0.0) / POLICY_KL_REFERENCE - 1.0,
                    )
                )
            )
            policy_state_raw = float(
                np.clip(
                    policy_update_health,
                    POLICY_STATE_MIN,
                    1.0,
                )
            )

            if self._policy_state_ema is None:
                self._policy_state_ema = policy_state_raw
            else:
                self._policy_state_ema = float(
                    POLICY_STATE_EMA_BETA * self._policy_state_ema
                    + (1.0 - POLICY_STATE_EMA_BETA) * policy_state_raw
                )

            policy_state = float(
                np.clip(
                    self._policy_state_ema,
                    POLICY_STATE_MIN,
                    1.0,
                )
            )
        else:
            policy_update_health = np.nan
            policy_state_raw = np.nan
            policy_state = np.nan

        custom["policy_update_state"] = policy_state
        custom["policy_update_state_raw"] = policy_state_raw
        custom["policy_update_state_valid"] = float(valid)
        custom["policy_update_health"] = policy_update_health
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
                    "CODA KL-DUQ donor-relative | KL=%.5f state=%.4f "
                    "U_delta=%.4f ref_entropy=%.5f applied_entropy=%.5f "
                    "sigma_c=%.5g sigma_d=%.5g gp_eps=%.5g mismatch=%d",
                    policy_kl,
                    policy_state,
                    custom["coda_o2i_incremental_uncertainty"],
                    custom["coda_o2i_reference_entropy_coeff"],
                    entropy_coeff,
                    custom["coda_o2i_candidate_posterior_std"],
                    custom["coda_o2i_donor_posterior_std"],
                    custom["coda_gp_temporal_epsilon"],
                    int(custom.get("coda_effective_hp_mismatch_count", 0.0)),
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
        "communication_design": "KL-only I2O + donor-referenced observed-GP-uncertainty O2I",
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
        "learning_rate_model_representation": "log10",
        "fixed_vf_loss_coeff": FIXED_VF_LOSS_COEFF,
        "ppo_gamma": FIXED_GAMMA,
        "perturbation_interval": int(
            w_pb2_params.get("perturbation_interval", 50_000)
        ),
        "quantile_fraction": float(
            w_pb2_params.get("quantile_fraction", 0.25)
        ),
        "min_valid_transitions": MIN_VALID_TRANSITIONS,
        "max_gp_points": MAX_GP_POINTS,
        "reward_context_normalization": {
            "method": "causal robust median/IQR",
            "iqr_scale_factor": 1.349,
            "mad_fallback_factor": 1.4826,
            "z_clip": REWARD_Z_CLIP,
            "uses_future_information": False,
        },
        "i2o_policy_update_state": {
            "source": "PPO approximate policy KL only",
            "reference_kl": POLICY_KL_REFERENCE,
            "raw_mapping": "exp(-max(0, max(KL,0)/reference_kl - 1))",
            "ema_beta": POLICY_STATE_EMA_BETA,
            "lower_bound": POLICY_STATE_MIN,
            "vf_explained_var_used_in_i2o": False,
        },
        "o2i_incremental_gp_uncertainty": {
            "source_model": "TV-GP fitted only to completed real observations",
            "pending_points_used_for_o2i": False,
            "pending_points_used_for_batch_ucb_variance": True,
            "reference_actuator_condition": "same donor context and latest applied entropy for candidate and donor queries",
            "raw_signal": "positive candidate-minus-donor posterior standard-deviation difference",
            "normalization": "max(candidate_posterior_std - donor_posterior_std, 0) / (candidate_prior_std + eps)",
            "normalized_range": [0.0, 1.0],
            "mapping": "base + min(max_increment, scale * incremental_uncertainty)",
            "uncertainty_scale": O2I_UNCERTAINTY_SCALE,
            "max_entropy_increment": O2I_MAX_INCREMENT,
            "entropy_guard": ENTROPY_GUARD,
            "base_entropy_coeff": BASE_ENTROPY_COEFF,
            "reachable_entropy_domain": [
                BASE_ENTROPY_COEFF,
                BASE_ENTROPY_COEFF + O2I_MAX_INCREMENT,
            ],
            "entropy_is_search_coordinate": False,
            "applied_entropy_is_gp_execution_feature": True,
            "recomputed_after_executable_mapping": True,
            "reference_entropy_is_added_to_new_entropy": False,
            "new_entropy_origin": "fixed baseline plus donor-relative incremental-uncertainty increment",
            "gp_fit_diagnostics_logged": [
                "kernel_variance",
                "lengthscale",
                "temporal_epsilon",
                "alpha_used",
            ],
        },
        "ppo": {
            "gamma": FIXED_GAMMA,
            "grad_clip": 0.5,
            "minibatch_size": 512,
            "num_sgd_iter": 10,
            "vf_clip_param": 10.0,
            "use_kl_loss": True,
            "kl_coeff": 0.2,
            "kl_target": POLICY_KL_REFERENCE,
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
            "num_env_runners": 4,
            "num_envs_per_env_runner": 8,
            "observation_filter": "MeanStdFilter",
            "num_gpus_per_trial": 0.2,
        },
        "checkpointing": {
            "checkpoint_at_end": True,
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


# Common PPO construction is provided by configs.ppo_config.

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
        "custom_metrics/policy_update_state",
        "custom_metrics/policy_update_state_raw",
        "custom_metrics/policy_update_state_valid",
        "custom_metrics/policy_update_health",
        "custom_metrics/policy_kl",
        "custom_metrics/vf_explained_var",
        "custom_metrics/policy_entropy",
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
        "custom_metrics/coda_o2i_reference_entropy_coeff",
        "custom_metrics/coda_entropy_increment",
        "custom_metrics/coda_base_entropy_coeff",
        "custom_metrics/coda_nominal_entropy_coeff",
        "custom_metrics/coda_applied_entropy_coeff",
        "custom_metrics/coda_desired_lr",
        "custom_metrics/coda_effective_lr",
        "custom_metrics/coda_mismatch_lr",
        "custom_metrics/coda_desired_clip_param",
        "custom_metrics/coda_effective_clip_param",
        "custom_metrics/coda_mismatch_clip_param",
        "custom_metrics/coda_desired_lambda",
        "custom_metrics/coda_effective_lambda",
        "custom_metrics/coda_mismatch_lambda",
        "custom_metrics/coda_desired_entropy_coeff",
        "custom_metrics/coda_effective_entropy_coeff",
        "custom_metrics/coda_mismatch_entropy_coeff",
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

        present = [c for c in columns_wanted if c in df.columns]
        out = df[present].copy()
        out["causal_order"] = np.arange(len(out), dtype=np.int64)
        out["entorno"] = env_name
        out["semilla"] = seed
        out["agente_id"] = f"Agente_{idx + 1}"
        frames.append(out)

    if frames:
        final = pd.concat(frames, ignore_index=True)
        final = final.sort_values(
            ["agente_id", "causal_order"],
            kind="stable",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_path, index=False)


def _save_trial_status_summary(
    result_grid,
    env_name: str,
    seed: int,
    output_path: Path,
) -> None:
    """Save terminal resource, checkpoint, and error status per trial."""
    rows = []

    for idx, result in enumerate(result_grid):
        df = result.metrics_dataframe
        n_rows = int(len(df)) if df is not None else 0
        final_steps = np.nan
        last_finite_reward = np.nan

        if df is not None and not df.empty:
            if TIME_ATTR in df.columns:
                steps = pd.to_numeric(df[TIME_ATTR], errors="coerce")
                if steps.notna().any():
                    final_steps = float(steps.iloc[-1])

            if METRIC in df.columns:
                rewards = pd.to_numeric(df[METRIC], errors="coerce")
                finite = rewards[np.isfinite(rewards)]
                if not finite.empty:
                    last_finite_reward = float(finite.iloc[-1])

        error = getattr(result, "error", None)
        rows.append(
            {
                "environment": env_name,
                "training_seed": int(seed),
                "agent_id": f"Agente_{idx + 1}",
                "n_metric_rows": n_rows,
                "final_timesteps_total": final_steps,
                "reached_target_steps": int(
                    np.isfinite(final_steps) and final_steps >= TIMESTEPS_MAX
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
    results = list(result_grid)
    problems = []
    if len(results) != POPULATION_SIZE:
        problems.append(
            f"expected {POPULATION_SIZE} trials, found {len(results)}"
        )

    mismatch_col = "custom_metrics/coda_effective_hp_mismatch_count"

    for idx, result in enumerate(results):
        agent = f"Agente_{idx + 1}"
        error = getattr(result, "error", None)
        if error is not None:
            problems.append(f"{agent}: Tune/RLlib error={error!r}")

        df = result.metrics_dataframe
        if df is None or df.empty:
            problems.append(f"{agent}: empty metrics dataframe")
            continue

        values = pd.to_numeric(df.get(TIME_ATTR), errors="coerce")
        finite = values[np.isfinite(values)] if values is not None else pd.Series(dtype=float)
        terminal = float(finite.iloc[-1]) if not finite.empty else np.nan

        if not np.isfinite(terminal) or terminal < target_steps:
            problems.append(
                f"{agent}: terminal {TIME_ATTR}={terminal}, expected >= {target_steps}"
            )

        if not result.checkpoint:
            problems.append(f"{agent}: missing final checkpoint")

        if mismatch_col in df.columns:
            mismatch = pd.to_numeric(df[mismatch_col], errors="coerce")
            finite_mismatch = mismatch[np.isfinite(mismatch)]
            if not finite_mismatch.empty and (finite_mismatch != 0).any():
                problems.append(f"{agent}: non-zero restore-time HP mismatch")

    if problems:
        raise RuntimeError(
            "Invalid CODA-PPO population run:\n  - " + "\n  - ".join(problems)
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
    root.mkdir(parents=True, exist_ok=True)

    best_reward = -np.inf
    best_agent = None

    for idx, result in enumerate(result_grid):
        if not result.checkpoint:
            logger.warning("No final checkpoint for Agente_%s", idx + 1)
            continue

        raw = result.metrics or {}
        reward = np.nan

        if isinstance(raw.get("env_runners"), dict):
            reward = raw["env_runners"].get(
                "episode_return_mean", np.nan
            )

        try:
            reward_is_finite = np.isfinite(float(reward))
        except (TypeError, ValueError):
            reward_is_finite = False

        if not reward_is_finite:
            reward = raw.get("episode_reward_mean", np.nan)

        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = np.nan

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
            f"Best final worker (diagnostic only): {best_agent} | "
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
        perturbation_interval=int(
            w_pb2_params.get("perturbation_interval", 50_000)
        ),
        quantile_fraction=float(
            w_pb2_params.get("quantile_fraction", 0.25)
        ),
        hyperparam_bounds=HYPERPARAM_BOUNDS,
        context_bounds=context_bounds,
        variant=VARIANT,
        min_valid_transitions=MIN_VALID_TRANSITIONS,
        max_gp_points=MAX_GP_POINTS,
        base_entropy_coeff=BASE_ENTROPY_COEFF,
        o2i_uncertainty_scale=O2I_UNCERTAINTY_SCALE,
        max_entropy_increment=O2I_MAX_INCREMENT,
        entropy_guard=ENTROPY_GUARD,
        reward_z_clip=REWARD_Z_CLIP,
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

        ppo_config = build_tunable_ppo_config(
            env_name,
            seed,
            callbacks=CODAKLUQCallback,
        )

        storage_root = (
            RESULTS_ROOT / "ray_tune_logs" / OUTPUT_NAME
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

        scheduler_path = (
            (RESULTS_ROOT / "scheduler")
            / env_name
            / f"scheduler_{OUTPUT_NAME}_seed{seed}.csv"
        )
        scheduler_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler.data.to_csv(scheduler_path, index=False)

        status_path = (
            Path("./results/status")
            / env_name
            / f"status_{OUTPUT_NAME}_seed{seed}.csv"
        )
        _save_trial_status_summary(
            results,
            env_name,
            seed,
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
            w_pb2_params=w_pb2_params,
        )

        _copy_final_checkpoints(
            results,
            env_name,
            seed,
        )

        _validate_result_grid(
            results,
            target_steps=TIMESTEPS_MAX,
        )

        print(
            f"Completed {OUTPUT_NAME}: {env_name}, seed={seed}"
        )
        return True

    except Exception as exc:
        print(
            f"\nCRITICAL ERROR in {env_name}, seed={seed}: {exc}"
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

    if CODA_PPO_SMOKE_TEST:
        if CODA_PPO_SMOKE_ENV not in ENVIRONMENTS:
            raise KeyError(f"Unknown smoke environment: {CODA_PPO_SMOKE_ENV}")
        experiment_items = [
            (
                CODA_PPO_SMOKE_ENV,
                {
                    **ENVIRONMENT_CONFIG[CODA_PPO_SMOKE_ENV],
                    "semillas": CODA_PPO_SMOKE_SEEDS,
                },
            )
        ]
        target_steps = CODA_PPO_SMOKE_STEPS
    else:
        experiment_items = list(ENVIRONMENT_CONFIG.items())
        target_steps = TIMESTEPS_MAX

    total = sum(len(cfg["semillas"]) for _, cfg in experiment_items)
    done = 0

    for env_name, env_cfg in experiment_items:
        for seed in env_cfg["semillas"]:
            original_max = globals()["TIMESTEPS_MAX"]
            try:
                globals()["TIMESTEPS_MAX"] = int(target_steps)
                ok = run_experiment(
                    env_name,
                    int(seed),
                    env_cfg,
                )
            finally:
                globals()["TIMESTEPS_MAX"] = original_max

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
