"""Final PPO configuration shared across CODA and external baselines.

This module centralizes the PPO learner configuration and search domain.
Method-specific behavior (PBT, PB2, ASHA, CODA, CODA-I2O, CODA-O2I)
belongs in the experiment/scheduler code, not here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Population / budget
# ---------------------------------------------------------------------------

POPULATION_SIZE = 4
MAX_TIMESTEPS_PER_WORKER = 2_000_000
PERTURBATION_INTERVAL = 50_000
QUANTILE_FRACTION = 0.25

# ---------------------------------------------------------------------------
# Four-dimensional PPO outer-loop search space
# ---------------------------------------------------------------------------

HYPERPARAM_BOUNDS: Dict[str, Any] = {
    "train_batch_size": [1_000, 60_000],
    "lambda": [0.90, 0.99],
    "clip_param": [0.10, 0.50],
    "lr": [1e-5, 1e-3],
}

# ---------------------------------------------------------------------------
# Fixed PPO learner settings
# ---------------------------------------------------------------------------

FIXED_GAMMA = 0.99
FIXED_VF_LOSS_COEFF = 0.5

# Entropy is not an outer-loop search coordinate. For CODA it is reserved
# for the O2I actuator; external baselines keep it fixed at zero.
BASE_ENTROPY_COEFF = 0.0
MAX_ENTROPY_COEFF = 0.004

MINIBATCH_SIZE = 512
NUM_SGD_ITER = 10
GRAD_CLIP = 0.5
VF_CLIP_PARAM = 10.0
KL_COEFF = 0.2
KL_TARGET = 0.01

MODEL_HIDDENS = [512, 512]
MODEL_ACTIVATION = "tanh"
VF_SHARE_LAYERS = False

NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_RUNNER = 8
OBSERVATION_FILTER = "MeanStdFilter"
NUM_GPUS_PER_TRIAL = 0.2

# ---------------------------------------------------------------------------
# CODA-PPO diagnostic/surrogate controls
# ---------------------------------------------------------------------------

POLICY_STATE_EMA_BETA = 0.75
POLICY_STATE_MIN = 1e-6

MIN_VALID_TRANSITIONS = 16
MAX_GP_POINTS = 1000
REWARD_Z_CLIP = 4.0

# Candidate acquisition / TV-GP settings used by the final CODA implementation.
UCB_BETA_OFFSET = 0.2
UCB_BETA_LOG_SCALE = 0.4
GP_LBFGS_RESTARTS = 10

# ---------------------------------------------------------------------------
# Champion-selection protocol
# ---------------------------------------------------------------------------

TERMINAL_WINDOW_STEPS = 200_000
MIN_TERMINAL_SUPPORT_STEPS = 100_000
MIN_POST_CONFIG_SUPPORT_STEPS = 100_000
MIN_TERMINAL_POINTS = 3


def search_space_dict() -> Dict[str, Any]:
    """Return a deep copy of the raw PPO search-domain bounds."""
    return deepcopy(HYPERPARAM_BOUNDS)


def tune_search_space() -> Dict[str, Any]:
    """Return Ray Tune initial sampling distributions for PPO.

    Ray is imported lazily so this config file remains importable in lightweight
    analysis environments.
    """
    from ray import tune

    return {
        "train_batch_size": tune.randint(1_000, 60_001),
        "lambda": tune.uniform(0.90, 0.99),
        "clip_param": tune.uniform(0.10, 0.50),
        "lr": tune.loguniform(1e-5, 1e-3),
    }


def build_ppo_config(
    env_name: str,
    seed: int,
    *,
    callbacks: Optional[type] = None,
):
    """Build the common RLlib PPOConfig used by every PPO-based method."""
    from ray.rllib.algorithms.ppo import PPOConfig

    cfg = (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(num_gpus=NUM_GPUS_PER_TRIAL)
        .debugging(seed=int(seed))
        .env_runners(
            num_env_runners=NUM_ENV_RUNNERS,
            num_envs_per_env_runner=NUM_ENVS_PER_RUNNER,
            observation_filter=OBSERVATION_FILTER,
        )
        .training(
            train_batch_size=1_000,  # placeholder; Tune/outer scheduler replaces it
            lr=1e-4,                 # placeholder
            lambda_=0.95,            # placeholder
            clip_param=0.2,          # placeholder
            entropy_coeff=BASE_ENTROPY_COEFF,
            vf_loss_coeff=FIXED_VF_LOSS_COEFF,
            minibatch_size=MINIBATCH_SIZE,
            use_kl_loss=True,
            kl_coeff=KL_COEFF,
            kl_target=KL_TARGET,
            gamma=FIXED_GAMMA,
            grad_clip=GRAD_CLIP,
            num_sgd_iter=NUM_SGD_ITER,
            vf_clip_param=VF_CLIP_PARAM,
            model={
                "fcnet_hiddens": list(MODEL_HIDDENS),
                "fcnet_activation": MODEL_ACTIVATION,
                "vf_share_layers": VF_SHARE_LAYERS,
            },
        )
    )

    if callbacks is not None:
        cfg = cfg.callbacks(callbacks)

    return cfg


def build_tunable_ppo_config(
    env_name: str,
    seed: int,
    *,
    callbacks: Optional[type] = None,
):
    """Build PPOConfig with the common Tune initialization distributions."""
    cfg = build_ppo_config(env_name, seed, callbacks=callbacks)
    space = tune_search_space()

    return cfg.training(
        train_batch_size=space["train_batch_size"],
        lr=space["lr"],
        lambda_=space["lambda"],
        clip_param=space["clip_param"],
    )
