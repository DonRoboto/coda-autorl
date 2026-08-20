"""Final SAC configuration shared across CODA and external baselines.

Method-specific scheduling/communication logic must remain outside this module.
This file defines only the common SAC learner, raw HPO domain, budget, and the
learner-specific constants required by the CODA SAC instantiation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Population / budget
# ---------------------------------------------------------------------------

POPULATION_SIZE = 4
MAX_TIMESTEPS_PER_WORKER = 1_000_000
PERTURBATION_INTERVAL = 50_000
QUANTILE_FRACTION = 0.25

# Nominal aggregate interaction budget for population-based methods.
NOMINAL_AGGREGATE_BUDGET = (
    POPULATION_SIZE * MAX_TIMESTEPS_PER_WORKER
)

# Matched ASHA protocol:
# 8*0.25M + 4*(0.50M-0.25M) + 2*(1.00M-0.50M) = 4M.
ASHA_NUM_SAMPLES = 8
ASHA_GRACE_PERIOD = 250_000
ASHA_REDUCTION_FACTOR = 2
ASHA_BRACKETS = 1

# ---------------------------------------------------------------------------
# Four-dimensional SAC outer-loop search space
# ---------------------------------------------------------------------------

HYPERPARAM_BOUNDS: Dict[str, Any] = {
    "train_batch_size": [256, 2048],
    "tau": [1e-3, 2e-2],
    "optimization": {
        "actor_learning_rate": [1e-5, 3e-4],
        "critic_learning_rate": [1e-4, 1e-3],
    },
}

# These three positive-valued coordinates are represented in log10 space by
# CODA's GP. External PB2 remains a native Ray PB2 baseline in raw space.
CODA_LOG10_COORDINATES = (
    "tau",
    "optimization/actor_learning_rate",
    "optimization/critic_learning_rate",
)

# ---------------------------------------------------------------------------
# Fixed SAC learner settings
# ---------------------------------------------------------------------------

FIXED_GAMMA = 0.99
FIXED_ENTROPY_LR = 3e-4
INITIAL_ALPHA = 1.0
N_STEP = 1
TARGET_NETWORK_UPDATE_FREQ = 0
LEARNING_STARTS = 1_500
TRAINING_INTENSITY = 4
TWIN_Q = True

REPLAY_BUFFER_CAPACITY = 250_000
REPLAY_BUFFER_CONFIG: Dict[str, Any] = {
    "type": "MultiAgentPrioritizedReplayBuffer",
    "capacity": REPLAY_BUFFER_CAPACITY,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "replay_sequence_length": 1,
}

STORE_BUFFER_IN_CHECKPOINTS = True

MODEL_HIDDENS = [256, 256]
MODEL_ACTIVATION = "relu"

NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_RUNNER = 8
ROLLOUT_FRAGMENT_LENGTH = 8
MIN_SAMPLE_TIMESTEPS_PER_ITERATION = 4_000
OBSERVATION_FILTER = "MeanStdFilter"
NUM_GPUS_PER_TRIAL = 0.2

# ---------------------------------------------------------------------------
# CODA-SAC I2O: TD-error learner state
# ---------------------------------------------------------------------------

TD_REFERENCE_EMA_BETA = 0.90
POLICY_STATE_EMA_BETA = 0.75
POLICY_STATE_MIN = 1e-6
TD_REFERENCE_EPS = 1e-8

# ---------------------------------------------------------------------------
# CODA-SAC O2I: uncertainty -> target entropy
# ---------------------------------------------------------------------------

TARGET_ENTROPY_MAX_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Shared CODA surrogate controls
# ---------------------------------------------------------------------------

MIN_VALID_TRANSITIONS = 16
MAX_GP_POINTS = 1000
REWARD_Z_CLIP = 4.0

# ---------------------------------------------------------------------------
# Champion-selection protocol
# ---------------------------------------------------------------------------

TERMINAL_WINDOW_STEPS = 200_000
MIN_TERMINAL_SUPPORT_STEPS = 100_000
MIN_POST_CONFIG_SUPPORT_STEPS = 100_000
MIN_TERMINAL_POINTS = 3


def search_space_dict() -> Dict[str, Any]:
    """Return a deep copy of the raw SAC search-domain bounds."""
    return deepcopy(HYPERPARAM_BOUNDS)


def tune_search_space() -> Dict[str, Any]:
    """Return Ray Tune initial sampling distributions for SAC."""
    from ray import tune

    return {
        "train_batch_size": tune.randint(256, 2049),
        "tau": tune.loguniform(1e-3, 2e-2),
        "optimization": {
            "actor_learning_rate": tune.loguniform(1e-5, 3e-4),
            "critic_learning_rate": tune.loguniform(1e-4, 1e-3),
        },
    }


def baseline_target_entropy(action_dim: int) -> float:
    """Standard baseline used by PBT/PB2/ASHA and CODA before O2I actuation."""
    if int(action_dim) <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    return -float(action_dim)


def target_entropy_bounds(action_dim: int) -> tuple[float, float]:
    """Return CODA-SAC O2I actuator bounds.

    H_target = -d_a + 0.1*d_a*U_delta, U_delta in [0, 1].
    """
    base = baseline_target_entropy(action_dim)
    upper = base + TARGET_ENTROPY_MAX_FRACTION * float(action_dim)
    return base, upper


def target_entropy_from_uncertainty(
    action_dim: int,
    uncertainty: float,
) -> float:
    """Map donor-relative uncertainty U_delta in [0,1] to target entropy."""
    import numpy as np

    u = float(np.clip(float(uncertainty), 0.0, 1.0))
    low, high = target_entropy_bounds(action_dim)
    value = low + (high - low) * u
    return float(np.clip(value, low, high))


def build_sac_config(
    env_name: str,
    seed: int,
    *,
    target_entropy: float,
    callbacks: Optional[type] = None,
):
    """Build the common RLlib SACConfig used by every SAC-based method."""
    from ray.rllib.algorithms.sac import SACConfig

    cfg = (
        SACConfig()
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
            rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
            observation_filter=OBSERVATION_FILTER,
        )
        .reporting(
            min_sample_timesteps_per_iteration=(
                MIN_SAMPLE_TIMESTEPS_PER_ITERATION
            ),
        )
        .training(
            twin_q=TWIN_Q,
            train_batch_size=256,  # placeholder; Tune/outer scheduler replaces it
            tau=5e-3,              # placeholder
            initial_alpha=INITIAL_ALPHA,
            target_entropy=float(target_entropy),
            optimization_config={
                "actor_learning_rate": 3e-5,   # placeholder
                "critic_learning_rate": 3e-4,  # placeholder
                "entropy_learning_rate": FIXED_ENTROPY_LR,
            },
            n_step=N_STEP,
            gamma=FIXED_GAMMA,
            target_network_update_freq=TARGET_NETWORK_UPDATE_FREQ,
            num_steps_sampled_before_learning_starts=LEARNING_STARTS,
            training_intensity=TRAINING_INTENSITY,
            store_buffer_in_checkpoints=STORE_BUFFER_IN_CHECKPOINTS,
            replay_buffer_config=deepcopy(REPLAY_BUFFER_CONFIG),
            q_model_config={
                "fcnet_hiddens": list(MODEL_HIDDENS),
                "fcnet_activation": MODEL_ACTIVATION,
            },
            policy_model_config={
                "fcnet_hiddens": list(MODEL_HIDDENS),
                "fcnet_activation": MODEL_ACTIVATION,
            },
        )
    )

    if callbacks is not None:
        cfg = cfg.callbacks(callbacks)

    return cfg


def build_tunable_sac_config(
    env_name: str,
    seed: int,
    *,
    target_entropy: float,
    callbacks: Optional[type] = None,
):
    """Build SACConfig with the common Tune initialization distributions."""
    cfg = build_sac_config(
        env_name,
        seed,
        target_entropy=target_entropy,
        callbacks=callbacks,
    )

    space = tune_search_space()

    return cfg.training(
        train_batch_size=space["train_batch_size"],
        tau=space["tau"],
        optimization_config={
            "actor_learning_rate": space["optimization"]["actor_learning_rate"],
            "critic_learning_rate": space["optimization"]["critic_learning_rate"],
            "entropy_learning_rate": FIXED_ENTROPY_LR,
        },
    )
