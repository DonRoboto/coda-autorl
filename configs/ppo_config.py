"""Final PPO configuration shared across CODA and external baselines.

This module is the single source of truth for the PPO experiments. Method-
specific scheduling behavior (PBT, PB2, ASHA, CODA) remains in the corresponding
experiment/scheduler modules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

# =============================================================================
# POPULATION / BUDGET
# =============================================================================

POPULATION_SIZE = 4
MAX_TIMESTEPS_PER_WORKER = 2_000_000
PERTURBATION_INTERVAL = 50_000
QUANTILE_FRACTION = 0.25

NOMINAL_AGGREGATE_BUDGET = POPULATION_SIZE * MAX_TIMESTEPS_PER_WORKER

# Final ASHA resource-allocation protocol. Because ASHA is asynchronous, the
# realized aggregate number of interactions must still be reported.
ASHA_NUM_SAMPLES = 30
ASHA_GRACE_PERIOD = 50_000
ASHA_REDUCTION_FACTOR = 2
ASHA_BRACKETS = 1

ASHA_BUDGET_EXCELLENT_LOW = 0.95
ASHA_BUDGET_EXCELLENT_HIGH = 1.05
ASHA_BUDGET_ACCEPTABLE_LOW = 0.90
ASHA_BUDGET_ACCEPTABLE_HIGH = 1.10

# =============================================================================
# FOUR-DIMENSIONAL PPO OUTER SEARCH SPACE
# =============================================================================

HYPERPARAM_BOUNDS: Dict[str, Any] = {
    "train_batch_size": [1_000, 60_000],
    "lambda": [0.90, 0.99],
    "clip_param": [0.10, 0.50],
    "lr": [1e-5, 1e-3],
}

# =============================================================================
# FIXED PPO LEARNER SETTINGS
# =============================================================================

FIXED_GAMMA = 0.99
FIXED_VF_LOSS_COEFF = 0.5

# Entropy is excluded from the outer HPO coordinates. For CODA it is the O2I
# actuator; external baselines keep it fixed at zero.
BASE_ENTROPY_COEFF = 0.0

# Gated O2I actuator used by the PB2-core CODA implementation.
O2I_UNCERTAINTY_THRESHOLD = 0.10
O2I_WARMUP_START = 16
O2I_WARMUP_END = 64

# Maximum PPO entropy intervention after gating.
O2I_UNCERTAINTY_SCALE = 0.004
O2I_MAX_INCREMENT = 0.004
ENTROPY_GUARD = 0.004

# Backward-compatible alias used by some auxiliary scripts.
MAX_ENTROPY_COEFF = ENTROPY_GUARD

MINIBATCH_SIZE = 512
NUM_SGD_ITER = 10
GRAD_CLIP = 0.5
VF_CLIP_PARAM = 10.0
USE_KL_LOSS = True
KL_COEFF = 0.2
POLICY_KL_REFERENCE = 0.01
KL_TARGET = POLICY_KL_REFERENCE

MODEL_HIDDENS = [512, 512]
MODEL_ACTIVATION = "tanh"
VF_SHARE_LAYERS = False

NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_RUNNER = 8
OBSERVATION_FILTER = "MeanStdFilter"
NUM_GPUS_PER_TRIAL = 0.2

# =============================================================================
# CODA-PPO I2O SETTINGS
# =============================================================================

POLICY_STATE_EMA_BETA = 0.75
POLICY_STATE_MIN = 1e-6

# Surrogate normalization, history selection, TV-GP geometry, and acquisition
# follow Ray PB2 directly in the PB2-core scheduler.

# =============================================================================
# CHAMPION-SELECTION PROTOCOL
# =============================================================================

TERMINAL_WINDOW_STEPS = 200_000
MIN_TERMINAL_SUPPORT_STEPS = 100_000
MIN_POST_CONFIG_SUPPORT_STEPS = 100_000
MIN_TERMINAL_POINTS = 3

# =============================================================================
# HELPERS
# =============================================================================

def search_space_dict() -> Dict[str, Any]:
    """Return a deep copy of the raw PPO search-domain bounds."""
    return deepcopy(HYPERPARAM_BOUNDS)


def tune_search_space() -> Dict[str, Any]:
    """Return common Ray Tune initialization distributions."""
    from ray import tune

    return {
        "train_batch_size": tune.randint(1_000, 60_001),
        "lambda": tune.uniform(0.90, 0.99),
        "clip_param": tune.uniform(0.10, 0.50),
        "lr": tune.loguniform(1e-5, 1e-3),
    }


def entropy_bounds() -> tuple[float, float]:
    """Return CODA-PPO O2I actuator bounds."""
    return BASE_ENTROPY_COEFF, ENTROPY_GUARD


def gated_o2i_uncertainty(
    uncertainty: float,
    n_observations: int,
) -> float:
    """Apply CODA's threshold and observation-count warm-up to O2I."""
    import numpy as np

    u = float(np.clip(float(uncertainty), 0.0, 1.0))
    u_thresholded = float(
        np.clip(
            (u - O2I_UNCERTAINTY_THRESHOLD)
            / (1.0 - O2I_UNCERTAINTY_THRESHOLD),
            0.0,
            1.0,
        )
    )

    if O2I_WARMUP_END <= O2I_WARMUP_START:
        gain = float(int(n_observations) >= O2I_WARMUP_END)
    else:
        gain = float(
            np.clip(
                (float(n_observations) - O2I_WARMUP_START)
                / (O2I_WARMUP_END - O2I_WARMUP_START),
                0.0,
                1.0,
            )
        )

    return float(u_thresholded * gain)


def entropy_from_uncertainty(
    uncertainty: float,
    n_observations: int,
) -> float:
    """Map raw donor-relative uncertainty to gated PPO entropy."""
    import numpy as np

    u_eff = gated_o2i_uncertainty(uncertainty, n_observations)
    increment = min(O2I_MAX_INCREMENT, O2I_UNCERTAINTY_SCALE * u_eff)
    return float(
        np.clip(
            BASE_ENTROPY_COEFF + increment,
            BASE_ENTROPY_COEFF,
            ENTROPY_GUARD,
        )
    )


def build_ppo_config(
    env_name: str,
    seed: int,
    *,
    callbacks: Optional[type] = None,
):
    """Build the common legacy-stack RLlib PPOConfig."""
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
            # Placeholders; Tune/outer scheduler replaces these.
            train_batch_size=1_000,
            lr=1e-4,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=BASE_ENTROPY_COEFF,
            vf_loss_coeff=FIXED_VF_LOSS_COEFF,
            minibatch_size=MINIBATCH_SIZE,
            use_kl_loss=USE_KL_LOSS,
            kl_coeff=KL_COEFF,
            kl_target=POLICY_KL_REFERENCE,
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
    """Build PPOConfig using the common initialization distributions."""
    cfg = build_ppo_config(env_name, seed, callbacks=callbacks)
    space = tune_search_space()

    return cfg.training(
        train_batch_size=space["train_batch_size"],
        lr=space["lr"],
        lambda_=space["lambda"],
        clip_param=space["clip_param"],
    )
