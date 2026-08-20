"""Environment definitions shared by PPO and SAC experiments.

The final cross-learner study uses the same four Gymnasium MuJoCo v5
continuous-control environments for PPO and SAC.

Keep environment selection in this module so training, evaluation, and
analysis scripts do not maintain independent copies of the benchmark list.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Final benchmark set
# ---------------------------------------------------------------------------

ENVIRONMENTS: List[str] = [
    "Hopper-v5",
    "HalfCheetah-v5",
    "Walker2d-v5",
    "Swimmer-v5",
]

# Optional human-readable labels for tables/figures.
ENVIRONMENT_LABELS: Dict[str, str] = {
    "Hopper-v5": "Hopper",
    "HalfCheetah-v5": "HalfCheetah",
    "Walker2d-v5": "Walker2d",
    "Swimmer-v5": "Swimmer",
}

# Environments previously used during development but excluded from the final
# common PPO/SAC comparison.
OPTIONAL_ENVIRONMENTS: List[str] = [
    "Ant-v5",
    "Humanoid-v5",
]


def validate_environment(env_name: str) -> str:
    """Return env_name if it belongs to the final benchmark set."""
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown final-study environment {env_name!r}. "
            f"Expected one of {ENVIRONMENTS}."
        )
    return env_name


def action_dimension(env_name: str) -> int:
    """Return the continuous action dimension of a Gymnasium environment.

    Gymnasium is imported lazily so analysis scripts that only need the
    environment names do not require MuJoCo to be initialized.
    """
    validate_environment(env_name)

    import gymnasium as gym
    import numpy as np

    env = gym.make(env_name)
    try:
        shape = getattr(env.action_space, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError(
                f"{env_name} does not expose a continuous Box action space: "
                f"{env.action_space}"
            )

        dim = int(np.prod(shape))
        if dim <= 0:
            raise ValueError(f"Invalid action dimension for {env_name}: {shape}")
        return dim
    finally:
        env.close()


def sac_baseline_target_entropy(env_name: str) -> float:
    """Return the fixed SAC baseline target entropy -action_dim."""
    return -float(action_dimension(env_name))
