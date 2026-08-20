"""Reproducibility seeds shared by training and held-out evaluation."""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Independent training seeds
# ---------------------------------------------------------------------------

TRAINING_SEEDS: List[int] = [
    1042,
    2854,
    3910,
    4721,
    5603,
    6198,
    7433,
    8256,
    9107,
    9845,
]

# ---------------------------------------------------------------------------
# Held-out evaluation seeds
# ---------------------------------------------------------------------------

N_HELDOUT_EPISODES = 100

# Primary held-out return evaluation.
HELDOUT_RETURN_SEED_START = 100_000
HELDOUT_RETURN_SEEDS: List[int] = list(
    range(
        HELDOUT_RETURN_SEED_START,
        HELDOUT_RETURN_SEED_START + N_HELDOUT_EPISODES,
    )
)

# Separate block for post-hoc physical/control-quality analysis.
STABILITY_SEED_START = 200_000
STABILITY_SEEDS: List[int] = list(
    range(
        STABILITY_SEED_START,
        STABILITY_SEED_START + N_HELDOUT_EPISODES,
    )
)

# Default smoke-test training seed.
SMOKE_TRAINING_SEED = TRAINING_SEEDS[0]


def validate_seed_protocol() -> None:
    """Reject accidental overlap between training and held-out seed blocks."""
    training = set(TRAINING_SEEDS)
    heldout_return = set(HELDOUT_RETURN_SEEDS)
    stability = set(STABILITY_SEEDS)

    if training & heldout_return:
        raise RuntimeError(
            "Training seeds overlap held-out return seeds: "
            f"{sorted(training & heldout_return)}"
        )

    if training & stability:
        raise RuntimeError(
            "Training seeds overlap stability seeds: "
            f"{sorted(training & stability)}"
        )

    if heldout_return & stability:
        raise RuntimeError(
            "Held-out return seeds overlap stability seeds: "
            f"{sorted(heldout_return & stability)}"
        )


validate_seed_protocol()
