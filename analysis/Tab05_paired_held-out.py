#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:02:45 2026

@author: yor5
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests


# ============================================================
# Configuration
# ============================================================


PPO_HELDOUT = Path("../results/ppo/heldout_reward_ppo_final/heldout_test_episodes.csv")
SAC_HELDOUT = Path("../results/sac/heldout_reward_sac_final/heldout_test_episodes.csv")


N_TEST_EPISODES = 100
N_TRAINING_SEEDS = 10

N_BOOT = 10_000
BOOT_SEED = 20260913

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

LEARNERS = [
    "PPO",
    "SAC",
]

OUTPUT_SEED_LEVEL = "primary_seed_level_heldout.csv"
OUTPUT_PAIRED = "primary_coda_pb2_paired_differences.csv"
OUTPUT_STATS = "primary_coda_pb2_stats.csv"
OUTPUT_LATEX = "primary_paired_table_body.tex"


# ============================================================
# Load original episode-level data from ZIP
# ============================================================

def read_episode_file(path, learner):
    if not path.exists():
        raise FileNotFoundError(
            f"Held-out file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "method",
        "environment",
        "training_seed",
        "test_return",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()
    df["learner"] = learner

    return df

ppo = read_episode_file(
    PPO_HELDOUT,
    "PPO"
)

sac = read_episode_file(
    SAC_HELDOUT,
    "SAC"
)

episodes = pd.concat(
    [ppo, sac],
    ignore_index=True
)


# ============================================================
# Validate original data
# ============================================================

required_columns = {
    "method",
    "environment",
    "training_seed",
    "test_return",
}

missing = required_columns - set(episodes.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


episodes["training_seed"] = pd.to_numeric(
    episodes["training_seed"],
    errors="raise"
).astype(int)

episodes["test_return"] = pd.to_numeric(
    episodes["test_return"],
    errors="raise"
)

if not np.isfinite(
    episodes["test_return"]
).all():
    raise ValueError(
        "Non-finite held-out returns detected."
    )


# ============================================================
# Keep only primary methods
# ============================================================

episodes = episodes[
    episodes["method"].isin(
        ["CODA", "PB2"]
    )
].copy()


# ============================================================
# Validate 100 episodes per training seed
# ============================================================

case_keys = [
    "learner",
    "method",
    "environment",
    "training_seed",
]

case_counts = (
    episodes
    .groupby(case_keys)
    .size()
)

bad_counts = case_counts[
    case_counts != N_TEST_EPISODES
]

if not bad_counts.empty:
    print(
        "\nCases with incorrect number of "
        "held-out episodes:"
    )
    print(bad_counts)

    raise ValueError(
        "Expected exactly "
        f"{N_TEST_EPISODES} held-out episodes "
        "per training seed."
    )


# ============================================================
# Optional validation: common test seeds
# ============================================================

if "test_seed" in episodes.columns:

    test_seed_counts = (
        episodes
        .groupby(case_keys)["test_seed"]
        .nunique()
    )

    if not (
        test_seed_counts == N_TEST_EPISODES
    ).all():
        raise ValueError(
            "At least one case does not contain "
            f"{N_TEST_EPISODES} unique test seeds."
        )


# ============================================================
# Optional validation: one champion per case
# ============================================================

if "champion_agent" in episodes.columns:

    champion_counts = (
        episodes
        .groupby(case_keys)[
            "champion_agent"
        ]
        .nunique()
    )

    if not (
        champion_counts == 1
    ).all():
        raise ValueError(
            "More than one champion appears in "
            "at least one held-out case."
        )


# ============================================================
# Optional validation: explore=False
# ============================================================

if "explore" in episodes.columns:

    explore_values = (
        episodes["explore"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if not explore_values.isin(
        ["false", "0"]
    ).all():
        raise ValueError(
            "At least one held-out episode "
            "was evaluated with explore=True."
        )


# ============================================================
# Step 1:
# Compute seed-level held-out mean
#
# G_{g,m,e,s}
# ============================================================

seed_level = (
    episodes
    .groupby(
        [
            "learner",
            "method",
            "environment",
            "training_seed",
        ],
        as_index=False
    )
    .agg(
        n_test_episodes=(
            "test_return",
            "size"
        ),
        test_mean_return=(
            "test_return",
            "mean"
        ),
    )
)


# ============================================================
# Validate 10 training seeds per method/cell
# ============================================================

seed_counts = (
    seed_level
    .groupby(
        [
            "learner",
            "method",
            "environment",
        ]
    )["training_seed"]
    .nunique()
)

print(
    "\nTraining seeds per cell:\n"
)

print(seed_counts)

if not (
    seed_counts == N_TRAINING_SEEDS
).all():

    bad = seed_counts[
        seed_counts != N_TRAINING_SEEDS
    ]

    print(
        "\nIncomplete cells:"
    )
    print(bad)

    raise ValueError(
        "Expected exactly "
        f"{N_TRAINING_SEEDS} training seeds "
        "for every CODA/PB2 cell."
    )


seed_level.to_csv(
    OUTPUT_SEED_LEVEL,
    index=False
)


# ============================================================
# Build matched CODA-PB2 seed differences
# ============================================================

def paired_seed_data(
    learner,
    environment
):

    coda = (
        seed_level[
            (seed_level["learner"] == learner)
            &
            (seed_level["environment"] == environment)
            &
            (seed_level["method"] == "CODA")
        ][
            [
                "training_seed",
                "test_mean_return",
            ]
        ]
        .rename(
            columns={
                "test_mean_return":
                    "CODA"
            }
        )
    )

    pb2 = (
        seed_level[
            (seed_level["learner"] == learner)
            &
            (seed_level["environment"] == environment)
            &
            (seed_level["method"] == "PB2")
        ][
            [
                "training_seed",
                "test_mean_return",
            ]
        ]
        .rename(
            columns={
                "test_mean_return":
                    "PB2"
            }
        )
    )

    paired = (
        coda
        .merge(
            pb2,
            on="training_seed",
            how="inner",
            validate="one_to_one"
        )
        .sort_values(
            "training_seed"
        )
        .reset_index(drop=True)
    )

    if len(paired) != N_TRAINING_SEEDS:
        raise RuntimeError(
            f"{learner}, {environment}: "
            f"expected {N_TRAINING_SEEDS} "
            f"matched seeds, got {len(paired)}"
        )

    paired["difference"] = (
        paired["CODA"]
        - paired["PB2"]
    )

    paired["learner"] = learner
    paired["environment"] = environment

    return paired[
        [
            "learner",
            "environment",
            "training_seed",
            "CODA",
            "PB2",
            "difference",
        ]
    ]


paired_frames = []

for learner in LEARNERS:
    for environment in ENVIRONMENTS:

        paired_frames.append(
            paired_seed_data(
                learner,
                environment
            )
        )

paired_all = pd.concat(
    paired_frames,
    ignore_index=True
)

paired_all.to_csv(
    OUTPUT_PAIRED,
    index=False
)


# ============================================================
# Matched-pairs rank-biserial correlation
# ============================================================

def rank_biserial(
    differences
):
    d = np.asarray(
        differences,
        dtype=float
    )

    # Wilcoxon "wilcox" convention:
    # zero differences are excluded
    d = d[d != 0]

    if len(d) == 0:
        return np.nan

    ranks = rankdata(
        np.abs(d),
        method="average"
    )

    w_plus = ranks[
        d > 0
    ].sum()

    w_minus = ranks[
        d < 0
    ].sum()

    return float(
        (w_plus - w_minus)
        /
        (w_plus + w_minus)
    )


# ============================================================
# Bootstrap setup
#
# Important:
# This reproduces the seed-resampling pattern used in
# the original recomputation script.
# ============================================================

rng = np.random.default_rng(
    BOOT_SEED
)

boot_idx = rng.integers(
    low=0,
    high=N_TRAINING_SEEDS,
    size=(
        N_BOOT,
        N_TRAINING_SEEDS
    )
)


def bootstrap_median_ci(
    differences
):
    d = np.asarray(
        differences,
        dtype=float
    )

    if len(d) != N_TRAINING_SEEDS:
        raise ValueError(
            "Bootstrap function expects "
            f"{N_TRAINING_SEEDS} paired seeds."
        )

    boot_medians = np.median(
        d[boot_idx],
        axis=1
    )

    lo, hi = np.quantile(
        boot_medians,
        [
            0.025,
            0.975,
        ]
    )

    return (
        float(
            np.median(d)
        ),
        float(lo),
        float(hi),
    )


# ============================================================
# Compute eight primary comparisons
# ============================================================

rows = []

for learner in LEARNERS:

    for environment in ENVIRONMENTS:

        paired = paired_all[
            (paired_all["learner"] == learner)
            &
            (
                paired_all["environment"]
                == environment
            )
        ].copy()

        d = paired[
            "difference"
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # W/T/L
        # ----------------------------------------------------

        wins = int(
            np.sum(d > 0)
        )

        ties = int(
            np.sum(d == 0)
        )

        losses = int(
            np.sum(d < 0)
        )

        # Primary table currently contains no zero differences.
        # Exact Wilcoxon is therefore well-defined.
        if ties > 0:
            raise RuntimeError(
                f"{learner}, {environment}: "
                "zero paired difference detected. "
                "The original primary analysis used "
                "exact Wilcoxon with zero_method='wilcox'; "
                "inspect this case before continuing."
            )

        # ----------------------------------------------------
        # Median + paired bootstrap CI
        # ----------------------------------------------------

        median_diff, ci_low, ci_high = (
            bootstrap_median_ci(d)
        )

        # ----------------------------------------------------
        # Wilcoxon signed-rank
        # ----------------------------------------------------

        test = wilcoxon(
            d,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="exact",
        )

        # ----------------------------------------------------
        # Rank-biserial
        # ----------------------------------------------------

        r_rb = rank_biserial(
            d
        )

        rows.append(
            {
                "learner":
                    learner,

                "environment":
                    environment,

                "median_difference":
                    median_diff,

                "bootstrap_95_lo":
                    ci_low,

                "bootstrap_95_hi":
                    ci_high,

                "wins":
                    wins,

                "ties":
                    ties,

                "losses":
                    losses,

                "W":
                    float(
                        test.statistic
                    ),

                "p_raw":
                    float(
                        test.pvalue
                    ),

                "r_rb":
                    r_rb,
            }
        )


stats = pd.DataFrame(
    rows
)


# ============================================================
# Holm correction across ALL 8 primary hypotheses
# ============================================================

stats["p_Holm"] = multipletests(
    stats["p_raw"],
    alpha=0.05,
    method="holm"
)[1]


# ============================================================
# Save numerical audit table
# ============================================================

stats.to_csv(
    OUTPUT_STATS,
    index=False
)


# ============================================================
# Print full-precision audit output
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "PRIMARY CODA vs PB2"
)

print(
    "=" * 80
)

print(
    stats.to_string(
        index=False
    )
)

print(
    "\nSciPy version:",
    scipy.__version__
)

print(
    "Wilcoxon: exact, two-sided, "
    "zero_method='wilcox', "
    "correction=False"
)

print(
    f"Bootstrap: N={N_BOOT}, "
    f"percentile 95% CI, "
    f"seed={BOOT_SEED}"
)

print(
    "Holm family: 8 primary "
    "learner-environment comparisons"
)


# ============================================================
# LaTeX formatting helpers
# ============================================================

def format_effect(
    row
):
    return (
        f"${row['median_difference']:.2f} "
        f"[{row['bootstrap_95_lo']:.2f}, "
        f"{row['bootstrap_95_hi']:.2f}]$"
    )


def format_wtl(
    row
):
    return (
        f"{int(row['wins'])}/"
        f"{int(row['ties'])}/"
        f"{int(row['losses'])}"
    )


def format_p(
    p
):
    return f"{p:.3f}"


def format_holm(
    p
):
    text = f"{p:.3f}"

    if p < 0.05:
        return (
            rf"\textbf{{{text}}}"
        )

    return text


def format_rrb(
    r
):
    return f"{r:.2f}"


# ============================================================
# Generate LaTeX TABLE BODY
# ============================================================

lines = []

for learner_idx, learner in enumerate(
    LEARNERS
):

    if learner_idx > 0:
        lines.append(
            r"\midrule"
        )
        lines.append("")

    learner_rows = stats[
        stats["learner"] == learner
    ]

    for environment in ENVIRONMENTS:

        row = learner_rows[
            learner_rows[
                "environment"
            ] == environment
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Expected exactly one row for "
                f"{learner}, {environment}"
            )

        row = row.iloc[0]

        latex_row = (
            f"{learner} & "
            f"{environment} &\n"
            f"{format_effect(row)} &\n"
            f"{format_wtl(row)} & "
            f"{format_p(row['p_raw'])} & "
            f"{format_holm(row['p_Holm'])} & "
            f"{format_rrb(row['r_rb'])} "
            r"\\"
        )

        lines.append(
            latex_row
        )

        lines.append("")


latex_body = (
    "\n".join(lines)
    .rstrip()
)


# ============================================================
# Save LaTeX body
# ============================================================

Path(
    OUTPUT_LATEX
).write_text(
    latex_body + "\n",
    encoding="utf-8"
)


print(
    "\n"
    + "=" * 80
)

print(
    "LATEX TABLE BODY"
)

print(
    "=" * 80
    + "\n"
)

print(
    latex_body
)

print(
    "\nGenerated files:"
)

print(
    f"  {OUTPUT_SEED_LEVEL}"
)

print(
    f"  {OUTPUT_PAIRED}"
)

print(
    f"  {OUTPUT_STATS}"
)

print(
    f"  {OUTPUT_LATEX}"
)