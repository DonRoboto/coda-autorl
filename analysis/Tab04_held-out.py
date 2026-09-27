#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 20:57:14 2026

@author: yor5
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
PPO_FILE = "../results/ppo/heldout_reward_ppo_final/heldout_test_episodes.csv"
SAC_FILE = "../results/sac/heldout_reward_sac_final/heldout_test_episodes.csv"

OUTPUT_BODY = "heldout_table_body.tex"
OUTPUT_SEED_LEVEL = "heldout_seed_level_means.csv"
OUTPUT_SUMMARY = "heldout_method_environment_summary.csv"

N_TRAINING_SEEDS = 10
N_TEST_EPISODES = 100

METHODS = [
    "PBT",
    "PB2",
    "ASHA",
    "CODA",
]

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


# ============================================================
# Load original episode-level held-out data
# ============================================================

def load_episodes(path, learner):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
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
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["learner"] = learner

    # Ensure numeric fields really are numeric
    df["training_seed"] = pd.to_numeric(
        df["training_seed"],
        errors="raise"
    ).astype(int)

    df["test_return"] = pd.to_numeric(
        df["test_return"],
        errors="raise"
    )

    # Keep only methods used in Table 4.
    # Directional CODA variants, if present, are intentionally excluded.
    df = df[
        df["method"].isin(METHODS)
    ].copy()

    return df


ppo = load_episodes(
    PPO_FILE,
    "PPO"
)

sac = load_episodes(
    SAC_FILE,
    "SAC"
)

episodes = pd.concat(
    [ppo, sac],
    ignore_index=True
)


# ============================================================
# Basic validation
# ============================================================

if not np.isfinite(
    episodes["test_return"]
).all():
    raise ValueError(
        "Non-finite values found in test_return."
    )


# ============================================================
# Validate each learner/method/environment/training-seed case
# ============================================================

case_keys = [
    "learner",
    "method",
    "environment",
    "training_seed",
]

case_check = (
    episodes
    .groupby(case_keys, as_index=False)
    .agg(
        n_rows=("test_return", "size"),
        n_champions=(
            "champion_agent",
            "nunique"
        ) if "champion_agent" in episodes.columns
        else ("test_return", "size"),
    )
)

# Exactly 100 held-out episodes per training seed
bad_episode_counts = case_check[
    case_check["n_rows"] != N_TEST_EPISODES
]

if not bad_episode_counts.empty:
    print(
        "\nCases with incorrect episode count:"
    )
    print(
        bad_episode_counts.to_string(index=False)
    )

    raise ValueError(
        "At least one case does not contain "
        f"exactly {N_TEST_EPISODES} held-out episodes."
    )


# If test_seed exists, verify 100 unique evaluation seeds
if "test_seed" in episodes.columns:

    unique_test_seeds = (
        episodes
        .groupby(case_keys)["test_seed"]
        .nunique()
    )

    if not (
        unique_test_seeds == N_TEST_EPISODES
    ).all():

        bad = unique_test_seeds[
            unique_test_seeds != N_TEST_EPISODES
        ]

        print(
            "\nCases with duplicated/missing test seeds:"
        )
        print(bad)

        raise ValueError(
            "Held-out test seeds are not unique "
            "within at least one case."
        )


# If champion identity exists, it must be constant within
# one learner/method/environment/training_seed case.
if "champion_agent" in episodes.columns:

    champion_counts = (
        episodes
        .groupby(case_keys)[
            "champion_agent"
        ]
        .nunique()
    )

    if not (champion_counts == 1).all():

        bad = champion_counts[
            champion_counts != 1
        ]

        print(
            "\nCases with more than one champion:"
        )
        print(bad)

        raise ValueError(
            "More than one champion appears "
            "within at least one held-out case."
        )


# Optional: verify evaluation was explore=False
if "explore" in episodes.columns:

    explore_normalized = (
        episodes["explore"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    valid_false = explore_normalized.isin(
        ["false", "0"]
    )

    if not valid_false.all():
        raise ValueError(
            "At least one held-out episode "
            "was not evaluated with explore=False."
        )


# ============================================================
# Step 1:
# Seed-level held-out outcome
#
# G_{g,m,e,s} = mean over the 100 held-out episodes
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
# Verify 10 independent training seeds per cell
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
    "\nTraining seeds per "
    "learner/method/environment:\n"
)
print(seed_counts)

if not (
    seed_counts == N_TRAINING_SEEDS
).all():

    bad = seed_counts[
        seed_counts != N_TRAINING_SEEDS
    ]

    print(
        "\nCells without exactly "
        f"{N_TRAINING_SEEDS} training seeds:"
    )
    print(bad)

    raise ValueError(
        "Incomplete held-out experimental design."
    )


# ============================================================
# Check that every expected cell exists
# ============================================================

expected = {
    (learner, method, env)
    for learner in LEARNERS
    for method in METHODS
    for env in ENVIRONMENTS
}

observed = set(
    seed_level[
        [
            "learner",
            "method",
            "environment",
        ]
    ]
    .itertuples(
        index=False,
        name=None
    )
)

missing_cells = expected - observed
extra_cells = observed - expected

if missing_cells:
    raise ValueError(
        f"Missing experimental cells: "
        f"{sorted(missing_cells)}"
    )

if extra_cells:
    print(
        "\nWARNING: extra experimental cells "
        "were found and will not be printed:"
    )
    print(
        sorted(extra_cells)
    )


# ============================================================
# Step 2:
# Aggregate across the 10 independent training seeds
# ============================================================

def q1(x):
    return float(
        x.quantile(0.25)
    )


def q3(x):
    return float(
        x.quantile(0.75)
    )


summary = (
    seed_level
    .groupby(
        [
            "learner",
            "method",
            "environment",
        ],
        as_index=False
    )
    .agg(
        n_training_seeds=(
            "training_seed",
            "nunique"
        ),
        median=(
            "test_mean_return",
            "median"
        ),
        q1=(
            "test_mean_return",
            q1
        ),
        q3=(
            "test_mean_return",
            q3
        ),
    )
)


# ============================================================
# Determine largest median within each learner/environment
# ============================================================

best_median = (
    summary
    .groupby(
        [
            "learner",
            "environment",
        ]
    )["median"]
    .transform("max")
)

summary["is_best"] = np.isclose(
    summary["median"],
    best_median,
    rtol=0.0,
    atol=1e-12
)


# ============================================================
# Save auditable intermediate tables
# ============================================================

seed_level.to_csv(
    OUTPUT_SEED_LEVEL,
    index=False
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# Helper: get one table cell
# ============================================================

def get_result(
    learner,
    environment,
    method
):
    row = summary[
        (summary["learner"] == learner)
        &
        (summary["environment"] == environment)
        &
        (summary["method"] == method)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one row for "
            f"{learner}, {environment}, {method}; "
            f"found {len(row)}."
        )

    return row.iloc[0]


# ============================================================
# Format cell exactly as:
#
# median [Q1, Q3]
#
# with boldface for largest marginal median
# ============================================================

def format_cell(row):
    text = (
        f"{row['median']:.1f} "
        f"[{row['q1']:.1f}, "
        f"{row['q3']:.1f}]"
    )

    if bool(row["is_best"]):
        return rf"\textbf{{{text}}}"

    return text


# ============================================================
# Generate LaTeX table BODY only
# ============================================================

lines = []

for learner_index, learner in enumerate(LEARNERS):

    if learner_index > 0:
        lines.append(r"\midrule")

    lines.append(
        rf"\multicolumn{{5}}{{l}}{{\textit{{{learner}}}}} \\[1pt]"
    )

    for environment in ENVIRONMENTS:

        cells = []

        for method in METHODS:
            row = get_result(
                learner,
                environment,
                method
            )

            cells.append(
                format_cell(row)
            )

        latex_row = (
            f"{environment} &\n"
            f"{cells[0]} &\n"
            f"{cells[1]} &\n"
            f"{cells[2]} &\n"
            f"{cells[3]} \\\\"
        )

        lines.append(
            latex_row
        )
        lines.append("")


latex_body = "\n".join(lines).rstrip()


# ============================================================
# Print and save
# ============================================================

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

Path(
    OUTPUT_BODY
).write_text(
    latex_body + "\n",
    encoding="utf-8"
)

print(
    "\nGenerated files:"
)

print(
    f"  {OUTPUT_BODY}"
)

print(
    f"  {OUTPUT_SEED_LEVEL}"
)

print(
    f"  {OUTPUT_SUMMARY}"
)