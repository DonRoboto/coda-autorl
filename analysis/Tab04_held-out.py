#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table 4: Held-out task performance under PPO and SAC.

For each learner, method, environment, and training seed, this script:

1. Reads episode-level held-out evaluation results.
2. Validates the held-out evaluation protocol.
3. Computes one held-out mean return per training seed.
4. Aggregates the ten independent training seeds using:
   - median,
   - first quartile (Q1),
   - third quartile (Q3).
5. Identifies the largest marginal median within each
   learner/environment cell.
6. Saves auditable intermediate numerical artifacts.
7. Generates the LaTeX table body used in the manuscript.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── Tab04_held-out.py
└── results/
    ├── ppo/
    │   └── heldout_reward_ppo_final/
    │       └── heldout_test_episodes.csv
    └── sac/
        └── heldout_reward_sac_final/
            └── heldout_test_episodes.csv

Outputs
-------

results/analysis/heldout_performance/
├── heldout_seed_level_means.csv
├── heldout_method_environment_summary.csv
└── heldout_table_body.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

# Works when this file is stored under:
#     <repo_root>/analysis/Tab04_held-out.py
#
# The fallback also allows execution from an interactive session.
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent

PPO_FILE = (
    REPO_ROOT
    / "results"
    / "ppo"
    / "heldout_reward_ppo_final"
    / "heldout_test_episodes.csv"
)

SAC_FILE = (
    REPO_ROOT
    / "results"
    / "sac"
    / "heldout_reward_sac_final"
    / "heldout_test_episodes.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "heldout_performance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_BODY = (
    OUTPUT_DIR
    / "heldout_table_body.tex"
)

OUTPUT_SEED_LEVEL = (
    OUTPUT_DIR
    / "heldout_seed_level_means.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "heldout_method_environment_summary.csv"
)


# ============================================================
# Analysis configuration
# ============================================================

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
# Input loading
# ============================================================

def load_episodes(
    path,
    learner,
):
    """
    Load one episode-level held-out evaluation file and attach
    the learner label.

    Only the four methods reported in Table 4 are retained.
    Directional CODA variants are intentionally excluded.
    """
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            "Held-out evaluation file not found:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "method",
        "environment",
        "training_seed",
        "test_return",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df[
        "learner"
    ] = learner

    df[
        "training_seed"
    ] = pd.to_numeric(
        df[
            "training_seed"
        ],
        errors="raise",
    ).astype(int)

    df[
        "test_return"
    ] = pd.to_numeric(
        df[
            "test_return"
        ],
        errors="raise",
    )

    # Keep only methods reported in Table 4.
    df = df[
        df[
            "method"
        ].isin(
            METHODS
        )
    ].copy()

    return df


# ============================================================
# Episode-level validation
# ============================================================

def validate_episode_level_data(
    episodes,
):
    """
    Validate the held-out evaluation protocol before aggregation.
    """
    required_columns = {
        "learner",
        "method",
        "environment",
        "training_seed",
        "test_return",
    }

    missing = (
        required_columns
        - set(episodes.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if not np.isfinite(
        episodes[
            "test_return"
        ]
    ).all():

        raise ValueError(
            "Non-finite values found in test_return."
        )

    case_keys = [
        "learner",
        "method",
        "environment",
        "training_seed",
    ]

    # --------------------------------------------------------
    # Exactly 100 held-out episodes per training seed
    # --------------------------------------------------------

    case_counts = (
        episodes
        .groupby(
            case_keys,
            as_index=False,
        )
        .agg(
            n_rows=(
                "test_return",
                "size",
            )
        )
    )

    bad_episode_counts = case_counts[
        case_counts[
            "n_rows"
        ]
        != N_TEST_EPISODES
    ]

    if not bad_episode_counts.empty:

        print(
            "\nCases with incorrect held-out episode count:"
        )

        print(
            bad_episode_counts.to_string(
                index=False
            )
        )

        raise ValueError(
            "At least one case does not contain "
            f"exactly {N_TEST_EPISODES} held-out episodes."
        )

    # --------------------------------------------------------
    # Optional validation: 100 unique test seeds
    # --------------------------------------------------------

    if (
        "test_seed"
        in episodes.columns
    ):

        unique_test_seeds = (
            episodes
            .groupby(
                case_keys
            )[
                "test_seed"
            ]
            .nunique()
        )

        if not (
            unique_test_seeds
            == N_TEST_EPISODES
        ).all():

            bad = unique_test_seeds[
                unique_test_seeds
                != N_TEST_EPISODES
            ]

            print(
                "\nCases with duplicated or missing test seeds:"
            )

            print(
                bad
            )

            raise ValueError(
                "Held-out test seeds are not unique "
                "within at least one case."
            )

    # --------------------------------------------------------
    # Optional validation: one champion per case
    # --------------------------------------------------------

    if (
        "champion_agent"
        in episodes.columns
    ):

        champion_counts = (
            episodes
            .groupby(
                case_keys
            )[
                "champion_agent"
            ]
            .nunique()
        )

        if not (
            champion_counts
            == 1
        ).all():

            bad = champion_counts[
                champion_counts
                != 1
            ]

            print(
                "\nCases with more than one champion:"
            )

            print(
                bad
            )

            raise ValueError(
                "More than one champion appears "
                "within at least one held-out case."
            )

    # --------------------------------------------------------
    # Optional validation: explore=False
    # --------------------------------------------------------

    if (
        "explore"
        in episodes.columns
    ):

        explore_normalized = (
            episodes[
                "explore"
            ]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        valid_false = (
            explore_normalized
            .isin(
                [
                    "false",
                    "0",
                ]
            )
        )

        if not valid_false.all():

            raise ValueError(
                "At least one held-out episode "
                "was not evaluated with explore=False."
            )


# ============================================================
# Seed-level held-out outcome
#
# G_{g,m,e,s} = mean over the 100 held-out episodes
# ============================================================

def compute_seed_level_results(
    episodes,
):
    """
    Compute one held-out mean return per
    learner/method/environment/training seed.
    """
    seed_level = (
        episodes
        .groupby(
            [
                "learner",
                "method",
                "environment",
                "training_seed",
            ],
            as_index=False,
        )
        .agg(
            n_test_episodes=(
                "test_return",
                "size",
            ),
            test_mean_return=(
                "test_return",
                "mean",
            ),
        )
    )

    return seed_level


# ============================================================
# Experimental-design validation
# ============================================================

def validate_design(
    seed_level,
):
    """
    Verify ten training seeds for every expected
    learner/method/environment cell and ensure that all expected
    cells are present.
    """
    seed_counts = (
        seed_level
        .groupby(
            [
                "learner",
                "method",
                "environment",
            ],
            as_index=False,
        )
        .agg(
            n_training_seeds=(
                "training_seed",
                "nunique",
            )
        )
    )

    bad = seed_counts[
        seed_counts[
            "n_training_seeds"
        ]
        != N_TRAINING_SEEDS
    ]

    if not bad.empty:

        print(
            "\nCells without exactly "
            f"{N_TRAINING_SEEDS} training seeds:"
        )

        print(
            bad.to_string(
                index=False
            )
        )

        raise ValueError(
            "Incomplete held-out experimental design."
        )

    expected = {
        (
            learner,
            method,
            environment,
        )
        for learner in LEARNERS
        for method in METHODS
        for environment in ENVIRONMENTS
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
            name=None,
        )
    )

    missing_cells = (
        expected
        - observed
    )

    extra_cells = (
        observed
        - expected
    )

    if missing_cells:

        raise ValueError(
            "Missing experimental cells: "
            f"{sorted(missing_cells)}"
        )

    if extra_cells:

        print(
            "\nWARNING: extra experimental cells "
            "were found and will not be printed:"
        )

        print(
            sorted(
                extra_cells
            )
        )

    return seed_counts


# ============================================================
# Across-seed summary
# ============================================================

def q1(
    x,
):
    return float(
        x.quantile(
            0.25
        )
    )


def q3(
    x,
):
    return float(
        x.quantile(
            0.75
        )
    )


def compute_summary(
    seed_level,
):
    """
    Aggregate the ten independent training seeds using
    median [Q1, Q3].
    """
    summary = (
        seed_level
        .groupby(
            [
                "learner",
                "method",
                "environment",
            ],
            as_index=False,
        )
        .agg(
            n_training_seeds=(
                "training_seed",
                "nunique",
            ),
            median=(
                "test_mean_return",
                "median",
            ),
            q1=(
                "test_mean_return",
                q1,
            ),
            q3=(
                "test_mean_return",
                q3,
            ),
        )
    )

    # Largest marginal median within each learner/environment.
    best_median = (
        summary
        .groupby(
            [
                "learner",
                "environment",
            ]
        )[
            "median"
        ]
        .transform(
            "max"
        )
    )

    summary[
        "is_best"
    ] = np.isclose(
        summary[
            "median"
        ],
        best_median,
        rtol=0.0,
        atol=1e-12,
    )

    return summary


# ============================================================
# LaTeX helpers
# ============================================================

def get_result(
    summary,
    learner,
    environment,
    method,
):
    """
    Return exactly one summary row for the requested table cell.
    """
    row = summary[
        summary[
            "learner"
        ].eq(
            learner
        )
        &
        summary[
            "environment"
        ].eq(
            environment
        )
        &
        summary[
            "method"
        ].eq(
            method
        )
    ]

    if len(row) != 1:

        raise ValueError(
            "Expected exactly one row for "
            f"{learner}, {environment}, {method}; "
            f"found {len(row)}."
        )

    return row.iloc[
        0
    ]


def format_cell(
    row,
):
    """
    Format one Table 4 cell as:

        median [Q1, Q3]

    Boldface is used for the largest marginal median within each
    learner/environment cell.
    """
    text_value = (
        f"{row['median']:.1f} "
        f"[{row['q1']:.1f}, "
        f"{row['q3']:.1f}]"
    )

    if bool(
        row[
            "is_best"
        ]
    ):

        return (
            rf"\textbf{{{text_value}}}"
        )

    return text_value


def build_latex_body(
    summary,
):
    """
    Build the body of manuscript Table 4.
    """
    lines = []

    for learner_index, learner in enumerate(
        LEARNERS
    ):

        if learner_index > 0:

            lines.append(
                r"\midrule"
            )

        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\textit{{{learner}}}}} \\[1pt]"
        )

        for environment in ENVIRONMENTS:

            cells = []

            for method in METHODS:

                row = get_result(
                    summary,
                    learner,
                    environment,
                    method,
                )

                cells.append(
                    format_cell(
                        row
                    )
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

            lines.append(
                ""
            )

    return (
        "\n".join(
            lines
        )
        .rstrip()
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load original episode-level data
    # --------------------------------------------------------

    print(
        "Loading PPO held-out evaluations..."
    )

    ppo = load_episodes(
        PPO_FILE,
        learner="PPO",
    )

    print(
        "Loading SAC held-out evaluations..."
    )

    sac = load_episodes(
        SAC_FILE,
        learner="SAC",
    )

    episodes = pd.concat(
        [
            ppo,
            sac,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validate episode-level protocol
    # --------------------------------------------------------

    validate_episode_level_data(
        episodes
    )

    # --------------------------------------------------------
    # Seed-level held-out outcomes
    # --------------------------------------------------------

    seed_level = (
        compute_seed_level_results(
            episodes
        )
    )

    seed_counts = (
        validate_design(
            seed_level
        )
    )

    print(
        "\nTraining seeds per "
        "learner/method/environment:"
    )

    print(
        seed_counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Aggregate across training seeds
    # --------------------------------------------------------

    summary = (
        compute_summary(
            seed_level
        )
    )

    # --------------------------------------------------------
    # Save auditable numerical artifacts
    # --------------------------------------------------------

    seed_level.to_csv(
        OUTPUT_SEED_LEVEL,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # Generate LaTeX table body
    # --------------------------------------------------------

    latex_body = (
        build_latex_body(
            summary
        )
    )

    OUTPUT_BODY.write_text(
        latex_body
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )

    print(
        "HELD-OUT TASK PERFORMANCE"
    )

    print(
        "=" * 88
    )

    print(
        summary[
            [
                "learner",
                "environment",
                "method",
                "n_training_seeds",
                "median",
                "q1",
                "q3",
                "is_best",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "LATEX TABLE BODY"
    )

    print(
        "=" * 88
        + "\n"
    )

    print(
        latex_body
    )

    # --------------------------------------------------------
    # Report generated files
    # --------------------------------------------------------

    print(
        "\nGenerated outputs:"
    )

    print(
        f"  Seed-level outcomes : {OUTPUT_SEED_LEVEL}"
    )

    print(
        f"  Summary             : {OUTPUT_SUMMARY}"
    )

    print(
        f"  LaTeX table body    : {OUTPUT_BODY}"
    )

    print(
        "\nDone."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()