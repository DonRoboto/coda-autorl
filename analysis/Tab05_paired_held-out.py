#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table 5: Primary paired held-out CODA vs PB2 analysis.

For each learner (PPO/SAC) and environment, this script:

1. Reads episode-level held-out evaluation results.
2. Validates the held-out evaluation protocol.
3. Computes one held-out mean return per training seed.
4. Forms matched CODA - PB2 seed-level differences.
5. Reports:
   - median paired difference,
   - percentile 95% paired-bootstrap confidence interval,
   - win/tie/loss counts,
   - exact two-sided Wilcoxon signed-rank test,
   - Holm-adjusted p-value across the eight primary comparisons,
   - matched-pairs rank-biserial correlation.
6. Saves all intermediate and final numerical artifacts.
7. Generates the LaTeX table body used for the manuscript.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── Tab05_paired_held-out.py
└── results/
    ├── ppo/
    │   └── heldout_reward_ppo_final/
    │       └── heldout_test_episodes.csv
    └── sac/
        └── heldout_reward_sac_final/
            └── heldout_test_episodes.csv

Outputs
-------

results/analysis/primary_paired_heldout/
├── primary_seed_level_heldout.csv
├── primary_coda_pb2_paired_differences.csv
├── primary_coda_pb2_stats.csv
└── primary_paired_table_body.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# Paths
# ============================================================

# Works when this file is stored under:
#     <repo_root>/analysis/Tab05_paired_held-out.py
#
# The fallback also allows execution from an interactive session.
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent

PPO_HELDOUT = (
    REPO_ROOT
    / "results"
    / "ppo"
    / "heldout_reward_ppo_final"
    / "heldout_test_episodes.csv"
)

SAC_HELDOUT = (
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
    / "primary_paired_heldout"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_SEED_LEVEL = (
    OUTPUT_DIR
    / "primary_seed_level_heldout.csv"
)

OUTPUT_PAIRED = (
    OUTPUT_DIR
    / "primary_coda_pb2_paired_differences.csv"
)

OUTPUT_STATS = (
    OUTPUT_DIR
    / "primary_coda_pb2_stats.csv"
)

OUTPUT_LATEX = (
    OUTPUT_DIR
    / "primary_paired_table_body.tex"
)


# ============================================================
# Analysis configuration
# ============================================================

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


# ============================================================
# Input loading
# ============================================================

def read_episode_file(
    path,
    learner,
):
    """
    Read one episode-level held-out evaluation file and attach
    its learner label.
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
    df["learner"] = learner

    return df


# ============================================================
# Data validation
# ============================================================

def validate_episode_level_data(
    episodes,
):
    """
    Validate the episode-level held-out evaluation protocol.
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

    episodes = episodes.copy()

    episodes[
        "training_seed"
    ] = pd.to_numeric(
        episodes[
            "training_seed"
        ],
        errors="raise",
    ).astype(int)

    episodes[
        "test_return"
    ] = pd.to_numeric(
        episodes[
            "test_return"
        ],
        errors="raise",
    )

    if not np.isfinite(
        episodes[
            "test_return"
        ]
    ).all():

        raise ValueError(
            "Non-finite held-out returns detected."
        )

    # Primary comparison uses CODA and PB2 only.
    episodes = episodes[
        episodes[
            "method"
        ].isin(
            ["CODA", "PB2"]
        )
    ].copy()

    case_keys = [
        "learner",
        "method",
        "environment",
        "training_seed",
    ]

    # --------------------------------------------------------
    # Validate held-out episode count
    # --------------------------------------------------------

    case_counts = (
        episodes
        .groupby(
            case_keys
        )
        .size()
    )

    bad_counts = case_counts[
        case_counts
        != N_TEST_EPISODES
    ]

    if not bad_counts.empty:

        print(
            "\nCases with incorrect number of "
            "held-out episodes:"
        )

        print(
            bad_counts
        )

        raise ValueError(
            "Expected exactly "
            f"{N_TEST_EPISODES} held-out episodes "
            "per training seed."
        )

    # --------------------------------------------------------
    # Optional validation: common test seeds
    # --------------------------------------------------------

    if (
        "test_seed"
        in episodes.columns
    ):

        test_seed_counts = (
            episodes
            .groupby(
                case_keys
            )[
                "test_seed"
            ]
            .nunique()
        )

        if not (
            test_seed_counts
            == N_TEST_EPISODES
        ).all():

            raise ValueError(
                "At least one case does not contain "
                f"{N_TEST_EPISODES} unique test seeds."
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

            raise ValueError(
                "More than one champion appears in "
                "at least one held-out case."
            )

    # --------------------------------------------------------
    # Optional validation: explore=False
    # --------------------------------------------------------

    if (
        "explore"
        in episodes.columns
    ):

        explore_values = (
            episodes[
                "explore"
            ]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        if not explore_values.isin(
            [
                "false",
                "0",
            ]
        ).all():

            raise ValueError(
                "At least one held-out episode "
                "was evaluated with explore=True."
            )

    return episodes


# ============================================================
# Seed-level held-out outcome
#
# G_{g,m,e,s}
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


def validate_training_seed_counts(
    seed_level,
):
    """
    Validate the expected number of independent training seeds.
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
            "\nIncomplete cells:"
        )

        print(
            bad.to_string(
                index=False
            )
        )

        raise ValueError(
            "Expected exactly "
            f"{N_TRAINING_SEEDS} training seeds "
            "for every CODA/PB2 cell."
        )

    return seed_counts


# ============================================================
# Matched CODA-PB2 seed differences
# ============================================================

def paired_seed_data(
    seed_level,
    learner,
    environment,
):
    """
    Build matched seed-level CODA and PB2 held-out outcomes.
    """
    coda = (
        seed_level[
            seed_level[
                "learner"
            ].eq(
                learner
            )
            &
            seed_level[
                "environment"
            ].eq(
                environment
            )
            &
            seed_level[
                "method"
            ].eq(
                "CODA"
            )
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
            seed_level[
                "learner"
            ].eq(
                learner
            )
            &
            seed_level[
                "environment"
            ].eq(
                environment
            )
            &
            seed_level[
                "method"
            ].eq(
                "PB2"
            )
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
            validate="one_to_one",
        )
        .sort_values(
            "training_seed"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(paired)
        != N_TRAINING_SEEDS
    ):

        raise RuntimeError(
            f"{learner}, {environment}: "
            f"expected {N_TRAINING_SEEDS} "
            f"matched seeds, got {len(paired)}"
        )

    paired[
        "difference"
    ] = (
        paired[
            "CODA"
        ]
        - paired[
            "PB2"
        ]
    )

    paired.insert(
        0,
        "environment",
        environment,
    )

    paired.insert(
        0,
        "learner",
        learner,
    )

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


# ============================================================
# Matched-pairs rank-biserial correlation
# ============================================================

def rank_biserial(
    differences,
):
    """
    Matched-pairs rank-biserial correlation.

    Zero differences are excluded, matching
    zero_method='wilcox'.
    """
    d = np.asarray(
        differences,
        dtype=float,
    )

    d = d[
        np.isfinite(d)
    ]

    d = d[
        d != 0
    ]

    if len(d) == 0:
        return np.nan

    ranks = rankdata(
        np.abs(d),
        method="average",
    )

    w_plus = ranks[
        d > 0
    ].sum()

    w_minus = ranks[
        d < 0
    ].sum()

    denominator = (
        w_plus
        + w_minus
    )

    if denominator == 0:
        return np.nan

    return float(
        (
            w_plus
            - w_minus
        )
        / denominator
    )


# ============================================================
# Bootstrap setup
#
# This preserves the resampling specification used in the
# current primary analysis.
# ============================================================

rng = np.random.default_rng(
    BOOT_SEED
)

boot_idx = rng.integers(
    low=0,
    high=N_TRAINING_SEEDS,
    size=(
        N_BOOT,
        N_TRAINING_SEEDS,
    ),
)


def bootstrap_median_ci(
    differences,
):
    """
    Percentile paired-bootstrap CI for the median
    CODA - PB2 matched-seed difference.
    """
    d = np.asarray(
        differences,
        dtype=float,
    )

    if (
        len(d)
        != N_TRAINING_SEEDS
    ):

        raise ValueError(
            "Bootstrap function expects "
            f"{N_TRAINING_SEEDS} paired seeds."
        )

    boot_medians = np.median(
        d[
            boot_idx
        ],
        axis=1,
    )

    lo, hi = np.quantile(
        boot_medians,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(
            np.median(
                d
            )
        ),
        float(lo),
        float(hi),
    )


# ============================================================
# Primary statistical analysis
# ============================================================

def compute_primary_stats(
    paired_all,
):
    """
    Compute the eight primary CODA-vs-PB2 comparisons.
    """
    rows = []

    for learner in LEARNERS:

        for environment in ENVIRONMENTS:

            paired = paired_all[
                paired_all[
                    "learner"
                ].eq(
                    learner
                )
                &
                paired_all[
                    "environment"
                ].eq(
                    environment
                )
            ].copy()

            d = paired[
                "difference"
            ].to_numpy(
                dtype=float
            )

            wins = int(
                np.sum(
                    d > 0
                )
            )

            ties = int(
                np.sum(
                    d == 0
                )
            )

            losses = int(
                np.sum(
                    d < 0
                )
            )

            # The current primary family contains no zero differences.
            # Exact Wilcoxon is therefore well-defined.
            if ties > 0:

                raise RuntimeError(
                    f"{learner}, {environment}: "
                    "zero paired difference detected. "
                    "The current primary analysis uses "
                    "exact Wilcoxon with zero_method='wilcox'; "
                    "inspect this case before continuing."
                )

            median_diff, ci_low, ci_high = (
                bootstrap_median_ci(
                    d
                )
            )

            test = wilcoxon(
                d,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="exact",
            )

            r_rb = (
                rank_biserial(
                    d
                )
            )

            rows.append(
                {
                    "learner":
                        learner,

                    "environment":
                        environment,

                    "n_matched_seeds":
                        len(d),

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

                    "W_T_L":
                        (
                            f"{wins}/"
                            f"{ties}/"
                            f"{losses}"
                        ),

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

    # Holm correction across all 8 primary hypotheses.
    stats[
        "p_Holm"
    ] = multipletests(
        stats[
            "p_raw"
        ],
        alpha=0.05,
        method="holm",
    )[1]

    stats[
        "significant_Holm"
    ] = (
        stats[
            "p_Holm"
        ]
        < 0.05
    )

    return stats


# ============================================================
# LaTeX formatting helpers
# ============================================================

def format_effect(
    row,
):
    return (
        f"${row['median_difference']:.2f} "
        f"[{row['bootstrap_95_lo']:.2f}, "
        f"{row['bootstrap_95_hi']:.2f}]$"
    )


def format_wtl(
    row,
):
    return str(
        row[
            "W_T_L"
        ]
    )


def format_p(
    p,
):
    return (
        f"{p:.3f}"
    )


def format_holm(
    p,
):
    text_value = (
        f"{p:.3f}"
    )

    if (
        p < 0.05
    ):

        return (
            rf"\textbf{{{text_value}}}"
        )

    return text_value


def format_rrb(
    r,
):
    return (
        f"{r:.2f}"
    )


def build_latex_body(
    stats,
):
    """
    Build the body of manuscript Table 5.
    """
    lines = []

    for learner_idx, learner in enumerate(
        LEARNERS
    ):

        if learner_idx > 0:

            lines.append(
                r"\midrule"
            )

            lines.append(
                ""
            )

        learner_rows = stats[
            stats[
                "learner"
            ].eq(
                learner
            )
        ]

        for environment in ENVIRONMENTS:

            row = learner_rows[
                learner_rows[
                    "environment"
                ].eq(
                    environment
                )
            ]

            if len(row) != 1:

                raise RuntimeError(
                    "Expected exactly one row for "
                    f"{learner}, {environment}"
                )

            row = row.iloc[
                0
            ]

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
    # Load original held-out episode-level data
    # --------------------------------------------------------

    print(
        "Loading PPO held-out evaluations..."
    )

    ppo = read_episode_file(
        PPO_HELDOUT,
        learner="PPO",
    )

    print(
        "Loading SAC held-out evaluations..."
    )

    sac = read_episode_file(
        SAC_HELDOUT,
        learner="SAC",
    )

    episodes = pd.concat(
        [
            ppo,
            sac,
        ],
        ignore_index=True,
    )

    episodes = (
        validate_episode_level_data(
            episodes
        )
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
        validate_training_seed_counts(
            seed_level
        )
    )

    print(
        "\nTraining seeds per learner/method/environment:"
    )

    print(
        seed_counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save seed-level data
    # --------------------------------------------------------

    seed_level.to_csv(
        OUTPUT_SEED_LEVEL,
        index=False,
    )

    # --------------------------------------------------------
    # Build all matched CODA-PB2 seed differences
    # --------------------------------------------------------

    paired_frames = []

    for learner in LEARNERS:

        for environment in ENVIRONMENTS:

            paired_frames.append(
                paired_seed_data(
                    seed_level,
                    learner,
                    environment,
                )
            )

    paired_all = pd.concat(
        paired_frames,
        ignore_index=True,
    )

    paired_all.to_csv(
        OUTPUT_PAIRED,
        index=False,
    )

    # --------------------------------------------------------
    # Compute primary statistical family
    # --------------------------------------------------------

    stats = (
        compute_primary_stats(
            paired_all
        )
    )

    stats.to_csv(
        OUTPUT_STATS,
        index=False,
    )

    # --------------------------------------------------------
    # Generate LaTeX table body
    # --------------------------------------------------------

    latex_body = (
        build_latex_body(
            stats
        )
    )

    OUTPUT_LATEX.write_text(
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
        "PRIMARY CODA vs PB2"
    )

    print(
        "=" * 88
    )

    print(
        stats.to_string(
            index=False
        )
    )

    print(
        "\nSciPy version:",
        scipy.__version__,
    )

    print(
        "Wilcoxon: exact, two-sided, "
        "zero_method='wilcox', "
        "correction=False"
    )

    print(
        f"Bootstrap: N={N_BOOT}, "
        "percentile 95% CI, "
        f"seed={BOOT_SEED}"
    )

    print(
        "Holm family: 8 primary "
        "learner-environment comparisons"
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
        f"  Paired differences  : {OUTPUT_PAIRED}"
    )

    print(
        f"  Primary statistics  : {OUTPUT_STATS}"
    )

    print(
        f"  LaTeX table body    : {OUTPUT_LATEX}"
    )

    print(
        "\nDone."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()