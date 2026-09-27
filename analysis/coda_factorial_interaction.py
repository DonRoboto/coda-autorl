#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory 2x2 factorial interaction analysis for CODA.

The four method cells are:

                 O2I = 0           O2I = 1
    I2O = 0      PB2               CODA-O2I
    I2O = 1      CODA-I2O          CODA

For each learner g, environment e, and matched training seed s:

    I_{g,e,s}
      = G_CODA
      - G_CODA-I2O
      - G_CODA-O2I
      + G_PB2

This is a seed-level difference-in-differences contrast on the
held-out-return scale.

Positive I:
    positive departure from additivity.

Negative I:
    negative departure from additivity.

I = 0:
    no seed-level departure from additivity on this scale.

The script:

1. Reads episode-level PPO and SAC held-out evaluation data.
2. Normalizes supported method aliases.
3. Validates the held-out evaluation protocol.
4. Computes one held-out mean return per training seed.
5. Verifies complete matched 2x2 data for every learner/environment.
6. Computes the factorial interaction contrast for each matched seed.
7. Reports:
   - median interaction contrast,
   - unadjusted percentile 95% paired-bootstrap CI,
   - positive/zero/negative seed-level contrast counts,
   - two-sided Wilcoxon signed-rank test,
   - matched-pairs rank-biserial correlation.
8. Applies Holm correction across the eight exploratory
   learner-environment interaction hypotheses.
9. Saves auditable seed-level and summary outputs.
10. Generates the LaTeX table body used by the manuscript/supplement.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── coda_factorial_interaction.py
└── results/
    ├── ppo/
    │   └── heldout_reward_ppo_final/
    │       └── heldout_test_episodes.csv
    └── sac/
        └── heldout_reward_sac_final/
            └── heldout_test_episodes.csv

Outputs
-------

results/analysis/factorial_interaction/
├── factorial_interaction_seed_level.csv
├── factorial_interaction_seed_counts.csv
├── factorial_interaction_summary.csv
└── factorial_interaction_table_body.tex
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
#     <repo_root>/analysis/coda_factorial_interaction.py
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
    / "factorial_interaction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_SEED_LEVEL = (
    OUTPUT_DIR
    / "factorial_interaction_seed_level.csv"
)

OUTPUT_SEED_COUNTS = (
    OUTPUT_DIR
    / "factorial_interaction_seed_counts.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "factorial_interaction_summary.csv"
)

OUTPUT_LATEX = (
    OUTPUT_DIR
    / "factorial_interaction_table_body.tex"
)


# ============================================================
# Analysis configuration
# ============================================================

N_TEST_EPISODES = 100
N_TRAINING_SEEDS = 10

N_BOOT = 10_000
BOOT_SEED = 20260913
ALPHA = 0.05

LEARNERS = [
    "PPO",
    "SAC",
]

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

REQUIRED_METHODS = [
    "PB2",
    "CODA-I2O",
    "CODA-O2I",
    "CODA",
]

METHOD_ALIASES = {
    # PB2
    "PB2": "PB2",

    # Full CODA
    "CODA": "CODA",
    "CODA_FULL": "CODA",
    "CODA-FULL": "CODA",
    "FULL": "CODA",

    # I2O-only
    "CODA-I2O": "CODA-I2O",
    "CODA_I2O": "CODA-I2O",
    "I2O": "CODA-I2O",
    "I2O-ONLY": "CODA-I2O",
    "I2O_ONLY": "CODA-I2O",

    # O2I-only
    "CODA-O2I": "CODA-O2I",
    "CODA_O2I": "CODA-O2I",
    "O2I": "CODA-O2I",
    "O2I-ONLY": "CODA-O2I",
    "O2I_ONLY": "CODA-O2I",
}


# ============================================================
# Input loading
# ============================================================

def canonical_method_name(value):
    """
    Normalize supported historical method aliases.
    """
    key = str(
        value
    ).strip()

    if key in METHOD_ALIASES:
        return METHOD_ALIASES[
            key
        ]

    key_upper = key.upper()

    for alias, canonical in METHOD_ALIASES.items():

        if alias.upper() == key_upper:
            return canonical

    return key


def load_heldout(
    path,
    learner,
):
    """
    Load one episode-level held-out evaluation file and attach
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
        - set(
            df.columns
        )
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
        "method"
    ] = (
        df[
            "method"
        ]
        .map(
            canonical_method_name
        )
    )

    df[
        "training_seed"
    ] = pd.to_numeric(
        df[
            "training_seed"
        ],
        errors="raise",
    ).astype(
        int
    )

    df[
        "test_return"
    ] = pd.to_numeric(
        df[
            "test_return"
        ],
        errors="raise",
    )

    if not np.isfinite(
        df[
            "test_return"
        ]
    ).all():

        raise ValueError(
            f"{path.name} contains non-finite "
            "test_return values."
        )

    # Retain only the 2x2 factorial cells.
    df = df[
        df[
            "method"
        ].isin(
            REQUIRED_METHODS
        )
    ].copy()

    return df


# ============================================================
# Held-out protocol validation
# ============================================================

def validate_episode_level_data(
    episodes,
):
    """
    Validate episode counts, optional common evaluation seeds,
    champion identity, and explore=False.
    """
    observed_methods = set(
        episodes[
            "method"
        ].unique()
    )

    missing_methods = (
        set(
            REQUIRED_METHODS
        )
        - observed_methods
    )

    if missing_methods:

        raise ValueError(
            "Missing factorial methods in held-out data: "
            f"{sorted(missing_methods)}"
        )

    case_keys = [
        "learner",
        "method",
        "environment",
        "training_seed",
    ]

    # --------------------------------------------------------
    # Exactly 100 held-out episodes per case
    # --------------------------------------------------------

    episode_counts = (
        episodes
        .groupby(
            case_keys,
            as_index=False,
        )
        .agg(
            n_test_episodes=(
                "test_return",
                "size",
            )
        )
    )

    bad_episode_counts = episode_counts[
        episode_counts[
            "n_test_episodes"
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

        raise RuntimeError(
            "Expected exactly "
            f"{N_TEST_EPISODES} held-out episodes "
            "per factorial cell and training seed."
        )

    # --------------------------------------------------------
    # Optional validation: 100 unique held-out seeds
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

        bad = unique_test_seeds[
            unique_test_seeds
            != N_TEST_EPISODES
        ]

        if not bad.empty:

            print(
                "\nCases with duplicated or missing test seeds:"
            )

            print(
                bad
            )

            raise RuntimeError(
                "Held-out test-seed validation failed."
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

        bad = champion_counts[
            champion_counts
            != 1
        ]

        if not bad.empty:

            print(
                "\nCases containing more than one champion:"
            )

            print(
                bad
            )

            raise RuntimeError(
                "Champion identity is not unique "
                "within at least one factorial case."
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

        valid = explore_values.isin(
            [
                "false",
                "0",
            ]
        )

        if not valid.all():

            raise RuntimeError(
                "At least one held-out episode was "
                "evaluated with explore=True."
            )


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


# ============================================================
# Experimental-design validation
# ============================================================

def validate_seed_counts(
    seed_level,
):
    """
    Require exactly ten training seeds for every expected
    learner/method/environment factorial cell.
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

    expected = pd.DataFrame(
        [
            (
                learner,
                method,
                environment,
            )
            for learner in LEARNERS
            for method in REQUIRED_METHODS
            for environment in ENVIRONMENTS
        ],
        columns=[
            "learner",
            "method",
            "environment",
        ],
    )

    checked = expected.merge(
        seed_counts,
        on=[
            "learner",
            "method",
            "environment",
        ],
        how="left",
        validate="one_to_one",
    )

    checked[
        "n_training_seeds"
    ] = (
        checked[
            "n_training_seeds"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    bad = checked[
        checked[
            "n_training_seeds"
        ]
        != N_TRAINING_SEEDS
    ]

    if not bad.empty:

        print(
            "\nIncomplete factorial cells:"
        )

        print(
            bad.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Expected exactly "
            f"{N_TRAINING_SEEDS} training seeds "
            "for every factorial cell."
        )

    return checked


def validate_common_test_seed_sets(
    episodes,
    seed_level,
):
    """
    If `test_seed` is available, verify identical held-out test-seed
    sets across PB2, CODA-I2O, CODA-O2I, and CODA within every
    learner/environment/training-seed block.
    """
    if (
        "test_seed"
        not in episodes.columns
    ):
        return

    for learner in LEARNERS:

        for environment in ENVIRONMENTS:

            training_seeds = sorted(
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
                ][
                    "training_seed"
                ]
                .unique()
            )

            for training_seed in training_seeds:

                reference = None

                for method in REQUIRED_METHODS:

                    current = set(
                        episodes[
                            episodes[
                                "learner"
                            ].eq(
                                learner
                            )
                            &
                            episodes[
                                "environment"
                            ].eq(
                                environment
                            )
                            &
                            episodes[
                                "training_seed"
                            ].eq(
                                training_seed
                            )
                            &
                            episodes[
                                "method"
                            ].eq(
                                method
                            )
                        ][
                            "test_seed"
                        ]
                        .astype(
                            int
                        )
                    )

                    if reference is None:

                        reference = current

                    elif current != reference:

                        raise RuntimeError(
                            "Common held-out test seeds do not match for:\n"
                            f"{learner}, "
                            f"{environment}, "
                            f"training_seed={training_seed}"
                        )


# ============================================================
# Build matched 2x2 seed-level interaction table
# ============================================================

def method_table(
    seed_level,
    learner,
    environment,
    method,
):
    """
    Return one seed-level held-out column for a factorial cell.
    """
    out = (
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
                method
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
                    method
            }
        )
        .copy()
    )

    duplicates = (
        out[
            "training_seed"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicates.any():

        raise RuntimeError(
            "Duplicate seed-level rows for "
            f"{learner} / {environment} / {method}:\n"
            f"{out.loc[duplicates].to_string(index=False)}"
        )

    return out


def factorial_interaction_table(
    seed_level,
    learner,
    environment,
):
    """
    Build the matched 2x2 table and compute the method-level
    factorial interaction contrast.

    I = CODA - CODA-I2O - CODA-O2I + PB2
    """
    tables = [
        method_table(
            seed_level,
            learner,
            environment,
            method,
        )
        for method in REQUIRED_METHODS
    ]

    merged = tables[
        0
    ]

    for table in tables[
        1:
    ]:

        merged = merged.merge(
            table,
            on="training_seed",
            how="inner",
            validate="one_to_one",
        )

    merged = (
        merged
        .sort_values(
            "training_seed"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            merged
        )
        != N_TRAINING_SEEDS
    ):

        raise RuntimeError(
            f"{learner} / {environment}: expected "
            f"{N_TRAINING_SEEDS} fully matched seeds, "
            f"got {len(merged)}."
        )

    merged[
        "interaction"
    ] = (
        merged[
            "CODA"
        ]
        - merged[
            "CODA-I2O"
        ]
        - merged[
            "CODA-O2I"
        ]
        + merged[
            "PB2"
        ]
    )

    # Conditional channel-addition decompositions retained for audit.
    merged[
        "I2O_when_O2I_off"
    ] = (
        merged[
            "CODA-I2O"
        ]
        - merged[
            "PB2"
        ]
    )

    merged[
        "I2O_when_O2I_on"
    ] = (
        merged[
            "CODA"
        ]
        - merged[
            "CODA-O2I"
        ]
    )

    merged[
        "O2I_when_I2O_off"
    ] = (
        merged[
            "CODA-O2I"
        ]
        - merged[
            "PB2"
        ]
    )

    merged[
        "O2I_when_I2O_on"
    ] = (
        merged[
            "CODA"
        ]
        - merged[
            "CODA-I2O"
        ]
    )

    merged.insert(
        0,
        "environment",
        environment,
    )

    merged.insert(
        0,
        "learner",
        learner,
    )

    return merged


# ============================================================
# Statistical helpers
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
        np.isfinite(
            d
        )
    ]

    d = d[
        d != 0
    ]

    if len(
        d
    ) == 0:

        return 0.0

    ranks = rankdata(
        np.abs(
            d
        ),
        method="average",
    )

    w_plus = float(
        ranks[
            d > 0
        ].sum()
    )

    w_minus = float(
        ranks[
            d < 0
        ].sum()
    )

    denominator = (
        w_plus
        + w_minus
    )

    if denominator == 0:

        return 0.0

    return float(
        (
            w_plus
            - w_minus
        )
        / denominator
    )


def wilcoxon_against_zero(
    values,
):
    """
    Two-sided Wilcoxon signed-rank test against zero.

    Exact calculation is used when no zero differences occur.
    If zeros occur, SciPy's asymptotic implementation is used
    with zero_method='wilcox'.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:

        return (
            np.nan,
            np.nan,
            "not_available",
        )

    if np.all(
        values
        == 0
    ):

        return (
            0.0,
            1.0,
            "all_zero",
        )

    has_zero = bool(
        np.any(
            values
            == 0
        )
    )

    method = (
        "approx"
        if has_zero
        else "exact"
    )

    test = wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method=method,
    )

    return (
        float(
            test.statistic
        ),
        float(
            test.pvalue
        ),
        method,
    )


# Same seed-resampling pattern for all eight interaction comparisons.
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
    values,
):
    """
    Percentile paired-bootstrap CI for the median seed-level
    interaction contrast.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    if (
        len(
            values
        )
        != N_TRAINING_SEEDS
    ):

        raise ValueError(
            "Expected exactly "
            f"{N_TRAINING_SEEDS} matched interaction values."
        )

    bootstrap_medians = np.median(
        values[
            boot_idx
        ],
        axis=1,
    )

    ci_low, ci_high = np.quantile(
        bootstrap_medians,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(
            np.median(
                values
            )
        ),
        float(
            ci_low
        ),
        float(
            ci_high
        ),
    )


# ============================================================
# Factorial statistical analysis
# ============================================================

def compute_factorial_analysis(
    seed_level,
):
    """
    Compute all eight learner-environment factorial interaction
    contrasts and their exploratory inferential summaries.
    """
    seed_tables = []
    summary_rows = []

    for learner in LEARNERS:

        for environment in ENVIRONMENTS:

            matched = factorial_interaction_table(
                seed_level,
                learner,
                environment,
            )

            seed_tables.append(
                matched
            )

            interaction = matched[
                "interaction"
            ].to_numpy(
                dtype=float
            )

            (
                median_interaction,
                ci_low,
                ci_high,
            ) = bootstrap_median_ci(
                interaction
            )

            n_positive = int(
                np.sum(
                    interaction > 0
                )
            )

            n_zero = int(
                np.sum(
                    interaction == 0
                )
            )

            n_negative = int(
                np.sum(
                    interaction < 0
                )
            )

            (
                wilcoxon_statistic,
                p_raw,
                wilcoxon_method,
            ) = wilcoxon_against_zero(
                interaction
            )

            r_rb = rank_biserial(
                interaction
            )

            summary_rows.append(
                {
                    "learner":
                        learner,

                    "environment":
                        environment,

                    "n_matched_seeds":
                        len(
                            interaction
                        ),

                    "median_interaction":
                        median_interaction,

                    "bootstrap_95_lo":
                        ci_low,

                    "bootstrap_95_hi":
                        ci_high,

                    "n_positive":
                        n_positive,

                    "n_zero":
                        n_zero,

                    "n_negative":
                        n_negative,

                    "P_Z_N":
                        (
                            f"{n_positive}/"
                            f"{n_zero}/"
                            f"{n_negative}"
                        ),

                    "wilcoxon_statistic":
                        wilcoxon_statistic,

                    "wilcoxon_method":
                        wilcoxon_method,

                    "p_raw":
                        p_raw,

                    "r_rb":
                        r_rb,
                }
            )

    seed_level_interaction = pd.concat(
        seed_tables,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    # Holm correction across the eight exploratory interaction hypotheses.
    summary[
        "p_Holm"
    ] = multipletests(
        summary[
            "p_raw"
        ].to_numpy(
            dtype=float
        ),
        alpha=ALPHA,
        method="holm",
    )[1]

    summary[
        "significant_Holm"
    ] = (
        summary[
            "p_Holm"
        ]
        < ALPHA
    )

    # Preserve manuscript ordering.
    learner_order = {
        "PPO": 0,
        "SAC": 1,
    }

    environment_order = {
        environment: index
        for index, environment
        in enumerate(
            ENVIRONMENTS
        )
    }

    summary[
        "_learner_order"
    ] = (
        summary[
            "learner"
        ]
        .map(
            learner_order
        )
    )

    summary[
        "_environment_order"
    ] = (
        summary[
            "environment"
        ]
        .map(
            environment_order
        )
    )

    summary = (
        summary
        .sort_values(
            [
                "_learner_order",
                "_environment_order",
            ]
        )
        .drop(
            columns=[
                "_learner_order",
                "_environment_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    seed_level_interaction[
        "_learner_order"
    ] = (
        seed_level_interaction[
            "learner"
        ]
        .map(
            learner_order
        )
    )

    seed_level_interaction[
        "_environment_order"
    ] = (
        seed_level_interaction[
            "environment"
        ]
        .map(
            environment_order
        )
    )

    seed_level_interaction = (
        seed_level_interaction
        .sort_values(
            [
                "_learner_order",
                "_environment_order",
                "training_seed",
            ]
        )
        .drop(
            columns=[
                "_learner_order",
                "_environment_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        seed_level_interaction,
        summary,
    )


# ============================================================
# LaTeX formatting
# ============================================================

def format_interaction(
    row,
):
    """
    Format median interaction and 95% CI.
    """
    return (
        f"${row['median_interaction']:.2f} "
        f"[{row['bootstrap_95_lo']:.2f}, "
        f"{row['bootstrap_95_hi']:.2f}]$"
    )


def format_p(
    value,
):
    return (
        f"{value:.3f}"
    )


def format_rrb(
    value,
):
    return (
        f"{value:.2f}"
    )


def build_latex_body(
    summary,
):
    """
    Build a compact LaTeX table body.

    P/Z/N denotes the number of positive/zero/negative
    seed-level interaction contrasts.
    """
    lines = []

    previous_learner = None

    for _, row in (
        summary
        .iterrows()
    ):

        learner = row[
            "learner"
        ]

        if (
            previous_learner is not None
            and learner
            != previous_learner
        ):

            lines.append(
                r"\midrule"
            )

            lines.append(
                ""
            )

        latex_row = (
            f"{learner} & "
            f"{row['environment']} &\n"
            f"{format_interaction(row)} &\n"
            f"{row['P_Z_N']} & "
            f"{format_p(row['p_raw'])} & "
            f"{format_p(row['p_Holm'])} & "
            f"{format_rrb(row['r_rb'])} "
            r"\\"
        )

        lines.append(
            latex_row
        )

        lines.append(
            ""
        )

        previous_learner = (
            learner
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
    # Load episode-level held-out data
    # --------------------------------------------------------

    print(
        "Loading PPO held-out evaluations..."
    )

    ppo = load_heldout(
        PPO_HELDOUT,
        learner="PPO",
    )

    print(
        "Loading SAC held-out evaluations..."
    )

    sac = load_heldout(
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

    # --------------------------------------------------------
    # Validate held-out evaluation protocol
    # --------------------------------------------------------

    validate_episode_level_data(
        episodes
    )

    # --------------------------------------------------------
    # Compute seed-level held-out means
    # --------------------------------------------------------

    seed_level = (
        compute_seed_level_results(
            episodes
        )
    )

    seed_counts = (
        validate_seed_counts(
            seed_level
        )
    )

    validate_common_test_seed_sets(
        episodes,
        seed_level,
    )

    print(
        "\nTraining seeds per factorial cell:"
    )

    print(
        seed_counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Compute factorial interaction
    # --------------------------------------------------------

    (
        seed_level_interaction,
        summary,
    ) = compute_factorial_analysis(
        seed_level
    )

    # --------------------------------------------------------
    # Save numerical artifacts
    # --------------------------------------------------------

    seed_level_interaction.to_csv(
        OUTPUT_SEED_LEVEL,
        index=False,
    )

    seed_counts.to_csv(
        OUTPUT_SEED_COUNTS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # Generate LaTeX table body
    # --------------------------------------------------------

    latex_body = build_latex_body(
        summary
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
        + "=" * 100
    )

    print(
        "CODA I2O x O2I FACTORIAL INTERACTION"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "learner",
        "environment",
        "n_matched_seeds",
        "median_interaction",
        "bootstrap_95_lo",
        "bootstrap_95_hi",
        "P_Z_N",
        "p_raw",
        "p_Holm",
        "r_rb",
        "significant_Holm",
    ]

    print(
        summary[
            display_columns
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nSciPy version:",
        scipy.__version__,
    )

    print(
        "Bootstrap: "
        f"N={N_BOOT}, percentile 95% CI, seed={BOOT_SEED}"
    )

    print(
        "Wilcoxon: two-sided, zero_method='wilcox', "
        "correction=False; exact when no zero interaction "
        "contrasts are present, approx otherwise"
    )

    print(
        "Holm family: 8 exploratory "
        "learner-environment interaction hypotheses"
    )

    print(
        "P/Z/N: positive / zero / negative "
        "seed-level interaction contrasts"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LATEX TABLE BODY"
    )

    print(
        "=" * 100
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
        f"  Seed-level interaction : {OUTPUT_SEED_LEVEL}"
    )

    print(
        f"  Seed counts            : {OUTPUT_SEED_COUNTS}"
    )

    print(
        f"  Summary                : {OUTPUT_SUMMARY}"
    )

    print(
        f"  LaTeX table body       : {OUTPUT_LATEX}"
    )

    print(
        "\nDone."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()