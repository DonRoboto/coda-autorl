#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table 6: Fixed-support terminal training performance (tAUC100k).

For each learner (PPO/SAC), method (CODA/PB2), environment, and training
seed, this script:

1. Reads the held-out evaluation records to identify the
   training-selected champion for each seed.
2. Reads the corresponding archived training metrics.
3. Recovers the champion's final monotone execution segment after the
   last progress-counter rollback.
4. Computes fixed-support terminal training performance over the final
   100,000 environment interactions:

       tAUC100k = (1 / W) * integral R(T) dT,

   with W = 100,000 interactions.

5. Aggregates the ten independent training seeds using median [Q1, Q3].
6. Forms matched CODA - PB2 seed-level differences.
7. Reports:
   - median paired difference,
   - win/tie/loss counts,
   - two-sided Wilcoxon signed-rank test,
   - Holm-adjusted p-value across the eight terminal-training comparisons.
8. Saves all intermediate and final numerical artifacts.
9. Generates the LaTeX table body used in the manuscript.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── Tab06_terminal_training.py
└── results/
    ├── ppo/
    │   ├── metrics.zip
    │   └── heldout_reward_ppo_final/
    │       └── heldout_test_episodes.csv
    └── sac/
        ├── metrics.zip
        └── heldout_reward_sac_final/
            └── heldout_test_episodes.csv

Outputs
-------

results/analysis/terminal_training/
├── terminal_tauc_seed_level.csv
├── terminal_tauc_seed_counts.csv
├── terminal_tauc_marginal_summary.csv
├── terminal_tauc_paired_differences.csv
├── terminal_tauc_primary_stats.csv
└── terminal_tauc_table_body.tex
"""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# Paths
# ============================================================

# Works when this file is stored under:
#     <repo_root>/analysis/Tab06_terminal_training.py
#
# The fallback also allows execution from an interactive session.
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent

PPO_TRAIN_ZIP = (
    REPO_ROOT
    / "results"
    / "ppo"
    / "ppo_train.zip"
)

SAC_TRAIN_ZIP = (
    REPO_ROOT
    / "results"
    / "sac"
    / "sac_train.zip"
)

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
    / "terminal_training"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_SEED_LEVEL = (
    OUTPUT_DIR
    / "terminal_tauc_seed_level.csv"
)

OUTPUT_SEED_COUNTS = (
    OUTPUT_DIR
    / "terminal_tauc_seed_counts.csv"
)

OUTPUT_MARGINAL = (
    OUTPUT_DIR
    / "terminal_tauc_marginal_summary.csv"
)

OUTPUT_PAIRED = (
    OUTPUT_DIR
    / "terminal_tauc_paired_differences.csv"
)

OUTPUT_STATS = (
    OUTPUT_DIR
    / "terminal_tauc_primary_stats.csv"
)

OUTPUT_LATEX_BODY = (
    OUTPUT_DIR
    / "terminal_tauc_table_body.tex"
)


# ============================================================
# Analysis configuration
# ============================================================

TAUC_WINDOW = 100_000
N_TRAINING_SEEDS = 10
N_TEST_EPISODES = 100

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

METHODS = [
    "CODA",
    "PB2",
]


# ============================================================
# Column names used in training logs
# ============================================================

AGENT_COL = "agente_id"
STEP_COL = "timesteps_total"
RETURN_COL = "env_runners/episode_return_mean"

# If causal_order exists, it is used to preserve logging order.
CAUSAL_ORDER_COL = "causal_order"


# ============================================================
# Held-out champion loading and validation
# ============================================================

def load_heldout(
    path,
    learner,
):
    """
    Load held-out evaluation data and retain CODA/PB2 only.
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
        "champion_agent",
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

    if not np.isfinite(
        df[
            "test_return"
        ]
    ).all():

        raise ValueError(
            f"Non-finite held-out returns detected in {path.name}."
        )

    # Table 6 uses CODA and PB2 only.
    df = df[
        df[
            "method"
        ].isin(
            METHODS
        )
    ].copy()

    return df


def validate_heldout_cases(
    heldout,
):
    """
    Validate episode count and champion identity for every
    learner/method/environment/training-seed case.
    """
    case_keys = [
        "learner",
        "method",
        "environment",
        "training_seed",
    ]

    episode_counts = (
        heldout
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

        raise ValueError(
            "Expected exactly "
            f"{N_TEST_EPISODES} held-out episodes "
            "per training seed."
        )

    champion_counts = (
        heldout
        .groupby(
            case_keys
        )[
            "champion_agent"
        ]
        .nunique()
    )

    bad_champions = champion_counts[
        champion_counts != 1
    ]

    if not bad_champions.empty:

        print(
            "\nCases with non-unique champion identity:"
        )

        print(
            bad_champions
        )

        raise ValueError(
            "At least one learner/method/environment/seed "
            "contains more than one champion."
        )

    # Optional validation of common evaluation seeds.
    if (
        "test_seed"
        in heldout.columns
    ):

        test_seed_counts = (
            heldout
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
                "At least one held-out case does not contain "
                f"{N_TEST_EPISODES} unique test seeds."
            )

    # Optional validation of explore=False.
    if (
        "explore"
        in heldout.columns
    ):

        explore_values = (
            heldout[
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


def build_champion_table(
    heldout,
):
    """
    Extract one champion identity per
    learner/method/environment/training seed.
    """
    champions = (
        heldout[
            [
                "learner",
                "method",
                "environment",
                "training_seed",
                "champion_agent",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "learner",
                "method",
                "environment",
                "training_seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return champions


# ============================================================
# Training ZIP utilities
# ============================================================

def validate_zip_path(
    zip_path,
):
    """
    Validate one archived training-metrics ZIP.
    """
    zip_path = Path(
        zip_path
    )

    if not zip_path.exists():
        raise FileNotFoundError(
            "Training metrics archive not found:\n"
            f"{zip_path}"
        )

    if not zipfile.is_zipfile(
        zip_path
    ):
        raise ValueError(
            f"Not a valid ZIP archive: {zip_path}"
        )


def expected_training_member(
    learner,
    method,
    environment,
    seed,
):
    """
    Return the expected archived CSV path for one training run.
    """
    if method == "CODA":

        return (
            f"metrics/{environment}/"
            f"metrics_CODA_FULL_seed{seed}.csv"
        )

    if method == "PB2":

        return (
            f"metrics/{environment}/"
            f"metrics_PB2_{learner}_HPO_seed{seed}.csv"
        )

    raise ValueError(
        f"Unsupported method: {method}"
    )


def find_training_member(
    zf,
    learner,
    method,
    environment,
    seed,
):
    """
    Locate one training metrics CSV inside the ZIP.

    The exact expected path is tried first. If the archive uses
    an additional top-level directory, the function falls back
    to a unique basename match.
    """
    expected = expected_training_member(
        learner,
        method,
        environment,
        seed,
    )

    names = zf.namelist()

    if expected in names:
        return expected

    expected_basename = Path(
        expected
    ).name

    candidates = [
        name
        for name in names
        if Path(
            name
        ).name
        == expected_basename
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "\nCould not uniquely locate training CSV.\n"
        f"Learner: {learner}\n"
        f"Method: {method}\n"
        f"Environment: {environment}\n"
        f"Seed: {seed}\n"
        f"Expected: {expected}\n"
        f"Candidates: {candidates}"
    )


def read_training_csv(
    zf,
    member,
):
    """
    Read one training metrics CSV directly from the archive.
    """
    with zf.open(
        member
    ) as file:
        return pd.read_csv(
            file
        )


# ============================================================
# Recover final execution segment for selected champion
# ============================================================

def get_final_training_segment(
    df,
    champion_agent,
):
    """
    Recover the champion's final monotone training segment after
    the last progress-counter rollback.

    Duplicate progress counters retain the latest causal report.
    """
    required = {
        AGENT_COL,
        STEP_COL,
        RETURN_COL,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Training CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    current = df[
        df[
            AGENT_COL
        ].astype(str)
        ==
        str(
            champion_agent
        )
    ].copy()

    if current.empty:
        raise ValueError(
            f"Champion '{champion_agent}' not found "
            "in training CSV."
        )

    # Preserve raw row order as a fallback.
    current[
        "_row_order"
    ] = np.arange(
        len(current)
    )

    if (
        CAUSAL_ORDER_COL
        in current.columns
    ):

        current[
            CAUSAL_ORDER_COL
        ] = pd.to_numeric(
            current[
                CAUSAL_ORDER_COL
            ],
            errors="coerce",
        )

        current = current.sort_values(
            CAUSAL_ORDER_COL,
            kind="stable",
        )

    else:

        current = current.sort_values(
            "_row_order",
            kind="stable",
        )

    current[
        STEP_COL
    ] = pd.to_numeric(
        current[
            STEP_COL
        ],
        errors="coerce",
    )

    current[
        RETURN_COL
    ] = pd.to_numeric(
        current[
            RETURN_COL
        ],
        errors="coerce",
    )

    current = current.dropna(
        subset=[
            STEP_COL,
            RETURN_COL,
        ]
    ).copy()

    if len(current) < 2:
        raise ValueError(
            "Not enough finite observations for champion."
        )

    # A new segment starts whenever timesteps_total decreases.
    steps_in_report_order = current[
        STEP_COL
    ].to_numpy(
        dtype=float
    )

    rollback = np.r_[
        False,
        np.diff(
            steps_in_report_order
        )
        < 0,
    ]

    current[
        "_segment"
    ] = np.cumsum(
        rollback
    )

    # The terminal segment begins after the final rollback.
    final_segment_id = current[
        "_segment"
    ].max()

    segment = current[
        current[
            "_segment"
        ]
        == final_segment_id
    ].copy()

    # Retain the latest causal report for duplicate progress values.
    if (
        CAUSAL_ORDER_COL
        in segment.columns
    ):

        segment = segment.sort_values(
            CAUSAL_ORDER_COL,
            kind="stable",
        )

    else:

        segment = segment.sort_values(
            "_row_order",
            kind="stable",
        )

    segment = (
        segment
        .drop_duplicates(
            subset=[
                STEP_COL
            ],
            keep="last",
        )
        .sort_values(
            STEP_COL
        )
        .reset_index(
            drop=True
        )
    )

    if len(segment) < 2:
        raise ValueError(
            "Final segment contains fewer than two "
            "unique progress values."
        )

    return segment


# ============================================================
# Compute tAUC100k
# ============================================================

def compute_terminal_tauc(
    segment,
    window=TAUC_WINDOW,
):
    """
    Compute fixed-support terminal training performance over the
    final `window` interactions.

    Linear interpolation is used only to evaluate the exact window
    boundaries. Complete support over the full window is required.
    """
    x = segment[
        STEP_COL
    ].to_numpy(
        dtype=float
    )

    y = segment[
        RETURN_COL
    ].to_numpy(
        dtype=float
    )

    if not (
        np.isfinite(
            x
        ).all()
        and
        np.isfinite(
            y
        ).all()
    ):

        raise ValueError(
            "Non-finite values in terminal segment."
        )

    if np.any(
        np.diff(
            x
        )
        <= 0
    ):

        raise ValueError(
            "Progress values must be strictly increasing "
            "after duplicate removal."
        )

    t_end = float(
        x[-1]
    )

    t_start = (
        t_end
        - window
    )

    # Complete support over the final fixed window is required.
    if (
        x[0]
        > t_start
    ):

        raise ValueError(
            "Incomplete tAUC support: "
            f"segment starts at {x[0]:.0f}, "
            f"required <= {t_start:.0f}"
        )

    # Interpolate exact boundaries.
    y_start = np.interp(
        t_start,
        x,
        y,
    )

    y_end = np.interp(
        t_end,
        x,
        y,
    )

    mask = (
        (x > t_start)
        & (x < t_end)
    )

    x_window = np.concatenate(
        [
            [t_start],
            x[
                mask
            ],
            [t_end],
        ]
    )

    y_window = np.concatenate(
        [
            [y_start],
            y[
                mask
            ],
            [y_end],
        ]
    )

    # NumPy compatibility.
    if hasattr(
        np,
        "trapezoid",
    ):

        area = np.trapezoid(
            y_window,
            x_window,
        )

    else:

        area = np.trapz(
            y_window,
            x_window,
        )

    tauc = (
        area
        / window
    )

    return {
        "tAUC100k":
            float(
                tauc
            ),

        "window_start":
            float(
                t_start
            ),

        "window_end":
            float(
                t_end
            ),

        "segment_start":
            float(
                x[0]
            ),

        "segment_end":
            float(
                x[-1]
            ),

        "n_points_window":
            int(
                len(
                    x_window
                )
            ),
    }


# ============================================================
# Compute tAUC for every selected champion
# ============================================================

def compute_from_zip(
    zip_path,
    learner,
    champion_table,
):
    """
    Compute terminal tAUC for all CODA/PB2 champions of one learner.
    """
    validate_zip_path(
        zip_path
    )

    rows = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        learner_champions = champion_table[
            champion_table[
                "learner"
            ].eq(
                learner
            )
        ]

        for _, champion in (
            learner_champions
            .iterrows()
        ):

            method = champion[
                "method"
            ]

            environment = champion[
                "environment"
            ]

            seed = int(
                champion[
                    "training_seed"
                ]
            )

            champion_agent = champion[
                "champion_agent"
            ]

            member = find_training_member(
                zf,
                learner,
                method,
                environment,
                seed,
            )

            print(
                f"{learner} | "
                f"{method} | "
                f"{environment} | "
                f"seed={seed} | "
                f"champion={champion_agent}"
            )

            train_df = read_training_csv(
                zf,
                member,
            )

            final_segment = (
                get_final_training_segment(
                    train_df,
                    champion_agent,
                )
            )

            result = (
                compute_terminal_tauc(
                    final_segment
                )
            )

            rows.append(
                {
                    "learner":
                        learner,

                    "method":
                        method,

                    "environment":
                        environment,

                    "training_seed":
                        seed,

                    "champion_agent":
                        champion_agent,

                    "training_member":
                        member,

                    **result,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Design validation and summaries
# ============================================================

def build_seed_counts(
    terminal,
):
    """
    Count valid tAUC values per learner/method/environment.
    """
    seed_counts = (
        terminal
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

    return seed_counts


def validate_terminal_design(
    seed_counts,
):
    """
    Require exactly ten valid tAUC values for every expected cell.
    """
    expected = pd.DataFrame(
        [
            (
                learner,
                method,
                environment,
            )
            for learner in LEARNERS
            for method in METHODS
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
        .fillna(0)
        .astype(int)
    )

    bad = checked[
        checked[
            "n_training_seeds"
        ]
        != N_TRAINING_SEEDS
    ]

    if not bad.empty:

        print(
            "\nIncomplete tAUC cells:"
        )

        print(
            bad.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Expected exactly "
            f"{N_TRAINING_SEEDS} valid tAUC values "
            "for every learner/method/environment."
        )

    return checked


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


def compute_marginal_summary(
    terminal,
):
    """
    Compute marginal median [Q1, Q3] of tAUC100k across seeds.
    """
    marginal = (
        terminal
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
                "tAUC100k",
                "median",
            ),
            q1=(
                "tAUC100k",
                q1,
            ),
            q3=(
                "tAUC100k",
                q3,
            ),
        )
    )

    return marginal


# ============================================================
# Matched CODA-PB2 seed differences
# ============================================================

def build_paired_differences(
    terminal,
):
    """
    Build matched CODA-PB2 seed-level tAUC differences.
    """
    paired_frames = []

    for learner in LEARNERS:

        for environment in ENVIRONMENTS:

            coda = (
                terminal[
                    terminal[
                        "learner"
                    ].eq(
                        learner
                    )
                    &
                    terminal[
                        "environment"
                    ].eq(
                        environment
                    )
                    &
                    terminal[
                        "method"
                    ].eq(
                        "CODA"
                    )
                ][
                    [
                        "training_seed",
                        "tAUC100k",
                    ]
                ]
                .rename(
                    columns={
                        "tAUC100k":
                            "CODA"
                    }
                )
            )

            pb2 = (
                terminal[
                    terminal[
                        "learner"
                    ].eq(
                        learner
                    )
                    &
                    terminal[
                        "environment"
                    ].eq(
                        environment
                    )
                    &
                    terminal[
                        "method"
                    ].eq(
                        "PB2"
                    )
                ][
                    [
                        "training_seed",
                        "tAUC100k",
                    ]
                ]
                .rename(
                    columns={
                        "tAUC100k":
                            "PB2"
                    }
                )
            )

            paired = (
                coda
                .merge(
                    pb2,
                    on="training_seed",
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
                len(
                    paired
                )
                != N_TRAINING_SEEDS
            ):

                raise RuntimeError(
                    f"{learner}, {environment}: "
                    f"expected {N_TRAINING_SEEDS} paired seeds, "
                    f"found {len(paired)}"
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

            paired_frames.append(
                paired[
                    [
                        "learner",
                        "environment",
                        "training_seed",
                        "CODA",
                        "PB2",
                        "difference",
                    ]
                ]
            )

    return pd.concat(
        paired_frames,
        ignore_index=True,
    )


# ============================================================
# Primary terminal-training statistics
# ============================================================

def compute_terminal_stats(
    paired_all,
    marginal,
):
    """
    Compute the eight CODA-vs-PB2 terminal-training comparisons.
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

            # Exact Wilcoxon is used when there are no zero
            # differences. With zeros and zero_method='wilcox',
            # SciPy requires the asymptotic implementation.
            if ties == 0:

                test = wilcoxon(
                    d,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="exact",
                )

                wilcoxon_method = (
                    "exact"
                )

            else:

                test = wilcoxon(
                    d,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="approx",
                )

                wilcoxon_method = (
                    "approx"
                )

            coda_marginal = marginal[
                marginal[
                    "learner"
                ].eq(
                    learner
                )
                &
                marginal[
                    "environment"
                ].eq(
                    environment
                )
                &
                marginal[
                    "method"
                ].eq(
                    "CODA"
                )
            ]

            pb2_marginal = marginal[
                marginal[
                    "learner"
                ].eq(
                    learner
                )
                &
                marginal[
                    "environment"
                ].eq(
                    environment
                )
                &
                marginal[
                    "method"
                ].eq(
                    "PB2"
                )
            ]

            if (
                len(
                    coda_marginal
                )
                != 1
                or
                len(
                    pb2_marginal
                )
                != 1
            ):

                raise RuntimeError(
                    "Expected exactly one marginal summary row for "
                    f"{learner}, {environment}, CODA/PB2."
                )

            coda_marginal = (
                coda_marginal
                .iloc[0]
            )

            pb2_marginal = (
                pb2_marginal
                .iloc[0]
            )

            rows.append(
                {
                    "learner":
                        learner,

                    "environment":
                        environment,

                    "n_matched_seeds":
                        len(
                            d
                        ),

                    "coda_median":
                        coda_marginal[
                            "median"
                        ],

                    "coda_q1":
                        coda_marginal[
                            "q1"
                        ],

                    "coda_q3":
                        coda_marginal[
                            "q3"
                        ],

                    "pb2_median":
                        pb2_marginal[
                            "median"
                        ],

                    "pb2_q1":
                        pb2_marginal[
                            "q1"
                        ],

                    "pb2_q3":
                        pb2_marginal[
                            "q3"
                        ],

                    # Median of matched differences, not
                    # difference between marginal medians.
                    "median_difference":
                        float(
                            np.median(
                                d
                            )
                        ),

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

                    "wilcoxon_W":
                        float(
                            test.statistic
                        ),

                    "wilcoxon_method":
                        wilcoxon_method,

                    "p_raw":
                        float(
                            test.pvalue
                        ),
                }
            )

    stats = pd.DataFrame(
        rows
    )

    # Separate Holm family of eight terminal-training comparisons.
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

def fmt_summary(
    median,
    q1_value,
    q3_value,
):
    return (
        f"{median:.1f} "
        f"[{q1_value:.1f}, "
        f"{q3_value:.1f}]"
    )


def fmt_delta(
    value,
):
    return (
        f"{value:.1f}"
    )


def fmt_p(
    value,
):
    return (
        f"{value:.3f}"
    )


def fmt_wtl(
    row,
):
    return str(
        row[
            "W_T_L"
        ]
    )


def build_latex_body(
    stats,
):
    """
    Build the body of manuscript Table 6.
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

        for environment in ENVIRONMENTS:

            row = stats[
                stats[
                    "learner"
                ].eq(
                    learner
                )
                &
                stats[
                    "environment"
                ].eq(
                    environment
                )
            ]

            if len(
                row
            ) != 1:

                raise RuntimeError(
                    "Expected exactly one result for "
                    f"{learner}, {environment}"
                )

            row = row.iloc[
                0
            ]

            coda_text = (
                fmt_summary(
                    row[
                        "coda_median"
                    ],
                    row[
                        "coda_q1"
                    ],
                    row[
                        "coda_q3"
                    ],
                )
            )

            pb2_text = (
                fmt_summary(
                    row[
                        "pb2_median"
                    ],
                    row[
                        "pb2_q1"
                    ],
                    row[
                        "pb2_q3"
                    ],
                )
            )

            latex_row = (
                f"{learner} & "
                f"{environment} &\n"
                f"{coda_text} &\n"
                f"{pb2_text} &\n"
                f"{fmt_delta(row['median_difference'])} & "
                f"{fmt_wtl(row)} & "
                f"{fmt_p(row['p_raw'])} & "
                f"{fmt_p(row['p_Holm'])} "
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
    # Load held-out evaluation records and champions
    # --------------------------------------------------------

    print(
        "Loading PPO held-out evaluations..."
    )

    ppo_heldout = (
        load_heldout(
            PPO_HELDOUT,
            learner="PPO",
        )
    )

    print(
        "Loading SAC held-out evaluations..."
    )

    sac_heldout = (
        load_heldout(
            SAC_HELDOUT,
            learner="SAC",
        )
    )

    heldout = pd.concat(
        [
            ppo_heldout,
            sac_heldout,
        ],
        ignore_index=True,
    )

    validate_heldout_cases(
        heldout
    )

    champions = (
        build_champion_table(
            heldout
        )
    )

    # --------------------------------------------------------
    # Compute tAUC100k for every selected champion
    # --------------------------------------------------------

    print(
        "\nComputing PPO terminal tAUC..."
    )

    ppo_tauc = compute_from_zip(
        PPO_TRAIN_ZIP,
        learner="PPO",
        champion_table=champions,
    )

    print(
        "\nComputing SAC terminal tAUC..."
    )

    sac_tauc = compute_from_zip(
        SAC_TRAIN_ZIP,
        learner="SAC",
        champion_table=champions,
    )

    terminal = pd.concat(
        [
            ppo_tauc,
            sac_tauc,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validate complete terminal-training design
    # --------------------------------------------------------

    seed_counts = (
        build_seed_counts(
            terminal
        )
    )

    seed_counts = (
        validate_terminal_design(
            seed_counts
        )
    )

    print(
        "\nTraining seeds with valid tAUC:"
    )

    print(
        seed_counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save seed-level terminal-training outcomes
    # --------------------------------------------------------

    terminal.to_csv(
        OUTPUT_SEED_LEVEL,
        index=False,
    )

    seed_counts.to_csv(
        OUTPUT_SEED_COUNTS,
        index=False,
    )

    # --------------------------------------------------------
    # Marginal median [Q1, Q3]
    # --------------------------------------------------------

    marginal = (
        compute_marginal_summary(
            terminal
        )
    )

    marginal.to_csv(
        OUTPUT_MARGINAL,
        index=False,
    )

    # --------------------------------------------------------
    # Matched CODA-PB2 seed differences
    # --------------------------------------------------------

    paired_all = (
        build_paired_differences(
            terminal
        )
    )

    paired_all.to_csv(
        OUTPUT_PAIRED,
        index=False,
    )

    # --------------------------------------------------------
    # Statistical analysis and Holm family
    # --------------------------------------------------------

    stats = (
        compute_terminal_stats(
            paired_all,
            marginal,
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

    OUTPUT_LATEX_BODY.write_text(
        latex_body
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TERMINAL tAUC100k — CODA vs PB2"
    )

    print(
        "=" * 90
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
        "Wilcoxon: two-sided, zero_method='wilcox', "
        "correction=False; exact when no zero differences, "
        "approx otherwise"
    )

    print(
        "Holm family: 8 terminal-training "
        "learner-environment comparisons"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "LATEX TABLE BODY"
    )

    print(
        "=" * 90
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
        f"  Seed-level tAUC      : {OUTPUT_SEED_LEVEL}"
    )

    print(
        f"  Seed counts          : {OUTPUT_SEED_COUNTS}"
    )

    print(
        f"  Marginal summary     : {OUTPUT_MARGINAL}"
    )

    print(
        f"  Paired differences   : {OUTPUT_PAIRED}"
    )

    print(
        f"  Terminal statistics  : {OUTPUT_STATS}"
    )

    print(
        f"  LaTeX table body     : {OUTPUT_LATEX_BODY}"
    )

    print(
        "\nDone."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()