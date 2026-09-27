#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:12:57 2026

@author: yor5
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# Configuration
# ============================================================

PPO_TRAIN_ZIP = Path("../results/ppo/ppo_train.zip")
SAC_TRAIN_ZIP = Path("../results/sac/sac_train.zip")

# These files identify the training-selected champion for each seed.
# Change the paths/names if yours are different.
PPO_HELDOUT = Path("../results/ppo/heldout_reward_ppo_final/heldout_test_episodes.csv")
SAC_HELDOUT = Path("../results/sac/heldout_reward_sac_final/heldout_test_episodes.csv")


TAUC_WINDOW = 100_000
N_TRAINING_SEEDS = 10

LEARNERS = ["PPO", "SAC"]

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

OUTPUT_SEED_LEVEL = "terminal_tauc_seed_level.csv"
OUTPUT_STATS = "terminal_tauc_primary_stats.csv"
OUTPUT_LATEX_BODY = "terminal_tauc_table_body.tex"


# ============================================================
# Column names used in training logs
# ============================================================

AGENT_COL = "agente_id"
STEP_COL = "timesteps_total"
RETURN_COL = "env_runners/episode_return_mean"

# If causal_order exists, it will be used to preserve logging order.
CAUSAL_ORDER_COL = "causal_order"


# ============================================================
# Load held-out data
# ============================================================

def load_heldout(path, learner):
    if not path.exists():
        raise FileNotFoundError(
            f"Held-out file not found:\n  {path.resolve()}"
        )

    df = pd.read_csv(path)

    required = {
        "method",
        "environment",
        "training_seed",
        "champion_agent",
        "test_return",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["learner"] = learner

    df["training_seed"] = pd.to_numeric(
        df["training_seed"],
        errors="raise"
    ).astype(int)

    # We only need CODA and PB2 for Table 6.
    df = df[df["method"].isin(METHODS)].copy()

    return df


ppo_heldout = load_heldout(
    PPO_HELDOUT,
    "PPO"
)

sac_heldout = load_heldout(
    SAC_HELDOUT,
    "SAC"
)

heldout = pd.concat(
    [ppo_heldout, sac_heldout],
    ignore_index=True
)


# ============================================================
# Extract one champion identity per training seed
# ============================================================

champion_counts = (
    heldout
    .groupby(
        [
            "learner",
            "method",
            "environment",
            "training_seed",
        ]
    )["champion_agent"]
    .nunique()
)

if not (champion_counts == 1).all():
    bad = champion_counts[champion_counts != 1]
    print(bad)
    raise ValueError(
        "At least one learner/method/environment/seed "
        "contains more than one champion."
    )


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
    .reset_index(drop=True)
)


# ============================================================
# Training ZIP naming
# ============================================================

def expected_training_member(
    learner,
    method,
    environment,
    seed
):
    """
    Expected paths inside your training ZIPs.

    Modify these two patterns if your ZIP uses different names.
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
    seed
):
    """
    First tries the exact expected path.
    If that fails, searches by basename.
    """

    expected = expected_training_member(
        learner,
        method,
        environment,
        seed
    )

    names = zf.namelist()

    if expected in names:
        return expected

    expected_basename = Path(expected).name

    candidates = [
        name
        for name in names
        if Path(name).name == expected_basename
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"\nCould not uniquely locate training CSV.\n"
        f"Learner: {learner}\n"
        f"Method: {method}\n"
        f"Environment: {environment}\n"
        f"Seed: {seed}\n"
        f"Expected: {expected}\n"
        f"Candidates: {candidates}"
    )


# ============================================================
# Read training CSV directly from ZIP
# ============================================================

def read_training_csv(
    zf,
    member
):
    with zf.open(member) as f:
        return pd.read_csv(f)


# ============================================================
# Recover final execution segment for selected champion
# ============================================================

def get_final_training_segment(
    df,
    champion_agent
):
    required = {
        AGENT_COL,
        STEP_COL,
        RETURN_COL,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Training CSV missing columns: {sorted(missing)}"
        )

    current = df[
        df[AGENT_COL].astype(str)
        ==
        str(champion_agent)
    ].copy()

    if current.empty:
        raise ValueError(
            f"Champion '{champion_agent}' not found "
            f"in training CSV."
        )

    # Preserve the actual report order if an explicit
    # causal/logging-order column exists.
    if CAUSAL_ORDER_COL in current.columns:
        current[CAUSAL_ORDER_COL] = pd.to_numeric(
            current[CAUSAL_ORDER_COL],
            errors="coerce"
        )

        current = current.sort_values(
            CAUSAL_ORDER_COL,
            kind="stable"
        )

    # Otherwise preserve original CSV row order.
    current["_row_order"] = np.arange(len(current))

    current[STEP_COL] = pd.to_numeric(
        current[STEP_COL],
        errors="coerce"
    )

    current[RETURN_COL] = pd.to_numeric(
        current[RETURN_COL],
        errors="coerce"
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

    # --------------------------------------------------------
    # Detect progress-counter rollbacks in report order.
    #
    # A new segment starts whenever timesteps_total decreases.
    # --------------------------------------------------------

    steps_in_report_order = current[
        STEP_COL
    ].to_numpy(float)

    rollback = np.r_[
        False,
        np.diff(
            steps_in_report_order
        ) < 0
    ]

    current["_segment"] = np.cumsum(
        rollback
    )

    # Manuscript: terminal segment begins after the
    # last recorded progress-counter decrease.
    final_segment_id = current[
        "_segment"
    ].max()

    segment = current[
        current["_segment"]
        ==
        final_segment_id
    ].copy()

    # --------------------------------------------------------
    # Duplicate progress counters:
    # retain the latest report.
    # --------------------------------------------------------

    if CAUSAL_ORDER_COL in segment.columns:
        segment = segment.sort_values(
            CAUSAL_ORDER_COL,
            kind="stable"
        )
    else:
        segment = segment.sort_values(
            "_row_order",
            kind="stable"
        )

    segment = (
        segment
        .drop_duplicates(
            subset=[STEP_COL],
            keep="last"
        )
        .sort_values(STEP_COL)
        .reset_index(drop=True)
    )

    if len(segment) < 2:
        raise ValueError(
            "Final segment contains fewer than two "
            "unique progress values."
        )

    return segment


# ============================================================
# Compute tAUC_100k
# ============================================================

def compute_terminal_tauc(
    segment,
    window=TAUC_WINDOW
):
    x = segment[
        STEP_COL
    ].to_numpy(float)

    y = segment[
        RETURN_COL
    ].to_numpy(float)

    if not (
        np.isfinite(x).all()
        and
        np.isfinite(y).all()
    ):
        raise ValueError(
            "Non-finite values in terminal segment."
        )

    if np.any(
        np.diff(x) <= 0
    ):
        raise ValueError(
            "Progress values must be strictly increasing "
            "after duplicate removal."
        )

    t_end = float(
        x[-1]
    )

    t_start = (
        t_end - window
    )

    # Manuscript requires complete support over the
    # final 100k interaction window.
    if x[0] > t_start:
        raise ValueError(
            f"Incomplete tAUC support: "
            f"segment starts at {x[0]:.0f}, "
            f"required <= {t_start:.0f}"
        )

    # --------------------------------------------------------
    # Interpolate exact window boundaries
    # --------------------------------------------------------

    y_start = np.interp(
        t_start,
        x,
        y
    )

    y_end = np.interp(
        t_end,
        x,
        y
    )

    mask = (
        (x > t_start)
        &
        (x < t_end)
    )

    x_window = np.concatenate(
        [
            [t_start],
            x[mask],
            [t_end],
        ]
    )

    y_window = np.concatenate(
        [
            [y_start],
            y[mask],
            [y_end],
        ]
    )

    # NumPy compatibility
    if hasattr(
        np,
        "trapezoid"
    ):
        area = np.trapezoid(
            y_window,
            x_window
        )
    else:
        area = np.trapz(
            y_window,
            x_window
        )

    tauc = (
        area
        /
        window
    )

    return {
        "tAUC100k":
            float(tauc),

        "window_start":
            float(t_start),

        "window_end":
            float(t_end),

        "segment_start":
            float(x[0]),

        "segment_end":
            float(x[-1]),

        "n_points_window":
            int(len(x_window)),
    }


# ============================================================
# Compute tAUC for every selected champion
# ============================================================

def compute_from_zip(
    zip_path,
    learner,
    champion_table
):

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Training ZIP not found:\n"
            f"  {zip_path.resolve()}"
        )

    rows = []

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zf:

        learner_champions = (
            champion_table[
                champion_table["learner"]
                ==
                learner
            ]
        )

        for _, champ in (
            learner_champions
            .iterrows()
        ):

            method = champ[
                "method"
            ]

            environment = champ[
                "environment"
            ]

            seed = int(
                champ[
                    "training_seed"
                ]
            )

            champion_agent = champ[
                "champion_agent"
            ]

            member = find_training_member(
                zf,
                learner,
                method,
                environment,
                seed
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
                member
            )

            final_segment = (
                get_final_training_segment(
                    train_df,
                    champion_agent
                )
            )

            result = compute_terminal_tauc(
                final_segment
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


ppo_tauc = compute_from_zip(
    PPO_TRAIN_ZIP,
    "PPO",
    champions
)

sac_tauc = compute_from_zip(
    SAC_TRAIN_ZIP,
    "SAC",
    champions
)

terminal = pd.concat(
    [
        ppo_tauc,
        sac_tauc,
    ],
    ignore_index=True
)


# ============================================================
# Validate complete design
# ============================================================

seed_counts = (
    terminal
    .groupby(
        [
            "learner",
            "method",
            "environment",
        ]
    )[
        "training_seed"
    ]
    .nunique()
)

print(
    "\nTraining seeds with valid tAUC:\n"
)

print(
    seed_counts
)

if not (
    seed_counts
    ==
    N_TRAINING_SEEDS
).all():

    bad = seed_counts[
        seed_counts
        !=
        N_TRAINING_SEEDS
    ]

    print(
        "\nIncomplete cells:"
    )

    print(
        bad
    )

    raise RuntimeError(
        "Expected exactly "
        f"{N_TRAINING_SEEDS} valid tAUC values "
        "for every learner/method/environment."
    )


terminal.to_csv(
    OUTPUT_SEED_LEVEL,
    index=False
)


# ============================================================
# Marginal median [Q1, Q3]
# ============================================================

def q1(x):
    return float(
        x.quantile(0.25)
    )


def q3(x):
    return float(
        x.quantile(0.75)
    )


marginal = (
    terminal
    .groupby(
        [
            "learner",
            "method",
            "environment",
        ],
        as_index=False
    )
    .agg(
        median=(
            "tAUC100k",
            "median"
        ),

        q1=(
            "tAUC100k",
            q1
        ),

        q3=(
            "tAUC100k",
            q3
        ),
    )
)


# ============================================================
# Build paired CODA-PB2 statistics
# ============================================================

rows = []

for learner in LEARNERS:

    for environment in ENVIRONMENTS:

        coda = (
            terminal[
                (terminal["learner"] == learner)
                &
                (
                    terminal["environment"]
                    ==
                    environment
                )
                &
                (
                    terminal["method"]
                    ==
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
                (terminal["learner"] == learner)
                &
                (
                    terminal["environment"]
                    ==
                    environment
                )
                &
                (
                    terminal["method"]
                    ==
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
                validate="one_to_one"
            )
            .sort_values(
                "training_seed"
            )
        )

        if len(
            paired
        ) != N_TRAINING_SEEDS:

            raise RuntimeError(
                f"{learner}, {environment}: "
                f"expected {N_TRAINING_SEEDS} "
                f"paired seeds, found {len(paired)}"
            )

        d = (
            paired["CODA"]
            -
            paired["PB2"]
        ).to_numpy(float)

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

        # Wilcoxon
        #
        # If there are no zero differences, exact is used.
        # If zeros exist, SciPy cannot use the exact
        # distribution with zero_method='wilcox', so we
        # fall back to the asymptotic implementation.
        if ties == 0:

            test = wilcoxon(
                d,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="exact",
            )

        else:

            test = wilcoxon(
                d,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="approx",
            )

        coda_marginal = marginal[
            (marginal["learner"] == learner)
            &
            (
                marginal["environment"]
                ==
                environment
            )
            &
            (
                marginal["method"]
                ==
                "CODA"
            )
        ].iloc[0]

        pb2_marginal = marginal[
            (marginal["learner"] == learner)
            &
            (
                marginal["environment"]
                ==
                environment
            )
            &
            (
                marginal["method"]
                ==
                "PB2"
            )
        ].iloc[0]

        rows.append(
            {
                "learner":
                    learner,

                "environment":
                    environment,

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

                # Important:
                # median of paired differences,
                # NOT difference of marginal medians.
                "median_difference":
                    float(
                        np.median(d)
                    ),

                "wins":
                    wins,

                "ties":
                    ties,

                "losses":
                    losses,

                "wilcoxon_W":
                    float(
                        test.statistic
                    ),

                "p_raw":
                    float(
                        test.pvalue
                    ),
            }
        )


stats = pd.DataFrame(
    rows
)


# ============================================================
# Holm correction over all 8 terminal-training comparisons
# ============================================================

stats["p_Holm"] = multipletests(
    stats["p_raw"],
    alpha=0.05,
    method="holm",
)[1]


stats.to_csv(
    OUTPUT_STATS,
    index=False
)


# ============================================================
# Print audit values
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "TERMINAL tAUC_100k — CODA vs PB2"
)

print(
    "=" * 90
)

print(
    stats.to_string(
        index=False
    )
)


# ============================================================
# Formatting helpers
# ============================================================

def fmt_summary(
    median,
    q1_value,
    q3_value
):
    return (
        f"{median:.1f} "
        f"[{q1_value:.1f}, "
        f"{q3_value:.1f}]"
    )


def fmt_delta(
    value
):
    return (
        f"{value:.1f}"
    )


def fmt_p(
    value
):
    return (
        f"{value:.3f}"
    )


def fmt_wtl(
    row
):
    return (
        f"{int(row['wins'])}/"
        f"{int(row['ties'])}/"
        f"{int(row['losses'])}"
    )


# ============================================================
# Generate LaTeX BODY
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

    for environment in ENVIRONMENTS:

        row = stats[
            (stats["learner"] == learner)
            &
            (
                stats["environment"]
                ==
                environment
            )
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Expected exactly one result for "
                f"{learner}, {environment}"
            )

        row = row.iloc[0]

        coda_text = fmt_summary(
            row["coda_median"],
            row["coda_q1"],
            row["coda_q3"],
        )

        pb2_text = fmt_summary(
            row["pb2_median"],
            row["pb2_q1"],
            row["pb2_q3"],
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

        lines.append("")


latex_body = (
    "\n".join(
        lines
    )
    .rstrip()
)


# ============================================================
# Save LaTeX body
# ============================================================

Path(
    OUTPUT_LATEX_BODY
).write_text(
    latex_body + "\n",
    encoding="utf-8"
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

print(
    "\nGenerated files:"
)

print(
    f"  {OUTPUT_SEED_LEVEL}"
)

print(
    f"  {OUTPUT_STATS}"
)

print(
    f"  {OUTPUT_LATEX_BODY}"
)