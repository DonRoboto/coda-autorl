#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5: Diagnostic contribution to surrogate modeling.

At the primary learner-horizon cutoff, this script summarizes three
seed-level surrogate comparisons:

1. Fixed-kernel diagnostic coordinate ablation:
       delta_fixed = RMSE_noS - RMSE_S

2. Independent kernel refit:
       delta_refit = RMSE_noS - RMSE_S

3. Shuffled-diagnostic control:
       delta_shuffled = RMSE_shuffledS - RMSE_S

Positive values favor the representation containing the original
diagnostic state S.

For each learner/environment condition, the figure reports the median
and interquartile range (IQR) across independent training seeds.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   ├── Fig05_diagnostic_surrogate.py
│   └── diagnostic_surrogate_results.csv
└── results/
    └── analysis/
        └── diagnostic_surrogate/
            └── ... generated outputs ...

Outputs
-------

results/analysis/diagnostic_surrogate/
├── diagnostic_surrogate_primary_seed_level.csv
├── diagnostic_surrogate_seed_counts.csv
├── diagnostic_surrogate_summary.csv
├── diagnostic_surrogate.pdf
└── diagnostic_surrogate.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

# Works when this file is stored under:
#     <repo_root>/analysis/Fig05_diagnostic_surrogate.py
#
# The fallback also allows execution from an interactive session.
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent

# The seed-level surrogate analysis file is kept next to this script.
RESULTS_FILE = (
    SCRIPT_DIR
    / "diagnostic_surrogate_results.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "diagnostic_surrogate"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PRIMARY = (
    OUTPUT_DIR
    / "diagnostic_surrogate_primary_seed_level.csv"
)

OUTPUT_SEED_COUNTS = (
    OUTPUT_DIR
    / "diagnostic_surrogate_seed_counts.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "diagnostic_surrogate_summary.csv"
)

OUTPUT_PDF = (
    OUTPUT_DIR
    / "diagnostic_surrogate.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "diagnostic_surrogate.png"
)


# ============================================================
# Analysis configuration
# ============================================================

PRIMARY_CUTOFF = 0.70
EXPECTED_TRAINING_SEEDS = 10

CONDITIONS = [
    ("PPO", "HalfCheetah-v5"),
    ("PPO", "Hopper-v5"),
    ("PPO", "Swimmer-v5"),
    ("PPO", "Walker2d-v5"),
    ("SAC", "HalfCheetah-v5"),
    ("SAC", "Hopper-v5"),
    ("SAC", "Swimmer-v5"),
    ("SAC", "Walker2d-v5"),
]

LABELS = [
    "PPO HalfCheetah",
    "PPO Hopper",
    "PPO Swimmer",
    "PPO Walker2d",
    "SAC HalfCheetah",
    "SAC Hopper",
    "SAC Swimmer",
    "SAC Walker2d",
]

SEPARATOR_Y = 3.5

METRICS = [
    (
        "fixed_kernel",
        "delta_fixed",
        "(a) Fixed kernel",
        r"no $S$ better",
        r"real $S$ better",
    ),
    (
        "independent_refit",
        "delta_refit",
        "(b) Independent refit",
        r"no $S$ better",
        r"real $S$ better",
    ),
    (
        "shuffled_diagnostic",
        "delta_shuffled",
        "(c) Shuffled diagnostic",
        r"shuffled $S$ better",
        r"real $S$ better",
    ),
]


# ============================================================
# Input loading and validation
# ============================================================

def load_results(
    path,
):
    """
    Load and validate the seed-level surrogate-analysis results.
    """
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            "Diagnostic-surrogate results file not found:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "learner",
        "environment",
        "seed",
        "cutoff",
        "delta_fixed",
        "delta_refit",
        "delta_shuffled",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df[
        "cutoff"
    ] = pd.to_numeric(
        df[
            "cutoff"
        ],
        errors="raise",
    )

    df[
        "seed"
    ] = pd.to_numeric(
        df[
            "seed"
        ],
        errors="raise",
    ).astype(int)

    for metric in [
        "delta_fixed",
        "delta_refit",
        "delta_shuffled",
    ]:

        df[
            metric
        ] = pd.to_numeric(
            df[
                metric
            ],
            errors="coerce",
        )

    return df


def select_primary_cutoff(
    df,
):
    """
    Select the primary learner-horizon cutoff used in Figure 5.
    """
    primary = df[
        np.isclose(
            df[
                "cutoff"
            ].to_numpy(
                dtype=float
            ),
            PRIMARY_CUTOFF,
        )
    ].copy()

    if primary.empty:
        raise ValueError(
            "No rows were found for the primary "
            f"cutoff {PRIMARY_CUTOFF:.2f}."
        )

    return primary


def build_seed_counts(
    primary,
):
    """
    Count distinct training seeds for every learner/environment cell.
    """
    seed_counts = (
        primary
        .groupby(
            [
                "learner",
                "environment",
            ],
            as_index=False,
        )
        .agg(
            n_training_seeds=(
                "seed",
                "nunique",
            )
        )
    )

    return seed_counts


def validate_seed_counts(
    seed_counts,
):
    """
    Require exactly the expected number of independent training seeds
    for every learner/environment cell.
    """
    # Ensure all expected conditions are present.
    expected = pd.DataFrame(
        CONDITIONS,
        columns=[
            "learner",
            "environment",
        ],
    )

    checked = expected.merge(
        seed_counts,
        on=[
            "learner",
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
        != EXPECTED_TRAINING_SEEDS
    ]

    if not bad.empty:

        print(
            "\nInvalid seed counts:"
        )

        print(
            bad.to_string(
                index=False
            )
        )

        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_TRAINING_SEEDS} training seeds "
            "for every learner/environment condition."
        )

    return checked


# ============================================================
# Summary statistics
# ============================================================

def summarize_metric(
    primary,
    metric,
    comparison_name,
):
    """
    Summarize one surrogate comparison using median and IQR
    across training seeds.
    """
    rows = []

    for learner, environment in CONDITIONS:

        current = primary[
            primary[
                "learner"
            ].eq(
                learner
            )
            &
            primary[
                "environment"
            ].eq(
                environment
            )
        ][
            [
                "seed",
                metric,
            ]
        ].copy()

        current = current.dropna(
            subset=[
                metric
            ]
        )

        if len(current) == 0:

            raise ValueError(
                "No finite values found for "
                f"{learner}, {environment}, {metric}"
            )

        if (
            current[
                "seed"
            ].nunique()
            != EXPECTED_TRAINING_SEEDS
        ):

            raise ValueError(
                f"{learner}, {environment}, {metric}: "
                f"expected {EXPECTED_TRAINING_SEEDS} seeds, "
                "but the metric does not contain one finite "
                "observation for every seed."
            )

        values = current[
            metric
        ]

        rows.append(
            {
                "comparison":
                    comparison_name,

                "metric":
                    metric,

                "learner":
                    learner,

                "environment":
                    environment,

                "cutoff":
                    PRIMARY_CUTOFF,

                "n_training_seeds":
                    current[
                        "seed"
                    ].nunique(),

                "median":
                    float(
                        values.median()
                    ),

                "q1":
                    float(
                        values.quantile(
                            0.25
                        )
                    ),

                "q3":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),

                "n_positive":
                    int(
                        (
                            values
                            > 0
                        ).sum()
                    ),

                "n_zero":
                    int(
                        (
                            values
                            == 0
                        ).sum()
                    ),

                "n_negative":
                    int(
                        (
                            values
                            < 0
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plotting helper
# ============================================================

def draw_panel(
    ax,
    summary,
    title,
    left_label,
    right_label,
    show_ylabels=False,
):
    """
    Draw one median-IQR forest-style panel.
    """
    # Enforce the manuscript condition ordering.
    order = pd.DataFrame(
        CONDITIONS,
        columns=[
            "learner",
            "environment",
        ],
    )

    summary = order.merge(
        summary,
        on=[
            "learner",
            "environment",
        ],
        how="left",
        validate="one_to_one",
    )

    y = np.arange(
        len(summary)
    )

    med = summary[
        "median"
    ].to_numpy(
        dtype=float
    )

    q1 = summary[
        "q1"
    ].to_numpy(
        dtype=float
    )

    q3 = summary[
        "q3"
    ].to_numpy(
        dtype=float
    )

    xerr = np.vstack(
        [
            med - q1,
            q3 - med,
        ]
    )

    # Reference line at zero.
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        zorder=1,
    )

    # Separator between PPO and SAC.
    ax.axhline(
        SEPARATOR_Y,
        linestyle="-",
        linewidth=0.8,
        alpha=0.7,
        zorder=1,
    )

    # Median + IQR.
    ax.errorbar(
        med,
        y,
        xerr=xerr,
        fmt="o",
        markersize=5,
        capsize=3,
        linewidth=1.4,
        zorder=3,
    )

    ax.set_title(
        title,
        fontsize=10,
    )

    ax.set_yticks(
        y
    )

    if show_ylabels:

        ax.set_yticklabels(
            LABELS,
            fontsize=8,
        )

        ax.tick_params(
            axis="y",
            length=0,
            pad=5,
        )

    else:

        ax.set_yticklabels(
            []
        )

        ax.tick_params(
            axis="y",
            length=0,
        )

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.30,
    )

    ax.tick_params(
        axis="x",
        labelsize=8,
    )

    # Direction labels.
    ax.text(
        0.02,
        -0.18,
        left_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    ax.text(
        0.98,
        -0.18,
        right_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )


# ============================================================
# Figure generation
# ============================================================

def make_figure(
    summaries,
):
    """
    Generate the three-panel diagnostic-surrogate figure.
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(11.5, 4.5),
        sharey=True,
    )

    panel_specs = [
        (
            summaries[
                "fixed_kernel"
            ],
            "(a) Fixed kernel",
            r"no $S$ better",
            r"real $S$ better",
        ),
        (
            summaries[
                "independent_refit"
            ],
            "(b) Independent refit",
            r"no $S$ better",
            r"real $S$ better",
        ),
        (
            summaries[
                "shuffled_diagnostic"
            ],
            "(c) Shuffled diagnostic",
            r"shuffled $S$ better",
            r"real $S$ better",
        ),
    ]

    for panel_idx, (
        summary,
        title,
        left_label,
        right_label,
    ) in enumerate(
        panel_specs
    ):

        draw_panel(
            axes[
                panel_idx
            ],
            summary,
            title=title,
            left_label=left_label,
            right_label=right_label,
            show_ylabels=(
                panel_idx == 0
            ),
        )

    # Common symmetric x-axis across all three panels.
    all_extremes = np.concatenate(
        [
            summary[
                "q1"
            ].to_numpy(
                dtype=float
            )
            for summary in summaries.values()
        ]
        +
        [
            summary[
                "q3"
            ].to_numpy(
                dtype=float
            )
            for summary in summaries.values()
        ]
    )

    if not np.isfinite(
        all_extremes
    ).any():

        raise ValueError(
            "No finite summary values available "
            "for plotting."
        )

    limit = float(
        np.nanmax(
            np.abs(
                all_extremes
            )
        )
    )

    if (
        not np.isfinite(
            limit
        )
        or limit <= 0
    ):
        limit = 1.0

    limit *= 1.10

    for ax in axes:

        ax.set_xlim(
            -limit,
            limit,
        )

    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        bottom=0.22,
        top=0.88,
        wspace=0.18,
    )

    return fig


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load and validate seed-level surrogate results
    # --------------------------------------------------------

    print(
        "Loading diagnostic-surrogate results..."
    )

    df = load_results(
        RESULTS_FILE
    )

    primary = (
        select_primary_cutoff(
            df
        )
    )

    seed_counts = (
        build_seed_counts(
            primary
        )
    )

    seed_counts = (
        validate_seed_counts(
            seed_counts
        )
    )

    print(
        "\nTraining seeds at the primary cutoff:"
    )

    print(
        seed_counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save the exact seed-level data used by Figure 5
    # --------------------------------------------------------

    primary_export = primary[
        [
            "learner",
            "environment",
            "seed",
            "cutoff",
            "delta_fixed",
            "delta_refit",
            "delta_shuffled",
        ]
    ].sort_values(
        [
            "learner",
            "environment",
            "seed",
        ]
    )

    primary_export.to_csv(
        OUTPUT_PRIMARY,
        index=False,
    )

    seed_counts.to_csv(
        OUTPUT_SEED_COUNTS,
        index=False,
    )

    # --------------------------------------------------------
    # Compute summaries
    # --------------------------------------------------------

    summaries = {}

    summary_frames = []

    for (
        comparison_name,
        metric,
        _,
        _,
        _,
    ) in METRICS:

        current_summary = (
            summarize_metric(
                primary,
                metric=metric,
                comparison_name=comparison_name,
            )
        )

        summaries[
            comparison_name
        ] = current_summary

        summary_frames.append(
            current_summary
        )

    summary_all = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    summary_all.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )

    print(
        "DIAGNOSTIC CONTRIBUTION TO SURROGATE MODELING"
    )

    print(
        "=" * 88
    )

    for comparison_name, _, title, _, _ in METRICS:

        print(
            f"\n{title}:"
        )

        print(
            summaries[
                comparison_name
            ][
                [
                    "learner",
                    "environment",
                    "n_training_seeds",
                    "median",
                    "q1",
                    "q3",
                    "n_positive",
                    "n_zero",
                    "n_negative",
                ]
            ]
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Generate and save figure
    # --------------------------------------------------------

    fig = make_figure(
        summaries
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    # --------------------------------------------------------
    # Report generated files
    # --------------------------------------------------------

    print(
        "\nGenerated outputs:"
    )

    print(
        f"  Primary seed-level data : {OUTPUT_PRIMARY}"
    )

    print(
        f"  Seed counts             : {OUTPUT_SEED_COUNTS}"
    )

    print(
        f"  Summary                 : {OUTPUT_SUMMARY}"
    )

    print(
        f"  PDF                     : {OUTPUT_PDF}"
    )

    print(
        f"  PNG                     : {OUTPUT_PNG}"
    )

    print(
        "\nDone."
    )

    plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()