#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4: Full-horizon training dynamics under PPO and SAC.

For each learner, environment, method, and training seed, this script:

1. Reads archived training metrics from the repository.
2. Splits each worker/trial trajectory at progress-counter rollbacks.
3. Interpolates only within monotone causal segments.
4. Aggregates workers/trials within each training seed using the median.
5. Aggregates the resulting seed-level curves across seeds using
   the median and interquartile range (IQR).
6. Generates the full-horizon training-dynamics figure used in the paper.
7. Saves the numerical seed-level and across-seed data used in the figure.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── Fig04_training_convergence.py
└── results/
    ├── ppo/
    │   └── metrics.zip
    └── sac/
        └── metrics.zip

Outputs
-------

results/analysis/training_convergence/
├── training_convergence_seed_curves.csv
├── training_convergence_summary.csv
├── training_convergence_seed_counts.csv
├── training_convergence.pdf
└── training_convergence.png
"""

from pathlib import Path, PurePosixPath
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

# Works when this file is stored under:
#     <repo_root>/analysis/Fig04_training_convergence.py
#
# The fallback also allows execution from an interactive session.
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent

PPO_ZIP = (
    REPO_ROOT
    / "results"
    / "ppo"
    / "ppo_train.zip"
)

SAC_ZIP = (
    REPO_ROOT
    / "results"
    / "sac"
    / "sac_train.zip"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "training_convergence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Analysis configuration
# ============================================================

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

METHODS = [
    "PBT",
    "PB2",
    "ASHA",
    "CODA",
]

EXPECTED_TRAINING_SEEDS = 10

PPO_MAX_STEPS = 2_000_000
SAC_MAX_STEPS = 1_000_000

# Number of common checkpoints used only for interpolation.
# This is NOT smoothing.
GRID_SIZE = 201


# ============================================================
# Method identification
# ============================================================

def identify_method(filename):
    """
    Map each archived metrics CSV to the method shown in Figure 4.

    CODA-I2O and CODA-O2I are intentionally excluded because this
    figure compares PBT, PB2, ASHA, and full CODA only.
    """
    name = PurePosixPath(filename).name

    if name.startswith("metrics_CODA_FULL_"):
        return "CODA"

    if name.startswith("metrics_PBT_"):
        return "PBT"

    if name.startswith("metrics_PB2_"):
        return "PB2"

    if name.startswith("metrics_ASHA_"):
        return "ASHA"

    return None


# ============================================================
# Rollback-aware trajectory handling
# ============================================================

def split_monotone_segments(df_agent):
    """
    Split one worker/trial trajectory whenever timesteps_total decreases.

    This ensures that interpolation never crosses a progress-counter
    rollback, consistent with the manuscript protocol.

    Duplicate progress counters within a segment retain the latest
    causal report.
    """
    df_agent = df_agent.copy()

    # Causal ordering is preferable to raw row ordering.
    if "causal_order" in df_agent.columns:
        df_agent = df_agent.sort_values(
            "causal_order",
            kind="stable",
        )

    columns = [
        "timesteps_total",
        "env_runners/episode_return_mean",
    ]

    if "causal_order" in df_agent.columns:
        columns.append("causal_order")

    df_agent = df_agent[columns].copy()

    df_agent["timesteps_total"] = pd.to_numeric(
        df_agent["timesteps_total"],
        errors="coerce",
    )

    df_agent["env_runners/episode_return_mean"] = pd.to_numeric(
        df_agent["env_runners/episode_return_mean"],
        errors="coerce",
    )

    df_agent = df_agent.dropna(
        subset=[
            "timesteps_total",
            "env_runners/episode_return_mean",
        ]
    )

    if df_agent.empty:
        return []

    # A decrease indicates restoration to an earlier progress counter.
    rollback = (
        df_agent["timesteps_total"]
        .diff()
        .lt(0)
        .fillna(False)
    )

    segment_id = rollback.cumsum()

    segments = []

    for _, segment in df_agent.groupby(
        segment_id,
        sort=False,
    ):

        if "causal_order" in segment.columns:
            segment = segment.sort_values(
                "causal_order",
                kind="stable",
            )

        # If the same progress appears more than once,
        # retain the latest causal report.
        segment = (
            segment
            .drop_duplicates(
                subset="timesteps_total",
                keep="last",
            )
            .sort_values("timesteps_total")
        )

        if not segment.empty:
            segments.append(segment)

    return segments


def interpolate_participant(df_agent, grid):
    """
    Interpolate one worker/trial onto the common progress grid.

    Interpolation occurs only inside individual monotone segments.
    No interpolation is performed across rollbacks.

    If two causal segments of the same worker support the same progress
    value, the later causal segment overwrites the earlier value. This
    avoids counting the same worker twice at one progress checkpoint.
    """
    trajectory = np.full(
        len(grid),
        np.nan,
        dtype=float,
    )

    segments = split_monotone_segments(
        df_agent
    )

    for segment in segments:

        x = segment[
            "timesteps_total"
        ].to_numpy(
            dtype=float
        )

        y = segment[
            "env_runners/episode_return_mean"
        ].to_numpy(
            dtype=float
        )

        if len(x) == 1:

            mask = np.isclose(
                grid,
                x[0],
                rtol=0,
                atol=1e-9,
            )

            trajectory[
                mask
            ] = y[0]

            continue

        # Interpolate only where this causal segment has support.
        mask = (
            (grid >= x[0])
            & (grid <= x[-1])
        )

        if np.any(mask):

            trajectory[
                mask
            ] = np.interp(
                grid[mask],
                x,
                y,
            )

    return trajectory


# ============================================================
# One seed -> population/trial median curve
# ============================================================

def compute_seed_curve(df, grid):
    """
    Compute the within-seed population/trial median curve:

        R_seed(T) = median_p R_p(T),

    using only workers/trials with valid support at T.
    """
    required = {
        "timesteps_total",
        "env_runners/episode_return_mean",
        "agente_id",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    participant_curves = []

    for _, df_agent in df.groupby(
        "agente_id",
        sort=False,
    ):

        participant_curves.append(
            interpolate_participant(
                df_agent,
                grid,
            )
        )

    if not participant_curves:
        return np.full(
            len(grid),
            np.nan,
            dtype=float,
        )

    participant_curves = np.vstack(
        participant_curves
    )

    # Pandas avoids runtime warnings at checkpoints with no support.
    seed_curve = (
        pd.DataFrame(
            participant_curves
        )
        .median(
            axis=0,
            skipna=True,
        )
        .to_numpy()
    )

    return seed_curve


# ============================================================
# ZIP utilities
# ============================================================

def validate_zip_path(zip_path):
    """
    Validate one training-metrics ZIP before analysis.
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


def infer_environment_from_path(filename):
    """
    Infer environment robustly from any path component instead of
    assuming a fixed ZIP nesting depth.
    """
    parts = PurePosixPath(
        filename
    ).parts

    matches = [
        part
        for part in parts
        if part in ENVIRONMENTS
    ]

    if len(matches) == 1:
        return matches[0]

    return None


# ============================================================
# Load one learner ZIP
# ============================================================

def load_training_curves(
    zip_path,
    learner,
    max_steps,
    grid_size=GRID_SIZE,
):
    """
    Read all main-method CSV files from a PPO/SAC metrics ZIP.

    Returns one seed-level population/trial curve per:
        learner x environment x method x training seed
    """
    validate_zip_path(
        zip_path
    )

    grid = np.linspace(
        0,
        max_steps,
        grid_size,
    )

    all_rows = []

    selected_files = 0

    with zipfile.ZipFile(
        zip_path
    ) as archive:

        for filename in archive.namelist():

            if not filename.endswith(
                ".csv"
            ):
                continue

            method = identify_method(
                filename
            )

            # Exclude CODA-I2O, CODA-O2I, and unrelated files.
            if method is None:
                continue

            environment = (
                infer_environment_from_path(
                    filename
                )
            )

            if environment is None:
                continue

            with archive.open(
                filename
            ) as file:
                df = pd.read_csv(
                    file
                )

            selected_files += 1

            # Prefer the seed recorded inside the file.
            if (
                "semilla" in df.columns
                and df["semilla"].notna().any()
            ):

                seed = int(
                    pd.to_numeric(
                        df["semilla"],
                        errors="coerce",
                    )
                    .dropna()
                    .iloc[0]
                )

            else:

                match = re.search(
                    r"_seed(\d+)\.csv$",
                    PurePosixPath(
                        filename
                    ).name,
                )

                if match is None:
                    raise ValueError(
                        "Could not determine training seed "
                        f"for {filename}"
                    )

                seed = int(
                    match.group(1)
                )

            seed_curve = compute_seed_curve(
                df,
                grid,
            )

            for step, value in zip(
                grid,
                seed_curve,
            ):

                all_rows.append(
                    {
                        "learner": learner,
                        "environment": environment,
                        "method": method,
                        "training_seed": seed,
                        "step": float(step),
                        "return": value,
                        "source_file": filename,
                    }
                )

    if selected_files == 0:
        raise RuntimeError(
            "No compatible main-method metrics CSVs "
            f"were found in {zip_path}"
        )

    curves = pd.DataFrame(
        all_rows
    )

    return curves


# ============================================================
# Across-seed aggregation
# ============================================================

def aggregate_across_seeds(curves):
    """
    Aggregate the seed-level curves pointwise across independent seeds.

    Returns:
        median,
        Q1,
        Q3,
        number of seeds with valid support.
    """
    summary = (
        curves
        .groupby(
            [
                "learner",
                "environment",
                "method",
                "step",
            ],
            as_index=False,
        )
        .agg(
            median=(
                "return",
                "median",
            ),
            q1=(
                "return",
                lambda x: x.quantile(
                    0.25
                ),
            ),
            q3=(
                "return",
                lambda x: x.quantile(
                    0.75
                ),
            ),
            n_seeds=(
                "return",
                "count",
            ),
        )
    )

    return summary


def build_seed_count_table(
    curves
):
    """
    Count distinct training seeds available for every
    learner/environment/method combination.
    """
    counts = (
        curves[
            [
                "learner",
                "environment",
                "method",
                "training_seed",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "learner",
                "environment",
                "method",
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

    return counts


def validate_seed_counts(
    seed_counts
):
    """
    Warn if any expected learner/environment/method combination
    does not contain the expected number of training seeds.
    """
    problems = seed_counts[
        seed_counts[
            "n_training_seeds"
        ]
        != EXPECTED_TRAINING_SEEDS
    ]

    if not problems.empty:

        print(
            "\nWARNING: some combinations do not contain "
            f"{EXPECTED_TRAINING_SEEDS} training seeds:"
        )

        print(
            problems.to_string(
                index=False
            )
        )


# ============================================================
# Plot
# ============================================================

def make_figure(
    ppo_summary,
    sac_summary,
):
    """
    Generate the 2 x 4 training-convergence figure.
    """
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12.0, 5.3),
    )

    learner_data = [
        (
            "PPO",
            ppo_summary,
            PPO_MAX_STEPS,
        ),
        (
            "SAC",
            sac_summary,
            SAC_MAX_STEPS,
        ),
    ]

    for row, (
        learner,
        summary,
        max_steps,
    ) in enumerate(
        learner_data
    ):

        for col, environment in enumerate(
            ENVIRONMENTS
        ):

            ax = axes[
                row,
                col,
            ]

            for method in METHODS:

                data = summary[
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
                ].sort_values(
                    "step"
                )

                if data.empty:
                    continue

                x = (
                    data[
                        "step"
                    ].to_numpy()
                    / 1_000_000
                )

                median = data[
                    "median"
                ].to_numpy()

                q1 = data[
                    "q1"
                ].to_numpy()

                q3 = data[
                    "q3"
                ].to_numpy()

                # Across-seed median.
                line, = ax.plot(
                    x,
                    median,
                    linewidth=1.4,
                    label=method,
                )

                # Interquartile range using the same line color.
                ax.fill_between(
                    x,
                    q1,
                    q3,
                    alpha=0.18,
                    color=line.get_color(),
                )

            short_environment = (
                environment.replace(
                    "-v5",
                    "",
                )
            )

            ax.set_title(
                f"{learner}: "
                f"{short_environment}",
                fontsize=10,
            )

            # Same x-axis range within each learner.
            ax.set_xlim(
                0,
                max_steps
                / 1_000_000,
            )

            ax.grid(
                axis="both",
                linestyle=":",
                alpha=0.30,
            )

            if col == 0:
                ax.set_ylabel(
                    "Training return",
                    fontsize=9,
                )

            if row == 1:
                ax.set_xlabel(
                    "Environment steps (M)",
                    fontsize=9,
                )

            ax.tick_params(
                axis="both",
                labelsize=8,
            )

    # Shared legend.
    handles, labels = (
        axes[
            0,
            0,
        ]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            1.01,
        ),
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.95,
        )
    )

    return fig


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading PPO training metrics..."
    )

    ppo_seed_curves = (
        load_training_curves(
            PPO_ZIP,
            learner="PPO",
            max_steps=PPO_MAX_STEPS,
            grid_size=GRID_SIZE,
        )
    )

    print(
        "Loading SAC training metrics..."
    )

    sac_seed_curves = (
        load_training_curves(
            SAC_ZIP,
            learner="SAC",
            max_steps=SAC_MAX_STEPS,
            grid_size=GRID_SIZE,
        )
    )

    all_seed_curves = pd.concat(
        [
            ppo_seed_curves,
            sac_seed_curves,
        ],
        ignore_index=True,
    )

    # ========================================================
    # Across-seed summaries
    # ========================================================

    ppo_summary = (
        aggregate_across_seeds(
            ppo_seed_curves
        )
    )

    sac_summary = (
        aggregate_across_seeds(
            sac_seed_curves
        )
    )

    all_summary = pd.concat(
        [
            ppo_summary,
            sac_summary,
        ],
        ignore_index=True,
    )

    # ========================================================
    # Seed-count validation
    # ========================================================

    seed_counts = (
        build_seed_count_table(
            all_seed_curves
        )
    )

    validate_seed_counts(
        seed_counts
    )

    print(
        "\nTraining seeds per learner/environment/method:"
    )

    print(
        seed_counts.to_string(
            index=False
        )
    )

    # ========================================================
    # Save numerical artifacts
    # ========================================================

    seed_curves_path = (
        OUTPUT_DIR
        / "training_convergence_seed_curves.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "training_convergence_summary.csv"
    )

    seed_counts_path = (
        OUTPUT_DIR
        / "training_convergence_seed_counts.csv"
    )

    all_seed_curves.to_csv(
        seed_curves_path,
        index=False,
    )

    all_summary.to_csv(
        summary_path,
        index=False,
    )

    seed_counts.to_csv(
        seed_counts_path,
        index=False,
    )

    # ========================================================
    # Generate figure
    # ========================================================

    fig = make_figure(
        ppo_summary,
        sac_summary,
    )

    pdf_path = (
        OUTPUT_DIR
        / "training_convergence.pdf"
    )

    png_path = (
        OUTPUT_DIR
        / "training_convergence.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    # ========================================================
    # Report generated files
    # ========================================================

    print(
        "\nGenerated outputs:"
    )

    print(
        f"  Seed curves : {seed_curves_path}"
    )

    print(
        f"  Summary     : {summary_path}"
    )

    print(
        f"  Seed counts : {seed_counts_path}"
    )

    print(
        f"  PDF         : {pdf_path}"
    )

    print(
        f"  PNG         : {png_path}"
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