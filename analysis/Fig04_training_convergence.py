#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:39:22 2026

@author: yor5
"""

import re
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PPO_ZIP = "../results/ppo/ppo_train.zip"
SAC_ZIP = "../results/sac/sac_train.zip"

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
    Maps each CSV filename to the method shown in the figure.

    CODA-I2O and CODA-O2I are intentionally excluded because this
    figure compares the main methods PBT, PB2, ASHA, and full CODA.
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
    rollback, consistent with the protocol in the manuscript.

    Duplicate progress counters within a segment retain the latest
    causal report.
    """

    df_agent = df_agent.copy()

    # Causal ordering is preferable to row ordering.
    if "causal_order" in df_agent.columns:
        df_agent = df_agent.sort_values(
            "causal_order",
            kind="stable"
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
        errors="coerce"
    )

    df_agent["env_runners/episode_return_mean"] = pd.to_numeric(
        df_agent["env_runners/episode_return_mean"],
        errors="coerce"
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
        sort=False
    ):

        if "causal_order" in segment.columns:
            segment = segment.sort_values(
                "causal_order",
                kind="stable"
            )

        # If the same progress appears more than once,
        # retain the latest report.
        segment = (
            segment
            .drop_duplicates(
                subset="timesteps_total",
                keep="last"
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
    value, the later causal segment is retained. This avoids counting
    the same worker twice at one progress checkpoint.
    """

    trajectory = np.full(
        len(grid),
        np.nan,
        dtype=float
    )

    segments = split_monotone_segments(df_agent)

    for segment in segments:

        x = segment[
            "timesteps_total"
        ].to_numpy(dtype=float)

        y = segment[
            "env_runners/episode_return_mean"
        ].to_numpy(dtype=float)

        if len(x) == 1:

            mask = np.isclose(
                grid,
                x[0],
                rtol=0,
                atol=1e-9
            )

            trajectory[mask] = y[0]
            continue

        # Only interpolate where this segment has causal support.
        mask = (
            (grid >= x[0])
            & (grid <= x[-1])
        )

        if np.any(mask):

            trajectory[mask] = np.interp(
                grid[mask],
                x,
                y
            )

    return trajectory


# ============================================================
# One seed -> population/trial median curve
# ============================================================

def compute_seed_curve(df, grid):
    """
    Implements the within-seed aggregation:

        R_pop(T) = median_p R_p(T)

    using only workers/trials with valid support at T.
    """

    required = {
        "timesteps_total",
        "env_runners/episode_return_mean",
        "agente_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    participant_curves = []

    for _, df_agent in df.groupby(
        "agente_id",
        sort=False
    ):

        participant_curves.append(
            interpolate_participant(
                df_agent,
                grid
            )
        )

    if not participant_curves:
        return np.full(
            len(grid),
            np.nan
        )

    participant_curves = np.vstack(
        participant_curves
    )

    # Pandas avoids warnings at checkpoints with no support.
    seed_curve = (
        pd.DataFrame(participant_curves)
        .median(
            axis=0,
            skipna=True
        )
        .to_numpy()
    )

    return seed_curve


# ============================================================
# Load one learner ZIP
# ============================================================

def load_training_curves(
    zip_path,
    max_steps,
    grid_size=201
):
    """
    Reads all main-method CSV files from a PPO/SAC ZIP.

    Returns one population curve per:
        environment x method x seed
    """

    grid = np.linspace(
        0,
        max_steps,
        grid_size
    )

    all_rows = []

    with zipfile.ZipFile(zip_path) as z:

        for filename in z.namelist():

            if not filename.endswith(".csv"):
                continue

            method = identify_method(filename)

            # Excludes CODA-I2O / CODA-O2I and unrelated files.
            if method is None:
                continue

            parts = PurePosixPath(
                filename
            ).parts

            if len(parts) < 3:
                continue

            environment = parts[1]

            if environment not in ENVIRONMENTS:
                continue

            with z.open(filename) as file:
                df = pd.read_csv(file)

            # Prefer the seed recorded in the file itself.
            if (
                "semilla" in df.columns
                and df["semilla"].notna().any()
            ):
                seed = int(
                    df["semilla"]
                    .dropna()
                    .iloc[0]
                )

            else:
                match = re.search(
                    r"_seed(\d+)\.csv$",
                    filename
                )

                if match is None:
                    raise ValueError(
                        f"Could not determine seed: "
                        f"{filename}"
                    )

                seed = int(
                    match.group(1)
                )

            seed_curve = compute_seed_curve(
                df,
                grid
            )

            for step, value in zip(
                grid,
                seed_curve
            ):

                all_rows.append({
                    "environment": environment,
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "return": value,
                })

    curves = pd.DataFrame(all_rows)

    return curves


# ============================================================
# Across-seed aggregation
# ============================================================

def aggregate_across_seeds(curves):
    """
    Pointwise aggregation across the ten independent training seeds.

    Output:
        median
        Q1
        Q3
        number of seeds with support
    """

    summary = (
        curves
        .groupby(
            [
                "environment",
                "method",
                "step",
            ],
            as_index=False
        )
        .agg(
            median=(
                "return",
                "median"
            ),
            q1=(
                "return",
                lambda x: x.quantile(0.25)
            ),
            q3=(
                "return",
                lambda x: x.quantile(0.75)
            ),
            n_seeds=(
                "return",
                "count"
            ),
        )
    )

    return summary


# ============================================================
# Load PPO and SAC
# ============================================================

ppo_seed_curves = load_training_curves(
    PPO_ZIP,
    PPO_MAX_STEPS,
    GRID_SIZE
)

sac_seed_curves = load_training_curves(
    SAC_ZIP,
    SAC_MAX_STEPS,
    GRID_SIZE
)

ppo_summary = aggregate_across_seeds(
    ppo_seed_curves
)

sac_summary = aggregate_across_seeds(
    sac_seed_curves
)


# ============================================================
# Sanity checks
# ============================================================

print("\nPPO seeds per method/environment:")
print(
    ppo_seed_curves
    .groupby(
        ["environment", "method"]
    )["seed"]
    .nunique()
    .unstack()
)

print("\nSAC seeds per method/environment:")
print(
    sac_seed_curves
    .groupby(
        ["environment", "method"]
    )["seed"]
    .nunique()
    .unstack()
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(12.0, 5.3)
)


learner_data = [
    (
        "PPO",
        ppo_summary,
        PPO_MAX_STEPS
    ),
    (
        "SAC",
        sac_summary,
        SAC_MAX_STEPS
    ),
]


for row, (
    learner,
    summary,
    max_steps
) in enumerate(learner_data):

    for col, environment in enumerate(
        ENVIRONMENTS
    ):

        ax = axes[row, col]

        for method in METHODS:

            data = summary[
                (
                    summary["environment"]
                    == environment
                )
                &
                (
                    summary["method"]
                    == method
                )
            ].sort_values("step")

            x = (
                data["step"]
                .to_numpy()
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

            # Median across seeds
            line, = ax.plot(
                x,
                median,
                linewidth=1.4,
                label=method
            )

            # Interquartile range
            ax.fill_between(
                x,
                q1,
                q3,
                alpha=0.18
            )

        short_env = environment.replace(
            "-v5",
            ""
        )

        ax.set_title(
            f"{learner}: {short_env}",
            fontsize=10
        )

        # Same x-axis range within each backbone
        ax.set_xlim(
            0,
            max_steps / 1_000_000
        )

        ax.grid(
            axis="both",
            linestyle=":",
            alpha=0.30
        )

        if col == 0:
            ax.set_ylabel(
                "Training return",
                fontsize=9
            )

        if row == 1:
            ax.set_xlabel(
                "Environment steps (M)",
                fontsize=9
            )

        ax.tick_params(
            axis="both",
            labelsize=8
        )


# ============================================================
# Shared legend
# ============================================================

handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.01)
)


# ============================================================
# Layout and export
# ============================================================

fig.tight_layout(
    rect=(0, 0, 1, 0.95)
)

plt.savefig(
    "training_convergence.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "training_convergence.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()