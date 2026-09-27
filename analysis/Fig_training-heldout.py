#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:50:51 2026

@author: yor5
"""
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# ============================================================
# Files
# ============================================================

PPO_TRAIN_ZIP = "../results/ppo/ppo_train.zip"
SAC_TRAIN_ZIP = "../results/sac/sac_train.zip"

PPO_HELDOUT_FILE = "../results/ppo/heldout_reward_ppo_final/heldout_test_episodes.csv"
SAC_HELDOUT_FILE = "../results/sac/heldout_reward_sac_final/heldout_test_episodes.csv"


# ============================================================
# Configuration
# ============================================================

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

TAUC_WINDOW = 100_000


# ============================================================
# 1. Held-out evaluation
# ============================================================

def load_heldout(filename, learner):
    """
    Read episode-level held-out results and produce one row per
    learner/method/environment/training seed.

    The champion identity recorded in the held-out file is used to
    locate the corresponding training trajectory in the ZIP.

    G = mean return over the 100 held-out episodes.
    """

    df = pd.read_csv(filename)

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
            f"Missing columns in {filename}: {missing}"
        )

    df = df[
        df["method"].isin(METHODS)
    ].copy()

    # --------------------------------------------------------
    # Verify that each method/environment/seed has one champion
    # --------------------------------------------------------

    champion_counts = (
        df.groupby(
            [
                "method",
                "environment",
                "training_seed",
            ]
        )["champion_agent"]
        .nunique()
    )

    if not (champion_counts == 1).all():
        raise ValueError(
            "At least one held-out run contains more than one "
            "champion_agent for the same training seed."
        )

    # --------------------------------------------------------
    # Collapse 100 episodes -> one seed-level held-out outcome
    # --------------------------------------------------------

    seed_level = (
        df.groupby(
            [
                "method",
                "environment",
                "training_seed",
            ],
            as_index=False
        )
        .agg(
            champion_agent=(
                "champion_agent",
                "first"
            ),
            heldout_mean=(
                "test_return",
                "mean"
            ),
            n_test_episodes=(
                "test_return",
                "count"
            ),
        )
    )

    seed_level["learner"] = learner

    seed_level = seed_level.rename(
        columns={
            "training_seed": "seed"
        }
    )

    # --------------------------------------------------------
    # Protocol validation
    # --------------------------------------------------------

    bad = seed_level[
        seed_level["n_test_episodes"] != 100
    ]

    if not bad.empty:
        print(
            f"\nWARNING: {filename} contains champions "
            "without exactly 100 held-out episodes:"
        )
        print(
            bad[
                [
                    "method",
                    "environment",
                    "seed",
                    "n_test_episodes",
                ]
            ]
        )

    return seed_level


# ============================================================
# 2. Locate the appropriate training CSV inside each ZIP
# ============================================================

def expected_training_member(
    learner,
    method,
    environment,
    seed
):
    """
    Return the expected CSV path inside the corresponding ZIP.
    """

    if method == "CODA":

        return (
            f"metrics/{environment}/"
            f"metrics_CODA_FULL_seed{seed}.csv"
        )

    elif method == "PB2":

        return (
            f"metrics/{environment}/"
            f"metrics_PB2_{learner}_HPO_seed{seed}.csv"
        )

    else:

        raise ValueError(
            f"Unsupported method: {method}"
        )


def find_training_member(
    zip_file,
    learner,
    method,
    environment,
    seed
):
    """
    Locate the training CSV inside the ZIP.

    First tries the exact expected path. If the ZIP structure changes
    slightly, falls back to matching the filename.
    """

    expected = expected_training_member(
        learner,
        method,
        environment,
        seed
    )

    names = zip_file.namelist()

    if expected in names:
        return expected

    # Fallback based on basename
    target_basename = PurePosixPath(
        expected
    ).name

    matches = [
        name
        for name in names
        if PurePosixPath(name).name == target_basename
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Training CSV not found for "
            f"{learner}, {method}, {environment}, seed={seed}"
        )

    raise RuntimeError(
        f"Multiple candidate training CSVs found for "
        f"{learner}, {method}, {environment}, seed={seed}: "
        f"{matches}"
    )


# ============================================================
# 3. Extract the terminal causal segment
# ============================================================

def get_final_training_segment(
    df,
    champion_agent
):
    """
    Extract the final monotone progress segment for the selected
    champion.

    A new segment begins whenever timesteps_total decreases,
    corresponding to a progress-counter rollback after checkpoint
    restoration.

    Duplicate progress counters retain the latest causal report.
    """

    required = {
        "agente_id",
        "timesteps_total",
        "env_runners/episode_return_mean",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Training CSV is missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Select champion
    # --------------------------------------------------------

    agent = df[
        df["agente_id"].astype(str)
        == str(champion_agent)
    ].copy()

    if agent.empty:
        raise ValueError(
            f"Champion {champion_agent} not found "
            "in training CSV."
        )

    # --------------------------------------------------------
    # Preserve causal report ordering
    # --------------------------------------------------------

    if "causal_order" in agent.columns:

        agent = agent.sort_values(
            "causal_order",
            kind="stable"
        )

    else:

        # Row order becomes the fallback causal ordering.
        agent = agent.reset_index(
            drop=False
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    agent["timesteps_total"] = pd.to_numeric(
        agent["timesteps_total"],
        errors="coerce"
    )

    agent[
        "env_runners/episode_return_mean"
    ] = pd.to_numeric(
        agent[
            "env_runners/episode_return_mean"
        ],
        errors="coerce"
    )

    agent = agent.dropna(
        subset=[
            "timesteps_total",
            "env_runners/episode_return_mean",
        ]
    )

    if agent.empty:
        raise ValueError(
            f"No finite training observations for "
            f"{champion_agent}."
        )

    # --------------------------------------------------------
    # Progress-counter rollback detection
    # --------------------------------------------------------

    rollback = (
        agent["timesteps_total"]
        .diff()
        .lt(0)
        .fillna(False)
    )

    segment_id = rollback.cumsum()

    final_segment_id = segment_id.iloc[-1]

    segment = agent[
        segment_id == final_segment_id
    ].copy()

    # --------------------------------------------------------
    # If the same progress counter appears repeatedly,
    # retain the latest causal observation.
    # --------------------------------------------------------

    if "causal_order" in segment.columns:

        segment = segment.sort_values(
            "causal_order",
            kind="stable"
        )

    segment = (
        segment
        .drop_duplicates(
            subset="timesteps_total",
            keep="last"
        )
        .sort_values(
            "timesteps_total"
        )
        .reset_index(drop=True)
    )

    return segment


# ============================================================
# 4. Compute tAUC_100k
# ============================================================

def compute_terminal_tauc(
    segment,
    window=100_000
):
    """
    Calculate terminal training performance:

        tAUC_W =
            (1/W) * integral_{T_end-W}^{T_end} R(T) dT

    using linear interpolation and trapezoidal integration.
    """

    x = segment[
        "timesteps_total"
    ].to_numpy(dtype=float)

    y = segment[
        "env_runners/episode_return_mean"
    ].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError(
            "Terminal segment has fewer than two observations."
        )

    t_end = x[-1]
    t_start = t_end - window

    # Complete support is required.
    if x[0] > t_start:

        raise ValueError(
            f"Insufficient terminal support: "
            f"segment starts at {x[0]:.0f}, "
            f"but tAUC requires support from "
            f"{t_start:.0f}."
        )

    # --------------------------------------------------------
    # Interior observed points
    # --------------------------------------------------------

    inside = (
        (x > t_start)
        & (x < t_end)
    )

    x_window = x[inside]
    y_window = y[inside]

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

    x_integral = np.concatenate(
        [
            [t_start],
            x_window,
            [t_end],
        ]
    )

    y_integral = np.concatenate(
        [
            [y_start],
            y_window,
            [y_end],
        ]
    )

    # --------------------------------------------------------
    # Support-normalized trapezoidal integral
    # --------------------------------------------------------

    auc = np.trapezoid(
        y_integral,
        x_integral
    )

    tauc = auc / window

    return {
        "tAUC100k": tauc,
        "segment_start": x[0],
        "segment_end": t_end,
        "window_start": t_start,
        "n_segment_points": len(segment),
    }


# ============================================================
# 5. Recompute tAUC directly from one learner ZIP
# ============================================================

def compute_training_metrics_from_zip(
    zip_path,
    heldout_seed_level,
    learner
):
    """
    For every CODA/PB2 held-out champion:
      1. locate its training CSV,
      2. extract the selected champion,
      3. identify its terminal causal segment,
      4. calculate tAUC_100k.
    """

    rows = []

    current_heldout = heldout_seed_level[
        heldout_seed_level["learner"]
        == learner
    ].copy()

    with zipfile.ZipFile(
        zip_path,
        mode="r"
    ) as z:

        for row in current_heldout.itertuples(
            index=False
        ):

            member = find_training_member(
                zip_file=z,
                learner=learner,
                method=row.method,
                environment=row.environment,
                seed=int(row.seed),
            )

            with z.open(member) as f:
                df_train = pd.read_csv(f)

            segment = get_final_training_segment(
                df_train,
                champion_agent=row.champion_agent
            )

            tauc_info = compute_terminal_tauc(
                segment,
                window=TAUC_WINDOW
            )

            rows.append({
                "learner": learner,
                "method": row.method,
                "environment": row.environment,
                "seed": int(row.seed),
                "champion_agent": row.champion_agent,
                "tAUC100k": tauc_info[
                    "tAUC100k"
                ],
                "segment_start": tauc_info[
                    "segment_start"
                ],
                "segment_end": tauc_info[
                    "segment_end"
                ],
                "window_start": tauc_info[
                    "window_start"
                ],
                "n_segment_points": tauc_info[
                    "n_segment_points"
                ],
            })

    return pd.DataFrame(rows)


# ============================================================
# 6. Read held-out results
# ============================================================

ppo_heldout = load_heldout(
    PPO_HELDOUT_FILE,
    learner="PPO"
)

sac_heldout = load_heldout(
    SAC_HELDOUT_FILE,
    learner="SAC"
)

heldout = pd.concat(
    [
        ppo_heldout,
        sac_heldout,
    ],
    ignore_index=True
)


# ============================================================
# 7. Calculate tAUC_100k directly from ZIPs
# ============================================================

ppo_training = compute_training_metrics_from_zip(
    PPO_TRAIN_ZIP,
    heldout,
    learner="PPO"
)

sac_training = compute_training_metrics_from_zip(
    SAC_TRAIN_ZIP,
    heldout,
    learner="SAC"
)

training = pd.concat(
    [
        ppo_training,
        sac_training,
    ],
    ignore_index=True
)


# ============================================================
# 8. Merge training and held-out outcomes
# ============================================================

data = training.merge(
    heldout[
        [
            "learner",
            "method",
            "environment",
            "seed",
            "champion_agent",
            "heldout_mean",
            "n_test_episodes",
        ]
    ],
    on=[
        "learner",
        "method",
        "environment",
        "seed",
        "champion_agent",
    ],
    how="inner",
    validate="one_to_one"
)


# ============================================================
# 9. Basic validation
# ============================================================

expected_rows = (
    2       # learners
    * 2     # methods
    * 4     # environments
    * 10    # seeds
)

print(
    f"\nRows obtained: {len(data)} "
    f"(expected {expected_rows})"
)

if len(data) != expected_rows:

    print(
        "WARNING: unexpected number of matched "
        "training/held-out observations."
    )


print(
    "\nSeeds available per learner/method/environment:\n"
)

print(
    data.groupby(
        [
            "learner",
            "method",
            "environment",
        ]
    )["seed"]
    .nunique()
)


# ============================================================
# 10. Compute Delta_eval and Spearman rho
# ============================================================

data["delta_eval"] = (
    data["heldout_mean"]
    - data["tAUC100k"]
)


summary_rows = []

for learner in [
    "PPO",
    "SAC",
]:

    for environment in ENVIRONMENTS:

        for method in METHODS:

            current = data[
                (
                    data["learner"]
                    == learner
                )
                &
                (
                    data["environment"]
                    == environment
                )
                &
                (
                    data["method"]
                    == method
                )
            ].copy()

            if len(current) != 10:

                print(
                    f"WARNING: {learner}, "
                    f"{environment}, {method}: "
                    f"{len(current)} seeds instead of 10."
                )

            rho, _ = spearmanr(
                current["tAUC100k"],
                current["heldout_mean"]
            )

            summary_rows.append({
                "learner": learner,
                "environment": environment,
                "method": method,
                "median_tAUC100k": (
                    current["tAUC100k"].median()
                ),
                "median_heldout": (
                    current["heldout_mean"].median()
                ),
                "median_delta_eval": (
                    current["delta_eval"].median()
                ),
                "rho_s": rho,
                "n": len(current),
            })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 11. Print values used in Results
# ============================================================

print(
    "\nTraining-to-held-out summary:\n"
)

print(
    summary[
        [
            "learner",
            "environment",
            "method",
            "median_tAUC100k",
            "median_heldout",
            "median_delta_eval",
            "rho_s",
            "n",
        ]
    ]
    .round(
        {
            "median_tAUC100k": 2,
            "median_heldout": 2,
            "median_delta_eval": 2,
            "rho_s": 3,
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 12. Figure: training-to-held-out relationship
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(12.5, 6.0)
)


for row_idx, learner in enumerate(
    ["PPO", "SAC"]
):

    for col_idx, environment in enumerate(
        ENVIRONMENTS
    ):

        ax = axes[
            row_idx,
            col_idx
        ]

        panel = data[
            (
                data["learner"]
                == learner
            )
            &
            (
                data["environment"]
                == environment
            )
        ].copy()

        # ----------------------------------------------------
        # CODA and PB2 points
        # ----------------------------------------------------

        for method, marker in [
            ("CODA", "o"),
            ("PB2", "s"),
        ]:

            current = panel[
                panel["method"]
                == method
            ]

            ax.scatter(
                current["tAUC100k"],
                current["heldout_mean"],
                marker=marker,
                s=42,
                alpha=0.85,
                label=method,
                zorder=3
            )

        # ----------------------------------------------------
        # Same numerical scale on x and y within each panel
        # ----------------------------------------------------

        all_values = np.concatenate(
            [
                panel[
                    "tAUC100k"
                ].to_numpy(),
                panel[
                    "heldout_mean"
                ].to_numpy(),
            ]
        )

        vmin = np.nanmin(
            all_values
        )

        vmax = np.nanmax(
            all_values
        )

        span = vmax - vmin

        if span == 0:
            span = 1.0

        margin = (
            0.08
            * span
        )

        lower = (
            vmin
            - margin
        )

        upper = (
            vmax
            + margin
        )

        ax.set_xlim(
            lower,
            upper
        )

        ax.set_ylim(
            lower,
            upper
        )

        # ----------------------------------------------------
        # Identity line: G = tAUC_100k
        # ----------------------------------------------------

        ax.plot(
            [
                lower,
                upper,
            ],
            [
                lower,
                upper,
            ],
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            zorder=1
        )

        # ----------------------------------------------------
        # Spearman rho annotation
        # ----------------------------------------------------

        coda_rho = summary.loc[
            (
                summary["learner"]
                == learner
            )
            &
            (
                summary["environment"]
                == environment
            )
            &
            (
                summary["method"]
                == "CODA"
            ),
            "rho_s"
        ].iloc[0]

        pb2_rho = summary.loc[
            (
                summary["learner"]
                == learner
            )
            &
            (
                summary["environment"]
                == environment
            )
            &
            (
                summary["method"]
                == "PB2"
            ),
            "rho_s"
        ].iloc[0]

        annotation = (
            rf"$\rho_S^{{\mathrm{{CODA}}}}"
            rf"={coda_rho:.2f}$"
            "\n"
            rf"$\rho_S^{{\mathrm{{PB2}}}}"
            rf"={pb2_rho:.2f}$"
        )

        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.75,
                edgecolor="none"
            )
        )

        # ----------------------------------------------------
        # Panel titles and axes
        # ----------------------------------------------------

        short_environment = (
            environment
            .replace(
                "-v5",
                ""
            )
        )

        ax.set_title(
            f"{learner}: "
            f"{short_environment}",
            fontsize=10
        )

        if row_idx == 1:

            ax.set_xlabel(
                r"Terminal training "
                r"$tAUC_{100k}$",
                fontsize=9
            )

        if col_idx == 0:

            ax.set_ylabel(
                "Held-out return",
                fontsize=9
            )

        ax.grid(
            linestyle=":",
            alpha=0.25
        )

        ax.tick_params(
            axis="both",
            labelsize=8
        )


# ============================================================
# 13. Shared legend
# ============================================================

handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(
        0.5,
        1.01
    )
)


# ============================================================
# 14. Layout
# ============================================================

fig.tight_layout(
    rect=(
        0,
        0,
        1,
        0.95
    )
)


# ============================================================
# 15. Export
# ============================================================

fig.savefig(
    "training_heldout.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "training_heldout.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()