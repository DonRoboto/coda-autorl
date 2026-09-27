#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:12:54 2026

@author: yor5
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = "diagnostic_surrogate_results.csv"
PRIMARY_CUTOFF = 0.70

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

SEPARATOR_Y = 3.5  # horizontal separator between PPO and SAC

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(RESULTS_FILE)

required_columns = {
    "learner",
    "environment",
    "seed",
    "cutoff",
    "delta_fixed",
    "delta_refit",
    "delta_shuffled",
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# Keep primary 70% cutoff
# ============================================================

primary = df[
    np.isclose(df["cutoff"].astype(float), PRIMARY_CUTOFF)
].copy()

# ============================================================
# Validate seeds
# ============================================================

print("\nNumber of seeds available:\n")
seed_counts = (
    primary
    .groupby(["learner", "environment"])["seed"]
    .nunique()
)
print(seed_counts)

if not (seed_counts == 10).all():
    print(
        "\nWARNING: at least one learner/environment "
        "does not contain exactly 10 seeds."
    )

# ============================================================
# Summary function
# ============================================================

def summarize_metric(df, metric):
    rows = []

    for learner, environment in CONDITIONS:
        current = df[
            (df["learner"] == learner) &
            (df["environment"] == environment)
        ][metric].dropna()

        if len(current) == 0:
            raise ValueError(
                f"No values found for {learner}, {environment}, {metric}"
            )

        rows.append({
            "learner": learner,
            "environment": environment,
            "n": len(current),
            "median": current.median(),
            "q1": current.quantile(0.25),
            "q3": current.quantile(0.75),
        })

    return pd.DataFrame(rows)

fixed = summarize_metric(primary, "delta_fixed")
refit = summarize_metric(primary, "delta_refit")
shuffled = summarize_metric(primary, "delta_shuffled")

# ============================================================
# Print numerical results
# ============================================================

print("\nFixed-kernel results:")
print(fixed.round(4).to_string(index=False))

print("\nIndependent-refit results:")
print(refit.round(4).to_string(index=False))

print("\nShuffled-diagnostic results:")
print(shuffled.round(4).to_string(index=False))

# ============================================================
# Helper to draw one forest panel
# ============================================================

def draw_panel(
    ax,
    summary,
    title,
    left_label,
    right_label,
    show_ylabels=False
):
    y = np.arange(len(summary))

    med = summary["median"].to_numpy()
    q1 = summary["q1"].to_numpy()
    q3 = summary["q3"].to_numpy()

    xerr = np.vstack([
        med - q1,
        q3 - med,
    ])

    # Reference line at zero
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        zorder=1
    )

    # Horizontal separator between PPO and SAC
    ax.axhline(
        SEPARATOR_Y,
        color="gray",
        linestyle="-",
        linewidth=0.8,
        alpha=0.7,
        zorder=1
    )

    # Median + IQR
    ax.errorbar(
        med,
        y,
        xerr=xerr,
        fmt="o",
        markersize=5,
        capsize=3,
        linewidth=1.4,
        zorder=3
    )

    ax.set_title(title, fontsize=10)
    ax.set_yticks(y)

    if show_ylabels:
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.tick_params(axis="y", length=0, pad=5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.30
    )

    ax.tick_params(axis="x", labelsize=8)

    # Direction labels
    ax.text(
        0.02,
        -0.18,
        left_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8
    )

    ax.text(
        0.98,
        -0.18,
        right_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8
    )

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(11.5, 4.5),
    sharey=True
)

# Panel A: fixed kernel
draw_panel(
    axes[0],
    fixed,
    title="(a) Fixed kernel",
    left_label=r"no $S$ better",
    right_label=r"real $S$ better",
    show_ylabels=True
)

# Panel B: independent refit
draw_panel(
    axes[1],
    refit,
    title="(b) Independent refit",
    left_label=r"no $S$ better",
    right_label=r"real $S$ better",
    show_ylabels=True
)

# Panel C: shuffled diagnostic
draw_panel(
    axes[2],
    shuffled,
    title="(c) Shuffled diagnostic",
    left_label=r"shuffled $S$ better",
    right_label=r"real $S$ better",
    show_ylabels=True
)

# ============================================================
# Use a common symmetric x-axis
# ============================================================

all_extremes = np.concatenate([
    fixed["q1"].to_numpy(),
    fixed["q3"].to_numpy(),
    refit["q1"].to_numpy(),
    refit["q3"].to_numpy(),
    shuffled["q1"].to_numpy(),
    shuffled["q3"].to_numpy(),
])

limit = np.nanmax(np.abs(all_extremes))
limit *= 1.10

for ax in axes:
    ax.set_xlim(-limit, limit)

# ============================================================
# Shared x label
# ============================================================

#fig.supxlabel(
#    r"Paired prediction-error difference, $\Delta$RMSE",
#    fontsize=10,
#    y=0.04
#)

# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.28,   # important: room for y-axis labels
    right=0.98,
    bottom=0.22,
    top=0.88,
    wspace=0.18
)

# ============================================================
# Export
# ============================================================

fig.savefig(
    "diagnostic_surrogate.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "diagnostic_surrogate.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # ============================================================
# # Configuration
# # ============================================================

# RESULTS_FILE = "diagnostic_surrogate_results.csv"

# PRIMARY_CUTOFF = 0.70

# ENVIRONMENTS = [
#     "HalfCheetah-v5",
#     "Hopper-v5",
#     "Swimmer-v5",
#     "Walker2d-v5",
# ]

# LEARNERS = [
#     "PPO",
#     "SAC",
# ]


# # Desired vertical order
# CONDITIONS = [
#     ("PPO", "HalfCheetah-v5"),
#     ("PPO", "Hopper-v5"),
#     ("PPO", "Swimmer-v5"),
#     ("PPO", "Walker2d-v5"),
#     ("SAC", "HalfCheetah-v5"),
#     ("SAC", "Hopper-v5"),
#     ("SAC", "Swimmer-v5"),
#     ("SAC", "Walker2d-v5"),
# ]


# LABELS = [
#     "PPO HalfCheetah",
#     "PPO Hopper",
#     "PPO Swimmer",
#     "PPO Walker2d",
#     "SAC HalfCheetah",
#     "SAC Hopper",
#     "SAC Swimmer",
#     "SAC Walker2d",
# ]


# # ============================================================
# # Load data
# # ============================================================

# df = pd.read_csv(RESULTS_FILE)

# required_columns = {
#     "learner",
#     "environment",
#     "seed",
#     "cutoff",
#     "delta_fixed",
#     "delta_refit",
#     "delta_shuffled",
# }

# missing = required_columns - set(df.columns)

# if missing:
#     raise ValueError(
#         f"Missing required columns: {missing}"
#     )


# # ============================================================
# # Keep primary 70% cutoff
# # ============================================================

# primary = df[
#     np.isclose(
#         df["cutoff"].astype(float),
#         PRIMARY_CUTOFF
#     )
# ].copy()


# # ============================================================
# # Validate seeds
# # ============================================================

# print("\nNumber of seeds available:\n")

# seed_counts = (
#     primary
#     .groupby(
#         ["learner", "environment"]
#     )["seed"]
#     .nunique()
# )

# print(seed_counts)

# if not (seed_counts == 10).all():
#     print(
#         "\nWARNING: at least one learner/environment "
#         "does not contain exactly 10 seeds."
#     )


# # ============================================================
# # Summary function
# # ============================================================

# def summarize_metric(df, metric):
#     """
#     Return median, Q1 and Q3 for each learner/environment.
#     """

#     rows = []

#     for learner, environment in CONDITIONS:

#         current = df[
#             (df["learner"] == learner)
#             &
#             (df["environment"] == environment)
#         ][metric].dropna()

#         if len(current) == 0:
#             raise ValueError(
#                 f"No values found for {learner}, "
#                 f"{environment}, {metric}"
#             )

#         rows.append({
#             "learner": learner,
#             "environment": environment,
#             "n": len(current),
#             "median": current.median(),
#             "q1": current.quantile(0.25),
#             "q3": current.quantile(0.75),
#         })

#     return pd.DataFrame(rows)


# fixed = summarize_metric(
#     primary,
#     "delta_fixed"
# )

# refit = summarize_metric(
#     primary,
#     "delta_refit"
# )

# shuffled = summarize_metric(
#     primary,
#     "delta_shuffled"
# )


# # ============================================================
# # Print numerical results
# # ============================================================

# print("\nFixed-kernel results:")
# print(
#     fixed.round(4).to_string(index=False)
# )

# print("\nIndependent-refit results:")
# print(
#     refit.round(4).to_string(index=False)
# )

# print("\nShuffled-diagnostic results:")
# print(
#     shuffled.round(4).to_string(index=False)
# )


# # ============================================================
# # Helper to draw one forest panel
# # ============================================================

# def draw_panel(
#     ax,
#     summary,
#     title,
#     left_label,
#     right_label,
#     show_ylabels=False
# ):

#     y = np.arange(
#         len(summary)
#     )

#     med = summary[
#         "median"
#     ].to_numpy()

#     q1 = summary[
#         "q1"
#     ].to_numpy()

#     q3 = summary[
#         "q3"
#     ].to_numpy()

#     # Asymmetric errors around the median
#     xerr = np.vstack([
#         med - q1,
#         q3 - med,
#     ])

#     # --------------------------------------------------------
#     # Reference line: Delta RMSE = 0
#     # --------------------------------------------------------

#     ax.axvline(
#         0,
#         linestyle="--",
#         linewidth=1.0,
#         alpha=0.75,
#         zorder=1
#     )

#     # --------------------------------------------------------
#     # Median + IQR
#     # --------------------------------------------------------

#     ax.errorbar(
#         med,
#         y,
#         xerr=xerr,
#         fmt="o",
#         markersize=5,
#         capsize=3,
#         linewidth=1.4,
#         zorder=3
#     )

#     # --------------------------------------------------------
#     # Formatting
#     # --------------------------------------------------------

#     ax.set_title(
#         title,
#         fontsize=10
#     )

#     ax.set_yticks(
#         y
#     )

#     if show_ylabels:

#         ax.set_yticklabels(
#             LABELS,
#             fontsize=8
#         )

#     else:

#         ax.set_yticklabels(
#             []
#         )

#     ax.invert_yaxis()

#     ax.grid(
#         axis="x",
#         linestyle=":",
#         alpha=0.30
#     )

#     ax.tick_params(
#         axis="x",
#         labelsize=8
#     )

#     # Labels describing direction
#     ax.text(
#         0.02,
#         -0.16,
#         left_label,
#         transform=ax.transAxes,
#         ha="left",
#         va="top",
#         fontsize=8
#     )

#     ax.text(
#         0.98,
#         -0.16,
#         right_label,
#         transform=ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=8
#     )


# # ============================================================
# # Figure
# # ============================================================

# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=3,
#     figsize=(10.5, 4.3),
#     sharey=True
# )


# # ------------------------------------------------------------
# # Panel A: fixed kernel
# # ------------------------------------------------------------

# draw_panel(
#     axes[0],
#     fixed,
#     title="(a) Fixed kernel",
#     left_label=r"no $S$ better",
#     right_label=r"real $S$ better",
#     show_ylabels=True
# )


# # ------------------------------------------------------------
# # Panel B: independent refit
# # ------------------------------------------------------------

# draw_panel(
#     axes[1],
#     refit,
#     title="(b) Independent refit",
#     left_label=r"no $S$ better",
#     right_label=r"real $S$ better",
#     show_ylabels=False
# )


# # ------------------------------------------------------------
# # Panel C: shuffled diagnostic
# # ------------------------------------------------------------

# draw_panel(
#     axes[2],
#     shuffled,
#     title="(c) Shuffled diagnostic",
#     left_label=r"shuffled $S$ better",
#     right_label=r"real $S$ better",
#     show_ylabels=False
# )


# # ============================================================
# # Use a common symmetric x-axis
# # ============================================================

# all_extremes = np.concatenate([
#     fixed["q1"].to_numpy(),
#     fixed["q3"].to_numpy(),

#     refit["q1"].to_numpy(),
#     refit["q3"].to_numpy(),

#     shuffled["q1"].to_numpy(),
#     shuffled["q3"].to_numpy(),
# ])

# limit = np.nanmax(
#     np.abs(all_extremes)
# )

# # Small margin
# limit *= 1.10

# for ax in axes:
#     ax.set_xlim(
#         -limit,
#         limit
#     )


# # ============================================================
# # Shared x label
# # ============================================================

# fig.supxlabel(
#     r"Paired prediction-error difference, $\Delta$RMSE",
#     fontsize=10,
#     y=0.02
# )


# # ============================================================
# # Layout
# # ============================================================

# fig.tight_layout(
#     rect=(
#         0,
#         0.08,
#         1,
#         1
#     ),
#     w_pad=1.0
# )


# # ============================================================
# # Export
# # ============================================================

# fig.savefig(
#     "diagnostic_surrogate.pdf",
#     bbox_inches="tight"
# )

# fig.savefig(
#     "diagnostic_surrogate.png",
#     dpi=600,
#     bbox_inches="tight"
# )

# plt.show()