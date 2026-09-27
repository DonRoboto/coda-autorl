#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:15:35 2026

@author: yor5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon


# ============================================================
# Configuration
# ============================================================

PPO_FILE = "../results/ppo/heldout_reward_ppo_final/heldout_test_episodes.csv"
SAC_FILE = "../results/sac/heldout_reward_sac_final/heldout_test_episodes.csv"

N_BOOT = 10_000
BOOTSTRAP_SEED = 2026

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]


# ============================================================
# Utilities
# ============================================================

def holm_adjust(p_values):
    """
    Holm step-down adjustment.
    Returns adjusted p-values in the original order.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)

    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adjusted = (m - i) * p
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    return adjusted


def rank_biserial_from_differences(differences):
    """
    Matched-pairs rank-biserial correlation associated with
    the Wilcoxon signed-rank statistic.

    Positive values favor CODA.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)

    # Average ranks, including ties
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()

    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()

    denom = w_pos + w_neg

    return (w_pos - w_neg) / denom if denom > 0 else 0.0


def paired_bootstrap_ci(
    differences,
    n_boot=10_000,
    random_seed=2026
):
    """
    Percentile paired-bootstrap CI for the median seed-level
    CODA - PB2 difference.

    Reinitializing the RNG with the same seed for each comparison
    reproduces the intervals used in the current results table.
    """
    d = np.asarray(differences, dtype=float)

    rng = np.random.default_rng(random_seed)

    samples = rng.choice(
        d,
        size=(n_boot, len(d)),
        replace=True
    )

    bootstrap_medians = np.median(samples, axis=1)

    return np.percentile(
        bootstrap_medians,
        [2.5, 97.5]
    )


# ============================================================
# Seed-level held-out performance
# ============================================================

def load_seed_level_results(filename):
    """
    Converts episode-level held-out data into one observation
    per method/environment/training seed.

    Each observation is the mean return over the 100 held-out
    evaluation episodes of the selected champion.
    """
    df = pd.read_csv(filename)

    required_columns = {
        "method",
        "environment",
        "training_seed",
        "test_episode",
        "test_return",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {filename}: {missing}"
        )

    # Optional validation of evaluation protocol
    episode_counts = (
        df.groupby(
            ["method", "environment", "training_seed"]
        )["test_return"]
        .count()
    )

    if not (episode_counts == 100).all():
        print(
            f"WARNING: not every selected champion in "
            f"{filename} has exactly 100 held-out episodes."
        )

    seed_level = (
        df.groupby(
            ["method", "environment", "training_seed"],
            as_index=False
        )
        .agg(
            heldout_mean=("test_return", "mean")
        )
    )

    return seed_level


# ============================================================
# CODA vs PB2 paired analysis
# ============================================================

def compute_coda_pb2_effects(seed_level, learner):
    rows = []

    for env in ENVIRONMENTS:

        current = seed_level[
            seed_level["environment"].eq(env)
            & seed_level["method"].isin(["CODA", "PB2"])
        ]

        paired = current.pivot(
            index="training_seed",
            columns="method",
            values="heldout_mean"
        )

        paired = paired.dropna(
            subset=["CODA", "PB2"]
        )

        if len(paired) != 10:
            print(
                f"WARNING: {learner} {env}: "
                f"{len(paired)} matched seeds instead of 10."
            )

        differences = (
            paired["CODA"] - paired["PB2"]
        ).to_numpy()

        delta = np.median(differences)

        ci_low, ci_high = paired_bootstrap_ci(
            differences,
            n_boot=N_BOOT,
            random_seed=BOOTSTRAP_SEED
        )

        stat = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox"
        )

        wins = np.sum(differences > 0)
        ties = np.sum(differences == 0)
        losses = np.sum(differences < 0)

        r_rb = rank_biserial_from_differences(
            differences
        )

        rows.append({
            "learner": learner,
            "environment": env,
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "p": stat.pvalue,
            "r_rb": r_rb,
        })

    return pd.DataFrame(rows)


# ============================================================
# Load PPO and SAC
# ============================================================

ppo_seed = load_seed_level_results(PPO_FILE)
sac_seed = load_seed_level_results(SAC_FILE)

ppo_results = compute_coda_pb2_effects(
    ppo_seed,
    learner="PPO"
)

sac_results = compute_coda_pb2_effects(
    sac_seed,
    learner="SAC"
)

results = pd.concat(
    [ppo_results, sac_results],
    ignore_index=True
)


# ============================================================
# Holm correction over the complete primary family of 8 tests
# ============================================================

results["p_holm"] = holm_adjust(
    results["p"].to_numpy()
)

results["significant_holm"] = (
    results["p_holm"] < 0.05
)


# ============================================================
# Print results used in the figure
# ============================================================

print(
    results[
        [
            "learner",
            "environment",
            "delta",
            "ci_low",
            "ci_high",
            "wins",
            "ties",
            "losses",
            "p",
            "p_holm",
            "r_rb",
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# Forest plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.0, 3.4),
    sharey=True
)

for ax, learner in zip(
    axes,
    ["PPO", "SAC"]
):

    sub = (
        results[
            results["learner"] == learner
        ]
        .set_index("environment")
        .loc[ENVIRONMENTS]
        .reset_index()
    )

    y = np.arange(len(sub))

    # Zero-effect reference
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        zorder=1
    )

    # Bootstrap confidence intervals
    for i, row in sub.iterrows():

        ax.hlines(
            y=i,
            xmin=row["ci_low"],
            xmax=row["ci_high"],
            linewidth=2.0,
            zorder=2
        )

        # End caps
        ax.vlines(
            [row["ci_low"], row["ci_high"]],
            ymin=i - 0.08,
            ymax=i + 0.08,
            linewidth=1.2,
            zorder=2
        )

        # Median paired difference
        if row["significant_holm"]:
            ax.scatter(
                row["delta"],
                i,
                marker="D",
                s=42,
                zorder=4
            )
        else:
            ax.scatter(
                row["delta"],
                i,
                marker="o",
                s=34,
                zorder=4
            )

    ax.set_title(
        f"{learner}: CODA $-$ PB2",
        fontsize=10
    )

    ax.set_xlabel(
        r"Paired held-out effect, $\Delta$ return",
        fontsize=9
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        [
            "HalfCheetah",
            "Hopper",
            "Swimmer",
            "Walker2d",
        ],
        fontsize=9
    )

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.35
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )


# ============================================================
# Optional annotation of Holm-significant result
# ============================================================

ppo_sig = results[
    (results["learner"] == "PPO")
    & results["significant_holm"]
]

for _, row in ppo_sig.iterrows():

    env_position = ENVIRONMENTS.index(
        row["environment"]
    )

    axes[0].annotate(
        r"$p_{\mathrm{Holm}}<0.05$",
        xy=(
            row["delta"],
            env_position
        ),
        xytext=(7, -12),
        textcoords="offset points",
        fontsize=7.5
    )


# ============================================================
# Layout and export
# ============================================================

fig.tight_layout()

fig.savefig(
    "paired_heldout.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "paired_heldout.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()