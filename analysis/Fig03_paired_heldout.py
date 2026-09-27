#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3: Paired held-out effects of CODA relative to PB2.

For each learner (PPO/SAC) and environment, this script:

1. Reads episode-level held-out evaluation results.
2. Computes the mean held-out return for each training seed.
3. Forms matched CODA - PB2 seed-level differences.
4. Reports:
   - median paired difference,
   - percentile 95% paired-bootstrap confidence interval,
   - win/tie/loss counts,
   - two-sided Wilcoxon signed-rank test,
   - Holm-adjusted p-value across the eight primary comparisons,
   - matched-pairs rank-biserial correlation.
5. Generates the paired-effect forest plot used in the manuscript.
6. Saves the numerical results used to construct the figure.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── Fig03_paired_heldout.py
└── results/
    ├── ppo/
    │   └── heldout_reward_ppo_final/
    │       └── heldout_test_episodes.csv
    └── sac/
        └── heldout_reward_sac_final/
            └── heldout_test_episodes.csv

Outputs
-------

results/analysis/paired_heldout/
├── paired_heldout_results.csv
├── paired_heldout.pdf
└── paired_heldout.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


# ============================================================
# Paths
# ============================================================

# Works when this file is stored under:
#     <repo_root>/analysis/Fig03_paired_heldout.py
#
# The fallback also allows execution from an interactive session.
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent


PPO_FILE = (
    REPO_ROOT
    / "results"
    / "ppo"
    / "heldout_reward_ppo_final"
    / "heldout_test_episodes.csv"
)

SAC_FILE = (
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
    / "paired_heldout"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Analysis configuration
# ============================================================

N_BOOT = 10_000
BOOTSTRAP_SEED = 2026

EXPECTED_TRAINING_SEEDS = 10
EXPECTED_TEST_EPISODES = 100

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]


# ============================================================
# Statistical utilities
# ============================================================

def holm_adjust(p_values):
    """
    Holm step-down multiple-comparison adjustment.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.

    Returns
    -------
    np.ndarray
        Holm-adjusted p-values in the original input order.
    """
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, p_value in enumerate(sorted_p):

        adjusted_value = (
            (m - i) * p_value
        )

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[order] = adjusted_sorted

    return adjusted


def rank_biserial_from_differences(
    differences,
):
    """
    Matched-pairs rank-biserial correlation associated with
    the Wilcoxon signed-rank statistic.

    Positive values favor CODA.

    Parameters
    ----------
    differences : array-like
        Matched CODA - PB2 differences.

    Returns
    -------
    float
        Rank-biserial correlation.
    """
    d = np.asarray(
        differences,
        dtype=float,
    )

    d = d[
        np.isfinite(d)
    ]

    # Wilcoxon zero_method="wilcox":
    # zero differences do not contribute to signed ranks.
    d = d[
        d != 0
    ]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)

    # Average ranks handle tied absolute differences.
    ranks = (
        pd.Series(abs_d)
        .rank(method="average")
        .to_numpy()
    )

    w_pos = ranks[
        d > 0
    ].sum()

    w_neg = ranks[
        d < 0
    ].sum()

    denominator = (
        w_pos + w_neg
    )

    if denominator == 0:
        return 0.0

    return float(
        (w_pos - w_neg)
        / denominator
    )


def paired_bootstrap_ci(
    differences,
    n_boot=N_BOOT,
    random_seed=BOOTSTRAP_SEED,
):
    """
    Percentile paired-bootstrap confidence interval for the
    median seed-level CODA - PB2 difference.

    The RNG is reinitialized with the same seed for each
    comparison, reproducing the procedure used for the
    current manuscript figure.

    Parameters
    ----------
    differences : array-like
        Matched seed-level differences.
    n_boot : int
        Number of bootstrap resamples.
    random_seed : int
        RNG seed.

    Returns
    -------
    tuple(float, float)
        Lower and upper 95% percentile-bootstrap limits.
    """
    d = np.asarray(
        differences,
        dtype=float,
    )

    if len(d) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        random_seed
    )

    samples = rng.choice(
        d,
        size=(
            n_boot,
            len(d),
        ),
        replace=True,
    )

    bootstrap_medians = np.median(
        samples,
        axis=1,
    )

    ci_low, ci_high = np.percentile(
        bootstrap_medians,
        [2.5, 97.5],
    )

    return (
        float(ci_low),
        float(ci_high),
    )


# ============================================================
# Input validation and seed-level aggregation
# ============================================================

def load_seed_level_results(
    filename,
):
    """
    Convert episode-level held-out data into one observation
    per method/environment/training seed.

    Each observation is the mean return over the held-out
    evaluation episodes of the training-selected champion.

    Parameters
    ----------
    filename : pathlib.Path
        Episode-level held-out CSV.

    Returns
    -------
    pd.DataFrame
        Seed-level held-out mean returns.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"Held-out file not found:\n{filename}"
        )

    df = pd.read_csv(
        filename
    )

    required_columns = {
        "method",
        "environment",
        "training_seed",
        "test_episode",
        "test_return",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns in "
            f"{filename}:\n"
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Validate held-out episode count
    # --------------------------------------------------------

    episode_counts = (
        df.groupby(
            [
                "method",
                "environment",
                "training_seed",
            ]
        )["test_return"]
        .count()
    )

    invalid_episode_counts = (
        episode_counts[
            episode_counts
            != EXPECTED_TEST_EPISODES
        ]
    )

    if not invalid_episode_counts.empty:

        print(
            "\nWARNING: some selected champions "
            "do not have exactly "
            f"{EXPECTED_TEST_EPISODES} "
            "held-out episodes:"
        )

        print(
            invalid_episode_counts.to_string()
        )

    # --------------------------------------------------------
    # Aggregate held-out episodes within training seed
    # --------------------------------------------------------

    seed_level = (
        df.groupby(
            [
                "method",
                "environment",
                "training_seed",
            ],
            as_index=False,
        )
        .agg(
            heldout_mean=(
                "test_return",
                "mean",
            )
        )
    )

    return seed_level


# ============================================================
# CODA vs PB2 paired analysis
# ============================================================

def compute_coda_pb2_effects(
    seed_level,
    learner,
):
    """
    Compute matched CODA-PB2 held-out effects for each
    environment.

    Parameters
    ----------
    seed_level : pd.DataFrame
        Seed-level held-out means.
    learner : str
        PPO or SAC.

    Returns
    -------
    pd.DataFrame
        One row per environment.
    """
    rows = []

    for environment in ENVIRONMENTS:

        current = seed_level[
            seed_level[
                "environment"
            ].eq(environment)
            &
            seed_level[
                "method"
            ].isin(
                ["CODA", "PB2"]
            )
        ].copy()

        paired = current.pivot(
            index="training_seed",
            columns="method",
            values="heldout_mean",
        )

        required_methods = {
            "CODA",
            "PB2",
        }

        missing_methods = (
            required_methods
            - set(paired.columns)
        )

        if missing_methods:
            raise RuntimeError(
                f"{learner} / {environment}: "
                "missing methods "
                f"{sorted(missing_methods)}"
            )

        paired = (
            paired[
                ["CODA", "PB2"]
            ]
            .dropna()
            .sort_index()
        )

        if len(paired) != EXPECTED_TRAINING_SEEDS:

            print(
                f"WARNING: "
                f"{learner} / {environment}: "
                f"{len(paired)} matched seeds "
                f"instead of "
                f"{EXPECTED_TRAINING_SEEDS}."
            )

        differences = (
            paired["CODA"]
            - paired["PB2"]
        ).to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # Median paired effect
        # ----------------------------------------------------

        delta = float(
            np.median(
                differences
            )
        )

        # ----------------------------------------------------
        # Paired-bootstrap CI
        # ----------------------------------------------------

        ci_low, ci_high = (
            paired_bootstrap_ci(
                differences,
                n_boot=N_BOOT,
                random_seed=BOOTSTRAP_SEED,
            )
        )

        # ----------------------------------------------------
        # Wilcoxon signed-rank test
        # ----------------------------------------------------

        if np.all(
            differences == 0
        ):
            statistic = 0.0
            p_value = 1.0
        else:
            test = wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )

            statistic = float(
                test.statistic
            )

            p_value = float(
                test.pvalue
            )

        # ----------------------------------------------------
        # Direction counts
        # ----------------------------------------------------

        wins = int(
            np.sum(
                differences > 0
            )
        )

        ties = int(
            np.sum(
                differences == 0
            )
        )

        losses = int(
            np.sum(
                differences < 0
            )
        )

        # ----------------------------------------------------
        # Effect size
        # ----------------------------------------------------

        r_rb = (
            rank_biserial_from_differences(
                differences
            )
        )

        rows.append(
            {
                "learner": learner,
                "environment": environment,
                "n_matched_seeds": len(paired),

                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,

                "wins": wins,
                "ties": ties,
                "losses": losses,
                "W_T_L": (
                    f"{wins}/"
                    f"{ties}/"
                    f"{losses}"
                ),

                "wilcoxon_statistic": statistic,
                "p": p_value,

                "r_rb": r_rb,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main analysis
# ============================================================

def main():

    # --------------------------------------------------------
    # Load episode-level results
    # --------------------------------------------------------

    print(
        "Loading PPO held-out evaluations..."
    )

    ppo_seed = (
        load_seed_level_results(
            PPO_FILE
        )
    )

    print(
        "Loading SAC held-out evaluations..."
    )

    sac_seed = (
        load_seed_level_results(
            SAC_FILE
        )
    )

    # --------------------------------------------------------
    # Compute matched effects
    # --------------------------------------------------------

    ppo_results = (
        compute_coda_pb2_effects(
            ppo_seed,
            learner="PPO",
        )
    )

    sac_results = (
        compute_coda_pb2_effects(
            sac_seed,
            learner="SAC",
        )
    )

    results = pd.concat(
        [
            ppo_results,
            sac_results,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Holm correction:
    # one primary family of eight tests
    # --------------------------------------------------------

    results[
        "p_holm"
    ] = holm_adjust(
        results[
            "p"
        ].to_numpy()
    )

    results[
        "significant_holm"
    ] = (
        results[
            "p_holm"
        ]
        < 0.05
    )

    # --------------------------------------------------------
    # Display numerical results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )

    print(
        "CODA vs PB2 paired held-out analysis"
    )

    print(
        "=" * 88
    )

    display_columns = [
        "learner",
        "environment",
        "n_matched_seeds",
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

    print(
        results[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "paired_heldout_results.csv"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # Forest plot
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.4),
        sharey=True,
    )

    for ax, learner in zip(
        axes,
        ["PPO", "SAC"],
    ):

        sub = (
            results[
                results[
                    "learner"
                ].eq(learner)
            ]
            .set_index(
                "environment"
            )
            .loc[
                ENVIRONMENTS
            ]
            .reset_index()
        )

        y = np.arange(
            len(sub)
        )

        # ----------------------------------------------------
        # Zero-effect reference
        # ----------------------------------------------------

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )

        # ----------------------------------------------------
        # Confidence intervals and point estimates
        # ----------------------------------------------------

        for i, row in sub.iterrows():

            ax.hlines(
                y=i,
                xmin=row["ci_low"],
                xmax=row["ci_high"],
                linewidth=2.0,
                zorder=2,
            )

            # CI end caps
            ax.vlines(
                [
                    row["ci_low"],
                    row["ci_high"],
                ],
                ymin=i - 0.08,
                ymax=i + 0.08,
                linewidth=1.2,
                zorder=2,
            )

            # Median paired difference
            if row[
                "significant_holm"
            ]:

                ax.scatter(
                    row["delta"],
                    i,
                    marker="D",
                    s=42,
                    zorder=4,
                )

            else:

                ax.scatter(
                    row["delta"],
                    i,
                    marker="o",
                    s=34,
                    zorder=4,
                )

        # ----------------------------------------------------
        # Axis formatting
        # ----------------------------------------------------

        ax.set_title(
            f"{learner}: CODA $-$ PB2",
            fontsize=10,
        )

        ax.set_xlabel(
            (
                r"Paired held-out effect, "
                r"$\Delta$ return"
            ),
            fontsize=9,
        )

        ax.set_yticks(
            y
        )

        ax.set_yticklabels(
            [
                "HalfCheetah",
                "Hopper",
                "Swimmer",
                "Walker2d",
            ],
            fontsize=9,
        )

        ax.invert_yaxis()

        ax.grid(
            axis="x",
            linestyle=":",
            alpha=0.35,
        )

        ax.tick_params(
            axis="both",
            labelsize=8,
        )

    # ========================================================
    # Annotate Holm-significant PPO results
    # ========================================================

    ppo_significant = results[
        results[
            "learner"
        ].eq("PPO")
        &
        results[
            "significant_holm"
        ]
    ]

    for _, row in (
        ppo_significant.iterrows()
    ):

        environment_position = (
            ENVIRONMENTS.index(
                row[
                    "environment"
                ]
            )
        )

        axes[0].annotate(
            r"$p_{\mathrm{Holm}}<0.05$",
            xy=(
                row["delta"],
                environment_position,
            ),
            xytext=(
                7,
                -12,
            ),
            textcoords=(
                "offset points"
            ),
            fontsize=7.5,
        )

    # ========================================================
    # Layout and export
    # ========================================================

    fig.tight_layout()

    pdf_path = (
        OUTPUT_DIR
        / "paired_heldout.pdf"
    )

    png_path = (
        OUTPUT_DIR
        / "paired_heldout.png"
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

    # --------------------------------------------------------
    # Report generated files
    # --------------------------------------------------------

    print(
        "\nGenerated outputs:"
    )

    print(
        f"  CSV : {csv_path}"
    )

    print(
        f"  PDF : {pdf_path}"
    )

    print(
        f"  PNG : {png_path}"
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