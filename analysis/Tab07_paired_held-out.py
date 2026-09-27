#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table 7: Paired held-out directional ablation effects for CODA.

For each learner (PPO/SAC), environment, and one-channel CODA variant,
this script:

1. Reads episode-level held-out evaluation results.
2. Normalizes supported method aliases.
3. Validates the held-out evaluation protocol.
4. Computes one held-out mean return per training seed.
5. Forms matched seed-level directional contrasts:

       Full - I2O = CODA - CODA-I2O
       Full - O2I = CODA - CODA-O2I

6. Reports the median paired difference, percentile 95% paired-bootstrap
   confidence interval, win/tie/loss counts, two-sided Wilcoxon signed-rank
   test, Holm-adjusted p-value over the complete family of 16 directional
   comparisons, and matched-pairs rank-biserial correlation.
7. Saves auditable intermediate and final numerical artifacts.
8. Generates the LaTeX table body used in the manuscript.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── Tab07_paired_held-out.py
└── results/
    ├── ppo/
    │   └── heldout_reward_ppo_final/
    │       └── heldout_test_episodes.csv
    └── sac/
        └── heldout_reward_sac_final/
            └── heldout_test_episodes.csv

Outputs
-------

results/analysis/directional_ablation/
├── ablation_seed_level_heldout.csv
├── ablation_seed_counts.csv
├── ablation_paired_differences.csv
├── ablation_directional_stats.csv
└── ablation_heldout_table_body.tex
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
    / "directional_ablation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SEED_LEVEL = OUTPUT_DIR / "ablation_seed_level_heldout.csv"
OUTPUT_SEED_COUNTS = OUTPUT_DIR / "ablation_seed_counts.csv"
OUTPUT_PAIRED = OUTPUT_DIR / "ablation_paired_differences.csv"
OUTPUT_STATS = OUTPUT_DIR / "ablation_directional_stats.csv"
OUTPUT_LATEX = OUTPUT_DIR / "ablation_heldout_table_body.tex"


# ============================================================
# Analysis configuration
# ============================================================

N_TEST_EPISODES = 100
N_TRAINING_SEEDS = 10

N_BOOT = 10_000
BOOT_SEED = 20260913

LEARNERS = ["PPO", "SAC"]

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

METHODS = [
    "CODA",
    "CODA-I2O",
    "CODA-O2I",
]

CONTRASTS = [
    ("CODA-I2O", "Full $-$ I2O"),
    ("CODA-O2I", "Full $-$ O2I"),
]

METHOD_ALIASES = {
    "CODA_FULL": "CODA",
    "CODA-FULL": "CODA",
    "CODA_I2O": "CODA-I2O",
    "CODA_O2I": "CODA-O2I",
}


# ============================================================
# Input loading and validation
# ============================================================

def load_heldout(path, learner):
    """Load one episode-level held-out evaluation file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Held-out evaluation file not found:\n"
            f"{path}"
        )

    df = pd.read_csv(path)

    required = {
        "method",
        "environment",
        "training_seed",
        "test_return",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()
    df["learner"] = learner

    df["training_seed"] = pd.to_numeric(
        df["training_seed"],
        errors="raise",
    ).astype(int)

    df["test_return"] = pd.to_numeric(
        df["test_return"],
        errors="raise",
    )

    if not np.isfinite(df["test_return"]).all():
        raise ValueError(
            f"{path.name} contains non-finite test_return values."
        )

    return df


def normalize_methods(episodes):
    """Normalize supported aliases and keep directional-ablation methods."""
    episodes = episodes.copy()
    episodes["method"] = (
        episodes["method"]
        .astype(str)
        .replace(METHOD_ALIASES)
    )

    print("\nMethods present after alias normalization:")
    print(sorted(episodes["method"].unique()))

    episodes = episodes[
        episodes["method"].isin(METHODS)
    ].copy()

    missing_methods = set(METHODS) - set(episodes["method"].unique())
    if missing_methods:
        raise ValueError(
            "Missing directional-ablation methods: "
            f"{sorted(missing_methods)}"
        )

    return episodes


def validate_episode_level_data(episodes):
    """Validate the held-out evaluation protocol."""
    case_keys = [
        "learner",
        "method",
        "environment",
        "training_seed",
    ]

    episode_counts = (
        episodes
        .groupby(case_keys, as_index=False)
        .agg(n_test_episodes=("test_return", "size"))
    )

    bad_episode_counts = episode_counts[
        episode_counts["n_test_episodes"] != N_TEST_EPISODES
    ]

    if not bad_episode_counts.empty:
        print("\nCases with incorrect held-out episode count:")
        print(bad_episode_counts.to_string(index=False))
        raise RuntimeError(
            "Every case must contain exactly "
            f"{N_TEST_EPISODES} held-out episodes."
        )

    if "test_seed" in episodes.columns:
        unique_test_seeds = (
            episodes
            .groupby(case_keys)["test_seed"]
            .nunique()
        )
        bad = unique_test_seeds[
            unique_test_seeds != N_TEST_EPISODES
        ]
        if not bad.empty:
            print("\nCases with duplicated or missing test seeds:")
            print(bad)
            raise RuntimeError(
                "Held-out test-seed validation failed."
            )

    if "champion_agent" in episodes.columns:
        champion_counts = (
            episodes
            .groupby(case_keys)["champion_agent"]
            .nunique()
        )
        bad = champion_counts[champion_counts != 1]
        if not bad.empty:
            print("\nCases containing more than one champion:")
            print(bad)
            raise RuntimeError(
                "Champion identity is not unique."
            )

    if "explore" in episodes.columns:
        explore_values = (
            episodes["explore"]
            .astype(str)
            .str.lower()
            .str.strip()
        )
        if not explore_values.isin(["false", "0"]).all():
            raise RuntimeError(
                "At least one episode was not evaluated with explore=False."
            )


# ============================================================
# Seed-level held-out outcomes
# ============================================================

def compute_seed_level_results(episodes):
    """Compute one held-out mean return per training seed."""
    return (
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
            n_test_episodes=("test_return", "size"),
            test_mean_return=("test_return", "mean"),
        )
    )


def validate_seed_counts(seed_level):
    """Require ten training seeds in every expected cell."""
    seed_counts = (
        seed_level
        .groupby(
            ["learner", "method", "environment"],
            as_index=False,
        )
        .agg(n_training_seeds=("training_seed", "nunique"))
    )

    expected = pd.DataFrame(
        [
            (learner, method, environment)
            for learner in LEARNERS
            for method in METHODS
            for environment in ENVIRONMENTS
        ],
        columns=["learner", "method", "environment"],
    )

    checked = expected.merge(
        seed_counts,
        on=["learner", "method", "environment"],
        how="left",
        validate="one_to_one",
    )
    checked["n_training_seeds"] = (
        checked["n_training_seeds"]
        .fillna(0)
        .astype(int)
    )

    bad = checked[
        checked["n_training_seeds"] != N_TRAINING_SEEDS
    ]
    if not bad.empty:
        print("\nIncomplete directional-ablation cells:")
        print(bad.to_string(index=False))
        raise RuntimeError(
            "Expected exactly "
            f"{N_TRAINING_SEEDS} training seeds per cell."
        )

    return checked


def validate_common_test_seed_sets(episodes, seed_level):
    """Verify identical test-seed sets across CODA variants when available."""
    if "test_seed" not in episodes.columns:
        return

    for learner in LEARNERS:
        for environment in ENVIRONMENTS:
            training_seeds = sorted(
                seed_level[
                    seed_level["learner"].eq(learner)
                    & seed_level["environment"].eq(environment)
                ]["training_seed"].unique()
            )

            for training_seed in training_seeds:
                reference = None

                for method in METHODS:
                    current = set(
                        episodes[
                            episodes["learner"].eq(learner)
                            & episodes["environment"].eq(environment)
                            & episodes["training_seed"].eq(training_seed)
                            & episodes["method"].eq(method)
                        ]["test_seed"].astype(int)
                    )

                    if reference is None:
                        reference = current
                    elif current != reference:
                        raise RuntimeError(
                            "Common held-out test seeds do not match for:\n"
                            f"{learner}, {environment}, "
                            f"training_seed={training_seed}"
                        )


# ============================================================
# Matched directional contrasts
# ============================================================

def build_paired_data(
    seed_level,
    learner,
    environment,
    variant,
    contrast_label,
):
    """Build matched full-CODA minus one-channel-variant differences."""
    full = (
        seed_level[
            seed_level["learner"].eq(learner)
            & seed_level["environment"].eq(environment)
            & seed_level["method"].eq("CODA")
        ][["training_seed", "test_mean_return"]]
        .rename(columns={"test_mean_return": "full_coda"})
    )

    variant_df = (
        seed_level[
            seed_level["learner"].eq(learner)
            & seed_level["environment"].eq(environment)
            & seed_level["method"].eq(variant)
        ][["training_seed", "test_mean_return"]]
        .rename(columns={"test_mean_return": "variant"})
    )

    paired = (
        full
        .merge(
            variant_df,
            on="training_seed",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("training_seed")
        .reset_index(drop=True)
    )

    if len(paired) != N_TRAINING_SEEDS:
        raise RuntimeError(
            f"{learner}, {environment}, {variant}: "
            f"expected {N_TRAINING_SEEDS} matched seeds, "
            f"found {len(paired)}."
        )

    paired["difference"] = paired["full_coda"] - paired["variant"]
    paired.insert(0, "contrast", contrast_label)
    paired.insert(0, "variant_name", variant)
    paired.insert(0, "environment", environment)
    paired.insert(0, "learner", learner)

    return paired[
        [
            "learner",
            "environment",
            "variant_name",
            "contrast",
            "training_seed",
            "full_coda",
            "variant",
            "difference",
        ]
    ]


def build_all_paired_differences(seed_level):
    """Build the complete family of 16 matched directional contrasts."""
    frames = []

    for learner in LEARNERS:
        for environment in ENVIRONMENTS:
            for variant, contrast_label in CONTRASTS:
                frames.append(
                    build_paired_data(
                        seed_level,
                        learner,
                        environment,
                        variant,
                        contrast_label,
                    )
                )

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Statistical utilities
# ============================================================

def rank_biserial(differences):
    """Matched-pairs rank-biserial correlation."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    d_nz = d[d != 0]

    if len(d_nz) == 0:
        return np.nan

    ranks = rankdata(
        np.abs(d_nz),
        method="average",
    )

    w_plus = ranks[d_nz > 0].sum()
    w_minus = ranks[d_nz < 0].sum()
    denominator = w_plus + w_minus

    if denominator == 0:
        return np.nan

    return float((w_plus - w_minus) / denominator)


def exact_wilcoxon(differences):
    """
    Exact two-sided Wilcoxon after explicit removal of zero differences.

    This preserves the convention used by the current Table 7 results.
    """
    d = np.asarray(differences, dtype=float)
    d_nz = d[d != 0]

    if len(d_nz) == 0:
        return 0.0, 1.0

    test = wilcoxon(
        d_nz,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="exact",
    )

    return float(test.statistic), float(test.pvalue)


# Same matched-seed bootstrap resamples for all 16 contrasts.
rng = np.random.default_rng(BOOT_SEED)
boot_idx = rng.integers(
    low=0,
    high=N_TRAINING_SEEDS,
    size=(N_BOOT, N_TRAINING_SEEDS),
)


def bootstrap_median_ci(differences):
    """Percentile paired-bootstrap CI for the median directional effect."""
    d = np.asarray(differences, dtype=float)

    if len(d) != N_TRAINING_SEEDS:
        raise ValueError(
            "Expected exactly "
            f"{N_TRAINING_SEEDS} paired differences."
        )

    boot_medians = np.median(d[boot_idx], axis=1)
    ci_low, ci_high = np.quantile(
        boot_medians,
        [0.025, 0.975],
    )

    return (
        float(np.median(d)),
        float(ci_low),
        float(ci_high),
    )


# ============================================================
# Directional-ablation statistics
# ============================================================

def compute_directional_stats(paired_all):
    """Compute the complete family of 16 directional comparisons."""
    rows = []

    for learner in LEARNERS:
        for environment in ENVIRONMENTS:
            for variant, contrast_label in CONTRASTS:
                current = (
                    paired_all[
                        paired_all["learner"].eq(learner)
                        & paired_all["environment"].eq(environment)
                        & paired_all["variant_name"].eq(variant)
                    ]
                    .sort_values("training_seed")
                )

                d = current["difference"].to_numpy(dtype=float)

                wins = int(np.sum(d > 0))
                ties = int(np.sum(d == 0))
                losses = int(np.sum(d < 0))

                median_diff, ci_low, ci_high = bootstrap_median_ci(d)
                wilcoxon_W, p_raw = exact_wilcoxon(d)
                r_rb = rank_biserial(d)

                rows.append(
                    {
                        "learner": learner,
                        "environment": environment,
                        "variant": variant,
                        "contrast": contrast_label,
                        "n_matched_seeds": len(d),
                        "median_difference": median_diff,
                        "bootstrap_95_lo": ci_low,
                        "bootstrap_95_hi": ci_high,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "W_T_L": f"{wins}/{ties}/{losses}",
                        "wilcoxon_W": wilcoxon_W,
                        "wilcoxon_method": "exact_after_zero_removal",
                        "p_raw": p_raw,
                        "r_rb": r_rb,
                    }
                )

    stats = pd.DataFrame(rows)

    stats["p_Holm"] = multipletests(
        stats["p_raw"].to_numpy(),
        alpha=0.05,
        method="holm",
    )[1]

    stats["significant_Holm"] = stats["p_Holm"] < 0.05

    learner_order = {"PPO": 0, "SAC": 1}
    environment_order = {
        env: i
        for i, env in enumerate(ENVIRONMENTS)
    }
    variant_order = {
        "CODA-I2O": 0,
        "CODA-O2I": 1,
    }

    stats["_learner_order"] = stats["learner"].map(learner_order)
    stats["_environment_order"] = stats["environment"].map(environment_order)
    stats["_variant_order"] = stats["variant"].map(variant_order)

    stats = (
        stats
        .sort_values(
            [
                "_learner_order",
                "_environment_order",
                "_variant_order",
            ]
        )
        .drop(
            columns=[
                "_learner_order",
                "_environment_order",
                "_variant_order",
            ]
        )
        .reset_index(drop=True)
    )

    return stats


# ============================================================
# LaTeX formatting
# ============================================================

def format_effect(row):
    return (
        f"${row['median_difference']:.2f} "
        f"[{row['bootstrap_95_lo']:.2f}, "
        f"{row['bootstrap_95_hi']:.2f}]$"
    )


def format_p(value):
    return f"{value:.3f}"


def format_rrb(value):
    return f"{value:.2f}"


def build_latex_body(stats):
    """Build the body of manuscript Table 7."""
    lines = []
    previous_learner = None

    for _, row in stats.iterrows():
        learner = row["learner"]

        if (
            previous_learner is not None
            and learner != previous_learner
        ):
            lines.append(r"\midrule")
            lines.append("")

        latex_row = (
            f"{learner} & "
            f"{row['environment']} &\n"
            f"{row['contrast']} &\n"
            f"{format_effect(row)} &\n"
            f"{row['W_T_L']} & "
            f"{format_p(row['p_raw'])} & "
            f"{format_p(row['p_Holm'])} & "
            f"{format_rrb(row['r_rb'])} "
            r"\\"
        )

        lines.append(latex_row)
        lines.append("")
        previous_learner = learner

    return "\n".join(lines).rstrip()


# ============================================================
# Main
# ============================================================

def main():
    print("Loading PPO held-out evaluations...")
    ppo = load_heldout(PPO_HELDOUT, learner="PPO")

    print("Loading SAC held-out evaluations...")
    sac = load_heldout(SAC_HELDOUT, learner="SAC")

    episodes = pd.concat(
        [ppo, sac],
        ignore_index=True,
    )

    episodes = normalize_methods(episodes)
    validate_episode_level_data(episodes)

    seed_level = compute_seed_level_results(episodes)
    seed_counts = validate_seed_counts(seed_level)
    validate_common_test_seed_sets(episodes, seed_level)

    print("\nTraining seeds per learner/method/environment:")
    print(seed_counts.to_string(index=False))

    seed_level.to_csv(
        OUTPUT_SEED_LEVEL,
        index=False,
    )

    seed_counts.to_csv(
        OUTPUT_SEED_COUNTS,
        index=False,
    )

    paired_all = build_all_paired_differences(seed_level)
    paired_all.to_csv(
        OUTPUT_PAIRED,
        index=False,
    )

    stats = compute_directional_stats(paired_all)
    stats.to_csv(
        OUTPUT_STATS,
        index=False,
    )

    latex_body = build_latex_body(stats)
    OUTPUT_LATEX.write_text(
        latex_body + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("DIRECTIONAL CODA ABLATIONS")
    print("=" * 100)
    print(stats.to_string(index=False))

    print("\nSciPy version:", scipy.__version__)
    print(
        "Bootstrap: "
        f"N={N_BOOT}, percentile 95% CI, seed={BOOT_SEED}"
    )
    print(
        "Wilcoxon: exact, two-sided, zero_method='wilcox', "
        "correction=False; zero differences explicitly removed first"
    )
    print(
        "Holm family: 16 directional "
        "learner-environment-contrast comparisons"
    )

    print("\n" + "=" * 100)
    print("LATEX TABLE BODY")
    print("=" * 100 + "\n")
    print(latex_body)

    print("\nGenerated outputs:")
    print(f"  Seed-level outcomes : {OUTPUT_SEED_LEVEL}")
    print(f"  Seed counts         : {OUTPUT_SEED_COUNTS}")
    print(f"  Paired differences  : {OUTPUT_PAIRED}")
    print(f"  Directional stats   : {OUTPUT_STATS}")
    print(f"  LaTeX table body    : {OUTPUT_LATEX}")
    print("\nDone.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()