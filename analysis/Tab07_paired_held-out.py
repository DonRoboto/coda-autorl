#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:18:44 2026

@author: yor5
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests


# ============================================================
# Configuration
# ============================================================

# Cambia estas rutas a tus CSV finales reales
PPO_HELDOUT = Path(
    "../results/ppo/heldout_reward_ppo_final/heldout_test_episodes.csv"
)

SAC_HELDOUT = Path(
    "../results/sac/heldout_reward_sac_final/heldout_test_episodes.csv"
)

N_TEST_EPISODES = 100
N_TRAINING_SEEDS = 10

N_BOOT = 10_000
BOOT_SEED = 20260913

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
    "CODA-I2O",
    "CODA-O2I",
]

# Order used in the final table
CONTRASTS = [
    (
        "CODA-I2O",
        "Full $-$ I2O",
    ),
    (
        "CODA-O2I",
        "Full $-$ O2I",
    ),
]

OUTPUT_SEED_LEVEL = (
    "ablation_seed_level_heldout.csv"
)

OUTPUT_PAIRED = (
    "ablation_paired_differences.csv"
)

OUTPUT_STATS = (
    "ablation_directional_stats.csv"
)

OUTPUT_LATEX = (
    "ablation_heldout_table.tex"
)


# ============================================================
# Load original episode-level held-out data
# ============================================================

def load_heldout(
    path,
    learner
):
    if not path.exists():
        raise FileNotFoundError(
            f"Held-out file not found:\n"
            f"  {path.resolve()}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "method",
        "environment",
        "training_seed",
        "test_return",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["learner"] = learner

    df["training_seed"] = (
        pd.to_numeric(
            df["training_seed"],
            errors="raise"
        )
        .astype(int)
    )

    df["test_return"] = (
        pd.to_numeric(
            df["test_return"],
            errors="raise"
        )
    )

    if not np.isfinite(
        df["test_return"]
    ).all():
        raise ValueError(
            f"{path.name} contains non-finite "
            "test_return values."
        )

    return df


ppo = load_heldout(
    PPO_HELDOUT,
    "PPO"
)

sac = load_heldout(
    SAC_HELDOUT,
    "SAC"
)

episodes = pd.concat(
    [
        ppo,
        sac,
    ],
    ignore_index=True
)


# ============================================================
# Inspect method names before filtering
# ============================================================

print(
    "\nMethods present in held-out files:"
)

print(
    sorted(
        episodes[
            "method"
        ]
        .astype(str)
        .unique()
    )
)


# ============================================================
# OPTIONAL:
# Normalize method aliases if your files use slightly
# different naming conventions.
#
# Edit ONLY if your actual CSVs use these names.
# ============================================================

METHOD_ALIASES = {
    "CODA_FULL": "CODA",
    "CODA-FULL": "CODA",
    "CODA_I2O": "CODA-I2O",
    "CODA_O2I": "CODA-O2I",
}

episodes["method"] = (
    episodes["method"]
    .replace(
        METHOD_ALIASES
    )
)


# ============================================================
# Keep only methods required for directional ablations
# ============================================================

episodes = episodes[
    episodes[
        "method"
    ].isin(METHODS)
].copy()


# ============================================================
# Verify that all required methods exist
# ============================================================

observed_methods = set(
    episodes["method"].unique()
)

missing_methods = (
    set(METHODS)
    - observed_methods
)

if missing_methods:
    raise ValueError(
        "Missing methods in held-out files: "
        f"{sorted(missing_methods)}\n"
        "Check the method names printed above."
    )


# ============================================================
# Validate episode counts
# ============================================================

CASE_KEYS = [
    "learner",
    "method",
    "environment",
    "training_seed",
]

episode_counts = (
    episodes
    .groupby(
        CASE_KEYS
    )
    .size()
)

bad_episode_counts = (
    episode_counts[
        episode_counts
        != N_TEST_EPISODES
    ]
)

if not bad_episode_counts.empty:

    print(
        "\nCases with incorrect number "
        "of held-out episodes:"
    )

    print(
        bad_episode_counts
    )

    raise RuntimeError(
        f"Every case must contain exactly "
        f"{N_TEST_EPISODES} held-out episodes."
    )


# ============================================================
# Validate test seeds if available
# ============================================================

if "test_seed" in episodes.columns:

    unique_test_seeds = (
        episodes
        .groupby(
            CASE_KEYS
        )["test_seed"]
        .nunique()
    )

    bad = unique_test_seeds[
        unique_test_seeds
        != N_TEST_EPISODES
    ]

    if not bad.empty:

        print(
            "\nCases with duplicated or missing "
            "test seeds:"
        )

        print(
            bad
        )

        raise RuntimeError(
            "Held-out test-seed validation failed."
        )


# ============================================================
# Validate one champion per case if available
# ============================================================

if "champion_agent" in episodes.columns:

    champion_counts = (
        episodes
        .groupby(
            CASE_KEYS
        )[
            "champion_agent"
        ]
        .nunique()
    )

    bad = champion_counts[
        champion_counts != 1
    ]

    if not bad.empty:

        print(
            "\nCases containing more than "
            "one champion:"
        )

        print(
            bad
        )

        raise RuntimeError(
            "Champion identity is not unique."
        )


# ============================================================
# Validate explore=False if available
# ============================================================

if "explore" in episodes.columns:

    explore_values = (
        episodes[
            "explore"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    valid = explore_values.isin(
        [
            "false",
            "0",
        ]
    )

    if not valid.all():
        raise RuntimeError(
            "At least one episode was not "
            "evaluated with explore=False."
        )


# ============================================================
# Step 1:
# Seed-level held-out mean
#
# G_{g,m,e,s}
# ============================================================

seed_level = (
    episodes
    .groupby(
        [
            "learner",
            "method",
            "environment",
            "training_seed",
        ],
        as_index=False
    )
    .agg(
        n_test_episodes=(
            "test_return",
            "size"
        ),
        test_mean_return=(
            "test_return",
            "mean"
        ),
    )
)


seed_level.to_csv(
    OUTPUT_SEED_LEVEL,
    index=False
)


# ============================================================
# Validate 10 independent training seeds
# ============================================================

seed_counts = (
    seed_level
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
    "\nTraining seeds per cell:\n"
)

print(
    seed_counts
)

bad_seed_counts = (
    seed_counts[
        seed_counts
        != N_TRAINING_SEEDS
    ]
)

if not bad_seed_counts.empty:

    print(
        "\nIncomplete cells:"
    )

    print(
        bad_seed_counts
    )

    raise RuntimeError(
        f"Expected exactly "
        f"{N_TRAINING_SEEDS} training seeds "
        "per cell."
    )


# ============================================================
# Verify identical test-seed sets across variants
# ============================================================

if "test_seed" in episodes.columns:

    for learner in LEARNERS:

        for environment in ENVIRONMENTS:

            for training_seed in sorted(
                seed_level[
                    (
                        seed_level["learner"]
                        == learner
                    )
                    &
                    (
                        seed_level["environment"]
                        == environment
                    )
                ][
                    "training_seed"
                ]
                .unique()
            ):

                reference = None

                for method in METHODS:

                    current = set(
                        episodes[
                            (
                                episodes["learner"]
                                == learner
                            )
                            &
                            (
                                episodes["environment"]
                                == environment
                            )
                            &
                            (
                                episodes["training_seed"]
                                == training_seed
                            )
                            &
                            (
                                episodes["method"]
                                == method
                            )
                        ][
                            "test_seed"
                        ]
                        .astype(int)
                    )

                    if reference is None:
                        reference = current

                    elif current != reference:

                        raise RuntimeError(
                            "Common held-out test seeds "
                            "do not match for:\n"
                            f"{learner}, "
                            f"{environment}, "
                            f"training_seed={training_seed}"
                        )


# ============================================================
# Build matched differences
#
# Full - I2O:
# CODA - CODA-I2O
#
# Full - O2I:
# CODA - CODA-O2I
# ============================================================

def build_paired_data(
    learner,
    environment,
    variant,
    contrast_label
):

    full = (
        seed_level[
            (
                seed_level["learner"]
                == learner
            )
            &
            (
                seed_level["environment"]
                == environment
            )
            &
            (
                seed_level["method"]
                == "CODA"
            )
        ][
            [
                "training_seed",
                "test_mean_return",
            ]
        ]
        .rename(
            columns={
                "test_mean_return":
                    "full_coda"
            }
        )
    )

    variant_df = (
        seed_level[
            (
                seed_level["learner"]
                == learner
            )
            &
            (
                seed_level["environment"]
                == environment
            )
            &
            (
                seed_level["method"]
                == variant
            )
        ][
            [
                "training_seed",
                "test_mean_return",
            ]
        ]
        .rename(
            columns={
                "test_mean_return":
                    "variant"
            }
        )
    )

    paired = (
        full
        .merge(
            variant_df,
            on="training_seed",
            how="inner",
            validate="one_to_one"
        )
        .sort_values(
            "training_seed"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        paired
    ) != N_TRAINING_SEEDS:

        raise RuntimeError(
            f"{learner}, {environment}, "
            f"{variant}: expected "
            f"{N_TRAINING_SEEDS} matched seeds, "
            f"found {len(paired)}."
        )

    paired[
        "difference"
    ] = (
        paired[
            "full_coda"
        ]
        -
        paired[
            "variant"
        ]
    )

    paired[
        "learner"
    ] = learner

    paired[
        "environment"
    ] = environment

    paired[
        "variant_name"
    ] = variant

    paired[
        "contrast"
    ] = contrast_label

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


paired_frames = []

for learner in LEARNERS:

    for environment in ENVIRONMENTS:

        for (
            variant,
            contrast_label
        ) in CONTRASTS:

            paired_frames.append(
                build_paired_data(
                    learner,
                    environment,
                    variant,
                    contrast_label
                )
            )


paired_all = pd.concat(
    paired_frames,
    ignore_index=True
)


paired_all.to_csv(
    OUTPUT_PAIRED,
    index=False
)


# ============================================================
# Rank-biserial correlation
#
# Uses Wilcoxon signed-rank convention:
# zero differences are excluded.
# ============================================================

def rank_biserial(
    differences
):

    d = np.asarray(
        differences,
        dtype=float
    )

    nonzero = (
        d != 0
    )

    d_nz = d[
        nonzero
    ]

    if len(
        d_nz
    ) == 0:

        return np.nan

    ranks = rankdata(
        np.abs(
            d_nz
        ),
        method="average"
    )

    w_plus = (
        ranks[
            d_nz > 0
        ]
        .sum()
    )

    w_minus = (
        ranks[
            d_nz < 0
        ]
        .sum()
    )

    return float(
        (
            w_plus
            -
            w_minus
        )
        /
        (
            w_plus
            +
            w_minus
        )
    )


# ============================================================
# Exact two-sided Wilcoxon
#
# Important:
# For zero_method='wilcox', zero differences are discarded.
#
# To preserve exact inference when a zero difference exists,
# discard the zeros explicitly BEFORE calling scipy.
# This reproduces the intended Wilcoxon convention.
# ============================================================

def exact_wilcoxon(
    differences
):

    d = np.asarray(
        differences,
        dtype=float
    )

    d_nz = d[
        d != 0
    ]

    if len(
        d_nz
    ) == 0:

        return (
            0.0,
            1.0,
        )

    test = wilcoxon(
        d_nz,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="exact",
    )

    return (
        float(
            test.statistic
        ),
        float(
            test.pvalue
        ),
    )


# ============================================================
# Common paired-bootstrap resampling
#
# Same matched-seed index resamples are used for all
# 16 directional comparisons.
# ============================================================

rng = np.random.default_rng(
    BOOT_SEED
)

boot_idx = rng.integers(
    low=0,
    high=N_TRAINING_SEEDS,
    size=(
        N_BOOT,
        N_TRAINING_SEEDS,
    )
)


def bootstrap_median_ci(
    differences
):

    d = np.asarray(
        differences,
        dtype=float
    )

    if len(
        d
    ) != N_TRAINING_SEEDS:

        raise ValueError(
            "Expected exactly "
            f"{N_TRAINING_SEEDS} paired differences."
        )

    boot_medians = np.median(
        d[
            boot_idx
        ],
        axis=1
    )

    ci_low, ci_high = np.quantile(
        boot_medians,
        [
            0.025,
            0.975,
        ]
    )

    return (
        float(
            np.median(
                d
            )
        ),
        float(
            ci_low
        ),
        float(
            ci_high
        ),
    )


# ============================================================
# Compute the 16 directional comparisons
# ============================================================

rows = []

for learner in LEARNERS:

    for environment in ENVIRONMENTS:

        for (
            variant,
            contrast_label
        ) in CONTRASTS:

            current = (
                paired_all[
                    (
                        paired_all["learner"]
                        == learner
                    )
                    &
                    (
                        paired_all["environment"]
                        == environment
                    )
                    &
                    (
                        paired_all["variant_name"]
                        == variant
                    )
                ]
                .sort_values(
                    "training_seed"
                )
            )

            d = (
                current[
                    "difference"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            # ---------------------------------------------
            # W/T/L
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Median + percentile paired-bootstrap CI
            # ---------------------------------------------

            (
                median_diff,
                ci_low,
                ci_high,
            ) = bootstrap_median_ci(
                d
            )

            # ---------------------------------------------
            # Exact paired Wilcoxon
            # ---------------------------------------------

            (
                W,
                p_raw,
            ) = exact_wilcoxon(
                d
            )

            # ---------------------------------------------
            # Matched rank-biserial correlation
            # ---------------------------------------------

            r_rb = rank_biserial(
                d
            )

            rows.append(
                {
                    "learner":
                        learner,

                    "environment":
                        environment,

                    "variant":
                        variant,

                    "contrast":
                        contrast_label,

                    "median_difference":
                        median_diff,

                    "bootstrap_95_lo":
                        ci_low,

                    "bootstrap_95_hi":
                        ci_high,

                    "wins":
                        wins,

                    "ties":
                        ties,

                    "losses":
                        losses,

                    "W":
                        W,

                    "p_raw":
                        p_raw,

                    "r_rb":
                        r_rb,
                }
            )


stats = pd.DataFrame(
    rows
)


# ============================================================
# Holm correction over COMPLETE FAMILY OF 16
# ============================================================

stats[
    "p_Holm"
] = multipletests(
    stats[
        "p_raw"
    ].to_numpy(),
    alpha=0.05,
    method="holm",
)[1]


# ============================================================
# Preserve desired table order
# ============================================================

learner_order = {
    "PPO": 0,
    "SAC": 1,
}

environment_order = {
    env: i
    for i, env
    in enumerate(
        ENVIRONMENTS
    )
}

variant_order = {
    "CODA-I2O": 0,
    "CODA-O2I": 1,
}

stats[
    "_learner_order"
] = (
    stats["learner"]
    .map(
        learner_order
    )
)

stats[
    "_environment_order"
] = (
    stats["environment"]
    .map(
        environment_order
    )
)

stats[
    "_variant_order"
] = (
    stats["variant"]
    .map(
        variant_order
    )
)

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
    .reset_index(
        drop=True
    )
)


# ============================================================
# Save numerical audit table
# ============================================================

stats.to_csv(
    OUTPUT_STATS,
    index=False
)


# ============================================================
# Print audit output
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "DIRECTIONAL CODA ABLATIONS"
)

print(
    "=" * 100
)

print(
    stats.to_string(
        index=False
    )
)

print(
    "\nBootstrap:"
)

print(
    f"  N = {N_BOOT}"
)

print(
    f"  seed = {BOOT_SEED}"
)

print(
    "  percentile 95% CI"
)

print(
    "\nWilcoxon:"
)

print(
    "  two-sided"
)

print(
    "  zero differences excluded"
)

print(
    "  exact distribution"
)

print(
    "\nHolm:"
)

print(
    "  one family containing all "
    "16 directional comparisons"
)


# ============================================================
# LaTeX formatting
# ============================================================

def format_effect(
    row
):
    return (
        f"${row['median_difference']:.2f} "
        f"[{row['bootstrap_95_lo']:.2f}, "
        f"{row['bootstrap_95_hi']:.2f}]$"
    )


def format_wtl(
    row
):
    return (
        f"{int(row['wins'])}/"
        f"{int(row['ties'])}/"
        f"{int(row['losses'])}"
    )


def format_p(
    value
):
    return (
        f"{value:.3f}"
    )


def format_rrb(
    value
):
    return (
        f"{value:.2f}"
    )


# ============================================================
# Generate COMPLETE LaTeX table
# ============================================================

latex_lines = []

latex_lines.extend(
    [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{\justifying",
        (
            "Paired held-out directional ablation effects "
            r"for CODA. $\Delta$ is the"
        ),
        (
            "median seed-wise difference between full CODA "
            "and the corresponding"
        ),
        (
            "one-channel variant, so positive values favor "
            "the complete bidirectional"
        ),
        (
            "configuration. Brackets report unadjusted "
            r"percentile 95\% paired-bootstrap"
        ),
        (
            "confidence intervals based on 10,000 resamples "
            "of the matched training"
        ),
        (
            "seeds. W/T/L denotes the number of matched seeds "
            "for which full CODA"
        ),
        (
            "obtains a higher/equal/lower held-out return than "
            "the corresponding"
        ),
        (
            r"variant. Raw $p$-values are from two-sided paired "
            "Wilcoxon signed-rank"
        ),
        (
            r"tests; $p_{\mathrm{Holm}}$ applies Holm correction "
            "over the complete"
        ),
        (
            r"family of 16 directional comparisons. "
            r"$r_{\mathrm{rb}}$ denotes"
        ),
        (
            "matched-pairs rank-biserial correlation.}"
        ),
        r"\label{tab:ablation-heldout}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        "",
        r"\begin{tabular}{@{}lllccccc@{}}",
        r"\toprule",
        r"\textbf{Learner} &",
        r"\textbf{Environment} &",
        r"\textbf{Contrast} &",
        r"\textbf{$\Delta [95\% CI]$} &",
        r"\textbf{W/T/L} &",
        r"$p$ &",
        r"$p_{\mathrm{Holm}}$ &",
        r"$r_{\mathrm{rb}}$ \\",
        r"\midrule",
        "",
    ]
)


# ============================================================
# Add data rows
# ============================================================

previous_learner = None

for _, row in (
    stats
    .iterrows()
):

    learner = row[
        "learner"
    ]

    if (
        previous_learner is not None
        and
        learner != previous_learner
    ):

        latex_lines.extend(
            [
                "",
                r"\midrule",
                "",
            ]
        )

    latex_lines.extend(
        [
            (
                f"{learner} & "
                f"{row['environment']} &"
            ),

            (
                f"{row['contrast']} &"
            ),

            (
                f"{format_effect(row)} &"
            ),

            (
                f"{format_wtl(row)} & "
                f"{format_p(row['p_raw'])} & "
                f"{format_p(row['p_Holm'])} & "
                f"{format_rrb(row['r_rb'])} "
                r"\\"
            ),

            "",
        ]
    )

    previous_learner = learner


latex_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
)


latex_text = (
    "\n".join(
        latex_lines
    )
)


# ============================================================
# Save LaTeX
# ============================================================

Path(
    OUTPUT_LATEX
).write_text(
    latex_text,
    encoding="utf-8"
)


print(
    "\n"
    + "=" * 100
)

print(
    "LATEX TABLE"
)

print(
    "=" * 100
    + "\n"
)

print(
    latex_text
)

print(
    "\nGenerated files:"
)

print(
    f"  {OUTPUT_SEED_LEVEL}"
)

print(
    f"  {OUTPUT_PAIRED}"
)

print(
    f"  {OUTPUT_STATS}"
)

print(
    f"  {OUTPUT_LATEX}"
)