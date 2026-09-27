#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory 2x2 factorial interaction analysis for CODA.

The four cells are:

                 O2I = 0           O2I = 1
    I2O = 0      PB2               CODA-O2I
    I2O = 1      CODA-I2O          CODA

For each learner g, environment e, and matched training seed s:

    I_{g,e,s}
      = G_CODA
      - G_CODA-I2O
      - G_CODA-O2I
      + G_PB2

This is a seed-level difference-in-differences contrast.

Positive I:
    positive / super-additive interaction on the held-out-return scale.

Negative I:
    sub-additive interaction on the held-out-return scale.

I approximately 0:
    no detectable departure from additivity on that scale.

The script:
1. Reads PPO and SAC held-out ZIP files.
2. Locates exactly one heldout_test_seed_summary.csv in each ZIP.
3. Normalizes method labels.
4. Verifies one observation per method x environment x training seed.
5. Computes the interaction contrast for each matched seed.
6. Reports median, unadjusted 95% percentile bootstrap CI, W/T/L,
   two-sided paired Wilcoxon signed-rank p-value, and rank-biserial effect.
7. Applies Holm correction across the eight learner-environment
   interaction hypotheses.
8. Saves both seed-level and summary CSV files.

Expected summary columns:
    method
    environment
    training_seed
    test_mean_return

Example
-------
python analysis/coda_factorial_interaction.py \
    --ppo heldout_stability_ppo_final.zip \
    --sac heldout_stability_sac_final.zip \
    --output-dir results/statistics/factorial_interaction
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Reproducibility
# =============================================================================

N_BOOT = 10_000
BOOT_SEED = 20260913
ALPHA = 0.05
EXPECTED_SEEDS = 10

ENV_ORDER = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

LEARNER_ORDER = ["PPO", "SAC"]

REQUIRED_METHODS = [
    "PB2",
    "CODA-I2O",
    "CODA-O2I",
    "CODA",
]


# =============================================================================
# Method-name normalization
# =============================================================================

METHOD_ALIASES: Dict[str, str] = {
    # PB2
    "PB2": "PB2",

    # Full CODA
    "CODA": "CODA",
    "CODA_FULL": "CODA",
    "FULL": "CODA",

    # I2O-only
    "CODA-I2O": "CODA-I2O",
    "CODA_I2O": "CODA-I2O",
    "I2O": "CODA-I2O",
    "I2O-ONLY": "CODA-I2O",
    "I2O_ONLY": "CODA-I2O",

    # O2I-only
    "CODA-O2I": "CODA-O2I",
    "CODA_O2I": "CODA-O2I",
    "O2I": "CODA-O2I",
    "O2I-ONLY": "CODA-O2I",
    "O2I_ONLY": "CODA-O2I",
}


def canonical_method_name(value: str) -> str:
    key = str(value).strip()
    key_upper = key.upper()

    # Exact dictionary lookup first.
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]

    # Then case-insensitive lookup.
    for alias, canonical in METHOD_ALIASES.items():
        if alias.upper() == key_upper:
            return canonical

    return key


# =============================================================================
# Input
# =============================================================================

def read_summary(zip_path: Path) -> pd.DataFrame:
    """
    Read exactly one heldout_test_seed_summary.csv from a ZIP archive.
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    if zip_path.suffix.lower() != ".zip":
        raise ValueError(
            f"{zip_path} is not a ZIP file. "
            "This script expects the held-out evaluation archive."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            name
            for name in zf.namelist()
            if name.endswith("heldout_test_seed_summary.csv")
        ]

        if len(names) != 1:
            raise RuntimeError(
                f"Expected exactly one heldout_test_seed_summary.csv "
                f"inside {zip_path}; found {len(names)}: {names}"
            )

        with zf.open(names[0]) as f:
            df = pd.read_csv(f)

    required_columns = {
        "method",
        "environment",
        "training_seed",
        "test_mean_return",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise KeyError(
            f"{zip_path} is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["method"] = df["method"].map(canonical_method_name)
    df["environment"] = df["environment"].astype(str)
    df["training_seed"] = pd.to_numeric(
        df["training_seed"],
        errors="raise",
    ).astype(int)

    df["test_mean_return"] = pd.to_numeric(
        df["test_mean_return"],
        errors="coerce",
    )

    if df["test_mean_return"].isna().any():
        bad = df[df["test_mean_return"].isna()]
        raise ValueError(
            "Non-finite test_mean_return values were found:\n"
            f"{bad[['method', 'environment', 'training_seed']].to_string(index=False)}"
        )

    return df


# =============================================================================
# Statistical helpers
# =============================================================================

def rank_biserial(differences: Iterable[float]) -> float:
    """
    Matched-pairs rank-biserial correlation.

    r_rb = (W+ - W-) / (W+ + W-)

    Zero differences are removed, matching the Wilcoxon zero_method='wilcox'
    convention.
    """
    d = np.asarray(list(differences), dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]

    if d.size == 0:
        return 0.0

    ranks = rankdata(
        np.abs(d),
        method="average",
    )

    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())

    denom = w_plus + w_minus

    if denom == 0.0:
        return 0.0

    return (w_plus - w_minus) / denom


def wilcoxon_against_zero(values: np.ndarray) -> Tuple[float, float, str]:
    """
    Two-sided Wilcoxon signed-rank test against zero.

    Uses exact calculation when there are no zero differences.
    If zero differences occur, uses SciPy's approximation because exact
    enumeration with zero_method='wilcox' is not generally available.

    Returns:
        statistic, p_value, method_used
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, "not_available"

    if np.all(values == 0.0):
        return 0.0, 1.0, "all_zero"

    has_zero = bool(np.any(values == 0.0))

    method = "approx" if has_zero else "exact"

    result = wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method=method,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
        method,
    )


def bootstrap_median_ci(
    values: np.ndarray,
    boot_idx: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI for the median using paired seed resampling.
    """
    values = np.asarray(values, dtype=float)

    medians = np.median(
        values[boot_idx],
        axis=1,
    )

    lo, hi = np.quantile(
        medians,
        [0.025, 0.975],
    )

    return (
        float(np.median(values)),
        float(lo),
        float(hi),
    )


# =============================================================================
# Matched 2x2 interaction
# =============================================================================

def method_table(
    df: pd.DataFrame,
    learner: str,
    environment: str,
    method: str,
) -> pd.DataFrame:
    out = (
        df[
            df["learner"].eq(learner)
            & df["environment"].eq(environment)
            & df["method"].eq(method)
        ][
            [
                "training_seed",
                "test_mean_return",
            ]
        ]
        .rename(
            columns={
                "test_mean_return": method,
            }
        )
        .copy()
    )

    duplicates = out["training_seed"].duplicated(
        keep=False
    )

    if duplicates.any():
        raise RuntimeError(
            f"Duplicate seed-level rows for "
            f"{learner} / {environment} / {method}:\n"
            f"{out.loc[duplicates].to_string(index=False)}"
        )

    return out


def factorial_interaction_table(
    df: pd.DataFrame,
    learner: str,
    environment: str,
    expected_seeds: int,
) -> pd.DataFrame:
    """
    Build a matched table with the four 2x2 cells and the interaction contrast.
    """
    tables = [
        method_table(
            df,
            learner,
            environment,
            method,
        )
        for method in REQUIRED_METHODS
    ]

    merged = tables[0]

    for table in tables[1:]:
        merged = merged.merge(
            table,
            on="training_seed",
            how="inner",
            validate="one_to_one",
        )

    merged = (
        merged
        .sort_values("training_seed")
        .reset_index(drop=True)
    )

    if len(merged) != expected_seeds:
        availability = {
            method: len(
                method_table(
                    df,
                    learner,
                    environment,
                    method,
                )
            )
            for method in REQUIRED_METHODS
        }

        raise RuntimeError(
            f"{learner} / {environment}: expected "
            f"{expected_seeds} fully matched seeds, got {len(merged)}.\n"
            f"Available rows by method: {availability}"
        )

    # Difference-in-differences:
    #
    # (CODA - CODA-O2I) - (CODA-I2O - PB2)
    #
    # equivalently:
    #
    # CODA - CODA-I2O - CODA-O2I + PB2
    merged["interaction"] = (
        merged["CODA"]
        - merged["CODA-I2O"]
        - merged["CODA-O2I"]
        + merged["PB2"]
    )

    # Optional equivalent decompositions for interpretability.
    merged["I2O_when_O2I_off"] = (
        merged["CODA-I2O"]
        - merged["PB2"]
    )

    merged["I2O_when_O2I_on"] = (
        merged["CODA"]
        - merged["CODA-O2I"]
    )

    merged["O2I_when_I2O_off"] = (
        merged["CODA-O2I"]
        - merged["PB2"]
    )

    merged["O2I_when_I2O_on"] = (
        merged["CODA"]
        - merged["CODA-I2O"]
    )

    merged.insert(
        0,
        "environment",
        environment,
    )
    merged.insert(
        0,
        "learner",
        learner,
    )

    return merged


# =============================================================================
# Main analysis
# =============================================================================

def run_analysis(
    ppo_zip: Path,
    sac_zip: Path,
    output_dir: Path,
    expected_seeds: int = EXPECTED_SEEDS,
    n_boot: int = N_BOOT,
    bootstrap_seed: int = BOOT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ppo = read_summary(ppo_zip)
    sac = read_summary(sac_zip)

    ppo["learner"] = "PPO"
    sac["learner"] = "SAC"

    df = pd.concat(
        [ppo, sac],
        ignore_index=True,
    )

    # Check that the required variants exist after normalization.
    available_methods = sorted(
        df["method"].dropna().unique().tolist()
    )

    missing_methods = [
        method
        for method in REQUIRED_METHODS
        if method not in available_methods
    ]

    if missing_methods:
        raise RuntimeError(
            "The held-out summaries do not contain all four required "
            f"2x2 variants. Missing: {missing_methods}\n"
            f"Available methods: {available_methods}"
        )

    # Same seed-resampling pattern for all eight comparisons.
    rng = np.random.default_rng(
        bootstrap_seed
    )

    boot_idx = rng.integers(
        0,
        expected_seeds,
        size=(n_boot, expected_seeds),
    )

    seed_tables = []
    summary_rows = []

    for learner in LEARNER_ORDER:
        for environment in ENV_ORDER:
            matched = factorial_interaction_table(
                df=df,
                learner=learner,
                environment=environment,
                expected_seeds=expected_seeds,
            )

            seed_tables.append(matched)

            interaction = matched[
                "interaction"
            ].to_numpy(dtype=float)

            median_I, ci_low, ci_high = (
                bootstrap_median_ci(
                    interaction,
                    boot_idx,
                )
            )

            wins = int(
                np.sum(interaction > 0.0)
            )
            ties = int(
                np.sum(interaction == 0.0)
            )
            losses = int(
                np.sum(interaction < 0.0)
            )

            statistic, p_raw, test_method = (
                wilcoxon_against_zero(
                    interaction
                )
            )

            r_rb = rank_biserial(
                interaction
            )

            summary_rows.append(
                {
                    "learner": learner,
                    "environment": environment,
                    "n_seeds": len(interaction),
                    "median_interaction": median_I,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "W_T_L": f"{wins}/{ties}/{losses}",
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_raw,
                    "wilcoxon_method": test_method,
                    "r_rb": r_rb,
                }
            )

    seed_level = pd.concat(
        seed_tables,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    # Holm correction across the eight exploratory interaction hypotheses.
    reject, p_holm, _, _ = multipletests(
        summary["p_raw"].to_numpy(dtype=float),
        alpha=ALPHA,
        method="holm",
    )

    summary["p_holm"] = p_holm
    summary["significant_holm"] = reject

    # Keep the intended learner/environment ordering.
    summary["learner"] = pd.Categorical(
        summary["learner"],
        categories=LEARNER_ORDER,
        ordered=True,
    )

    summary["environment"] = pd.Categorical(
        summary["environment"],
        categories=ENV_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            ["learner", "environment"]
        )
        .reset_index(drop=True)
    )

    summary["learner"] = (
        summary["learner"].astype(str)
    )
    summary["environment"] = (
        summary["environment"].astype(str)
    )

    seed_level["learner"] = pd.Categorical(
        seed_level["learner"],
        categories=LEARNER_ORDER,
        ordered=True,
    )

    seed_level["environment"] = pd.Categorical(
        seed_level["environment"],
        categories=ENV_ORDER,
        ordered=True,
    )

    seed_level = (
        seed_level
        .sort_values(
            [
                "learner",
                "environment",
                "training_seed",
            ]
        )
        .reset_index(drop=True)
    )

    seed_level["learner"] = (
        seed_level["learner"].astype(str)
    )
    seed_level["environment"] = (
        seed_level["environment"].astype(str)
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_level_path = (
        output_dir
        / "coda_i2o_o2i_interaction_seed_level.csv"
    )

    summary_path = (
        output_dir
        / "coda_i2o_o2i_interaction_summary.csv"
    )

    seed_level.to_csv(
        seed_level_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # A compact LaTeX-friendly table.
    latex_df = summary.copy()

    latex_df["interaction_ci"] = latex_df.apply(
        lambda row: (
            f"{row['median_interaction']:.2f} "
            f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        ),
        axis=1,
    )

    latex_df["p_holm_fmt"] = latex_df[
        "p_holm"
    ].map(
        lambda x: f"{x:.3f}"
    )

    latex_df["r_rb_fmt"] = latex_df[
        "r_rb"
    ].map(
        lambda x: f"{x:.2f}"
    )

    latex_table = latex_df[
        [
            "learner",
            "environment",
            "interaction_ci",
            "W_T_L",
            "p_holm_fmt",
            "r_rb_fmt",
        ]
    ]

    latex_path = (
        output_dir
        / "coda_i2o_o2i_interaction_table.tex"
    )

    with latex_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            latex_table.to_latex(
                index=False,
                escape=False,
                column_format="llcccc",
                header=[
                    "Learner",
                    "Environment",
                    r"$I$ [95\% CI]",
                    "W/T/L",
                    r"$p_{\mathrm{Holm}}$",
                    r"$r_{\mathrm{rb}}$",
                ],
            )
        )

    return seed_level, summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the exploratory I2O x O2I factorial interaction "
            "contrast from CODA held-out seed-level results."
        )
    )

    parser.add_argument(
        "--ppo",
        type=Path,
        default=Path(
            "../results/ppo/heldout_reward_ppo_final/heldout_test_seed_summary.zip"
        ),
        help=(
            "PPO held-out ZIP containing "
            "heldout_test_seed_summary.csv"
        ),
    )

    parser.add_argument(
        "--sac",
        type=Path,
        default=Path(
            "../results/sac/heldout_reward_sac_final/heldout_test_seed_summary.zip"
        ),
        help=(
            "SAC held-out ZIP containing "
            "heldout_test_seed_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/statistics/factorial_interaction"
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=EXPECTED_SEEDS,
        help="Expected number of matched training seeds per setting.",
    )

    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=N_BOOT,
        help="Number of bootstrap resamples.",
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOT_SEED,
        help="Bootstrap RNG seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_level, summary = run_analysis(
        ppo_zip=args.ppo,
        sac_zip=args.sac,
        output_dir=args.output_dir,
        expected_seeds=args.expected_seeds,
        n_boot=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )
    pd.set_option(
        "display.width",
        180,
    )

    print(
        "\nCODA I2O x O2I factorial interaction analysis"
    )
    print("=" * 78)

    display_cols = [
        "learner",
        "environment",
        "median_interaction",
        "ci_low",
        "ci_high",
        "W_T_L",
        "p_raw",
        "p_holm",
        "r_rb",
        "significant_holm",
    ]

    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nInterpretation:")
    print(
        "  interaction > 0 : positive/super-additive interaction "
        "on held-out-return scale"
    )
    print(
        "  interaction < 0 : sub-additive interaction "
        "on held-out-return scale"
    )
    print(
        "  p_holm < 0.05    : significant after Holm correction "
        "across the 8 exploratory interaction tests"
    )

    print(
        f"\nSeed-level output: "
        f"{args.output_dir / 'coda_i2o_o2i_interaction_seed_level.csv'}"
    )

    print(
        f"Summary output:    "
        f"{args.output_dir / 'coda_i2o_o2i_interaction_summary.csv'}"
    )

    print(
        f"LaTeX table:       "
        f"{args.output_dir / 'coda_i2o_o2i_interaction_table.tex'}"
    )


if __name__ == "__main__":
    main()
