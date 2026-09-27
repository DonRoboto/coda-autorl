#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce CODA computational and interaction-cost summaries from archived
Ray/RLlib training metrics.

For each learner-method-environment-training-seed HPO run, the script computes

    N_steps = sum_p T_end,p
    t_agg   = sum_p t_end,p
    C_comp  = 1e5 * t_agg / N_steps

where p indexes every trial/worker participating in the HPO run. T_end,p and
t_end,p are taken from the last exported report for that participant in causal
order. Thus, ``t_agg`` is aggregate *trial execution time*, not wall-clock time
for the parallel HPO run.

CODA overhead relative to a population baseline b in {PBT, PB2} is computed
within matched training seeds as

    100 * (C_comp_CODA - C_comp_b) / C_comp_b

and only then summarized across seeds by the median and IQR.

The script intentionally uses all trials participating in each HPO procedure;
it does not use only the selected champion.

Typical use from the repository root:

    python analysis/computational_cost.py

Outputs are written by default to:

    results/analysis/computational_cost/

Required columns in each metrics CSV:
    agente_id, timesteps_total, time_total_s

Preferred ordering column:
    causal_order
Fallback ordering columns:
    training_iteration, then original row order.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ENV_ORDER = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

LEARNER_ORDER = ["PPO", "SAC"]
METHOD_ORDER = ["PBT", "PB2", "ASHA", "CODA"]

METHOD_ALIASES: Dict[str, Dict[str, str]] = {
    "PPO": {
        "PBT_PPO_HPO": "PBT",
        "PB2_PPO_HPO": "PB2",
        "ASHA_PPO_HPO": "ASHA",
        "CODA_FULL": "CODA",
    },
    "SAC": {
        "PBT_SAC_HPO": "PBT",
        "PB2_SAC_HPO": "PB2",
        "ASHA_SAC_HPO": "ASHA",
        "CODA_FULL": "CODA",
    },
}

EXPECTED_PARTICIPANTS = {
    ("PPO", "PBT"): 4,
    ("PPO", "PB2"): 4,
    ("PPO", "CODA"): 4,
    ("PPO", "ASHA"): 30,
    ("SAC", "PBT"): 4,
    ("SAC", "PB2"): 4,
    ("SAC", "CODA"): 4,
    ("SAC", "ASHA"): 8,
}

EXPECTED_SEEDS = 10
NORMALIZATION_STEPS = 100_000.0


@dataclass(frozen=True)
class MetricFile:
    learner: str
    environment: str
    raw_method: str
    method: str
    training_seed: int
    source_name: str
    source_kind: str  # "directory" or "zip"


class MetricsSource:
    """Read metrics CSV files from an extracted directory or ZIP archive."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self._zip: Optional[zipfile.ZipFile] = None
        if self.path.is_file():
            if self.path.suffix.lower() != ".zip":
                raise ValueError(f"Expected a directory or .zip archive: {self.path}")
            self._zip = zipfile.ZipFile(self.path, "r")

    @property
    def kind(self) -> str:
        return "zip" if self._zip is not None else "directory"

    def names(self) -> Iterable[str]:
        if self._zip is not None:
            return [n for n in self._zip.namelist() if n.endswith(".csv")]
        return [
            str(p.relative_to(self.path)).replace("\\", "/")
            for p in self.path.rglob("*.csv")
        ]

    def read_csv(self, relative_name: str) -> pd.DataFrame:
        if self._zip is not None:
            with self._zip.open(relative_name) as fh:
                return pd.read_csv(fh)
        return pd.read_csv(self.path / relative_name)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "MetricsSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _last_report(group: pd.DataFrame) -> pd.Series:
    """Return the last exported report in causal order for one participant."""
    g = group.copy()
    g["__row_order"] = np.arange(len(g), dtype=np.int64)

    if "causal_order" in g.columns:
        order = _safe_numeric(g["causal_order"])
        if order.notna().any():
            max_order = order.max()
            candidates = g.loc[order.eq(max_order)]
            return candidates.sort_values("__row_order", kind="stable").iloc[-1]

    if "training_iteration" in g.columns:
        order = _safe_numeric(g["training_iteration"])
        if order.notna().any():
            max_order = order.max()
            candidates = g.loc[order.eq(max_order)]
            return candidates.sort_values("__row_order", kind="stable").iloc[-1]

    return g.sort_values("__row_order", kind="stable").iloc[-1]


def _discover_metric_files(
    source: MetricsSource,
    learner: str,
) -> List[MetricFile]:
    aliases = METHOD_ALIASES[learner]
    pattern = re.compile(r"metrics_(.+)_seed(\d+)\.csv$")
    discovered: List[MetricFile] = []

    for name in source.names():
        normalized = name.replace("\\", "/")
        filename = normalized.rsplit("/", 1)[-1]
        match = pattern.match(filename)
        if match is None:
            continue

        raw_method = match.group(1)
        if raw_method not in aliases:
            # Directional variants are deliberately excluded from the cost table.
            continue

        parts = normalized.split("/")
        environment = next((p for p in parts if p in ENV_ORDER), None)
        if environment is None:
            continue

        discovered.append(
            MetricFile(
                learner=learner,
                environment=environment,
                raw_method=raw_method,
                method=aliases[raw_method],
                training_seed=int(match.group(2)),
                source_name=name,
                source_kind=source.kind,
            )
        )

    return discovered


def _validate_columns(df: pd.DataFrame, source_name: str) -> None:
    required = {"agente_id", "timesteps_total", "time_total_s"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(
            f"{source_name}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def summarize_hpo_run(
    df: pd.DataFrame,
    metric_file: MetricFile,
) -> Tuple[dict, List[dict]]:
    """Compute one HPO-run cost record and participant-level audit rows."""
    _validate_columns(df, metric_file.source_name)

    participant_rows: List[dict] = []

    for agent_id, group in df.groupby("agente_id", sort=False, dropna=False):
        final = _last_report(group)

        final_steps = pd.to_numeric(
            pd.Series([final.get("timesteps_total", np.nan)]), errors="coerce"
        ).iloc[0]
        final_time = pd.to_numeric(
            pd.Series([final.get("time_total_s", np.nan)]), errors="coerce"
        ).iloc[0]

        all_times = _safe_numeric(group["time_total_s"])
        max_time = float(all_times.max()) if all_times.notna().any() else np.nan

        if not np.isfinite(final_steps) or final_steps <= 0:
            raise ValueError(
                f"{metric_file.source_name}, participant={agent_id}: "
                f"invalid final timesteps_total={final_steps}"
            )
        if not np.isfinite(final_time) or final_time < 0:
            raise ValueError(
                f"{metric_file.source_name}, participant={agent_id}: "
                f"invalid final time_total_s={final_time}"
            )

        participant_rows.append(
            {
                "learner": metric_file.learner,
                "environment": metric_file.environment,
                "method": metric_file.method,
                "raw_method": metric_file.raw_method,
                "training_seed": metric_file.training_seed,
                "participant": str(agent_id),
                "final_timesteps_total": float(final_steps),
                "final_time_total_s": float(final_time),
                "max_observed_time_total_s": max_time,
                "final_minus_max_time_s": (
                    float(final_time - max_time)
                    if np.isfinite(max_time)
                    else np.nan
                ),
                "n_metric_rows": int(len(group)),
                "source_file": metric_file.source_name,
            }
        )

    participants = pd.DataFrame(participant_rows)
    if participants.empty:
        raise RuntimeError(f"No participants found in {metric_file.source_name}")

    n_steps = float(participants["final_timesteps_total"].sum())
    t_agg = float(participants["final_time_total_s"].sum())

    if n_steps <= 0:
        raise ValueError(f"{metric_file.source_name}: aggregate N_steps <= 0")

    c_comp = NORMALIZATION_STEPS * t_agg / n_steps

    expected_n = EXPECTED_PARTICIPANTS.get(
        (metric_file.learner, metric_file.method), np.nan
    )

    run_row = {
        "learner": metric_file.learner,
        "environment": metric_file.environment,
        "method": metric_file.method,
        "raw_method": metric_file.raw_method,
        "training_seed": metric_file.training_seed,
        "n_participants": int(len(participants)),
        "expected_participants": expected_n,
        "participant_count_ok": (
            bool(len(participants) == expected_n)
            if np.isfinite(expected_n)
            else np.nan
        ),
        "aggregate_interactions": n_steps,
        "aggregate_interactions_million": n_steps / 1_000_000.0,
        "aggregate_trial_time_s": t_agg,
        "C_comp_s_per_100k_steps": c_comp,
        "source_file": metric_file.source_name,
    }

    return run_row, participant_rows


def _q1(x: pd.Series) -> float:
    return float(x.quantile(0.25))


def _q3(x: pd.Series) -> float:
    return float(x.quantile(0.75))


def make_cost_summary(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (learner, env, method), g in seed_level.groupby(
        ["learner", "environment", "method"], sort=False
    ):
        c = g["C_comp_s_per_100k_steps"]
        steps = g["aggregate_interactions_million"]
        time_s = g["aggregate_trial_time_s"]

        rows.append(
            {
                "learner": learner,
                "environment": env,
                "method": method,
                "n_seeds": int(g["training_seed"].nunique()),
                "C_comp_median": float(c.median()),
                "C_comp_q1": _q1(c),
                "C_comp_q3": _q3(c),
                "interactions_million_median": float(steps.median()),
                "interactions_million_q1": _q1(steps),
                "interactions_million_q3": _q3(steps),
                "aggregate_trial_time_s_median": float(time_s.median()),
                "aggregate_trial_time_s_q1": _q1(time_s),
                "aggregate_trial_time_s_q3": _q3(time_s),
            }
        )

    return pd.DataFrame(rows)


def make_relative_costs(
    seed_level: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: List[dict] = []

    for learner in LEARNER_ORDER:
        for env in ENV_ORDER:
            current = seed_level[
                seed_level["learner"].eq(learner)
                & seed_level["environment"].eq(env)
            ]

            coda = current[current["method"].eq("CODA")][
                ["training_seed", "C_comp_s_per_100k_steps"]
            ].rename(columns={"C_comp_s_per_100k_steps": "C_CODA"})

            for baseline in ["PBT", "PB2"]:
                base = current[current["method"].eq(baseline)][
                    ["training_seed", "C_comp_s_per_100k_steps"]
                ].rename(columns={"C_comp_s_per_100k_steps": "C_baseline"})

                matched = coda.merge(
                    base,
                    on="training_seed",
                    how="inner",
                    validate="one_to_one",
                ).sort_values("training_seed")

                if len(matched) != EXPECTED_SEEDS:
                    print(
                        f"WARNING: {learner} {env} CODA vs {baseline}: "
                        f"{len(matched)} matched seeds; expected {EXPECTED_SEEDS}.",
                        file=sys.stderr,
                    )

                matched["relative_difference_pct"] = 100.0 * (
                    matched["C_CODA"] - matched["C_baseline"]
                ) / matched["C_baseline"]

                for row in matched.itertuples(index=False):
                    seed_rows.append(
                        {
                            "learner": learner,
                            "environment": env,
                            "baseline": baseline,
                            "training_seed": int(row.training_seed),
                            "C_comp_CODA": float(row.C_CODA),
                            "C_comp_baseline": float(row.C_baseline),
                            "relative_difference_pct": float(
                                row.relative_difference_pct
                            ),
                        }
                    )

    seed_df = pd.DataFrame(seed_rows)

    summary_rows: List[dict] = []
    for (learner, env, baseline), g in seed_df.groupby(
        ["learner", "environment", "baseline"], sort=False
    ):
        x = g["relative_difference_pct"]
        summary_rows.append(
            {
                "learner": learner,
                "environment": env,
                "baseline": baseline,
                "n_matched_seeds": int(g["training_seed"].nunique()),
                "relative_difference_pct_median": float(x.median()),
                "relative_difference_pct_q1": _q1(x),
                "relative_difference_pct_q3": _q3(x),
                "relative_difference_pct_min": float(x.min()),
                "relative_difference_pct_max": float(x.max()),
            }
        )

    return seed_df, pd.DataFrame(summary_rows)


def _format_median_iqr(med: float, q1: float, q3: float, digits: int) -> str:
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a two-panel manuscript/supplement-style LaTeX table."""
    lookup = summary.set_index(["learner", "environment", "method"])

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Computational and interaction cost of the HPO procedures. "
        r"Panel A reports aggregate trial execution time normalized by total "
        r"sampled interactions (seconds per $10^5$ environment interactions). "
        r"Panel B reports total sampled environment interactions across all "
        r"trials participating in the HPO run, in millions. Values are median "
        r"$[Q_1,Q_3]$ across training seeds.}",
        r"\label{tab:computational-cost}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.2pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Learner} & \textbf{Environment} & \textbf{PBT} & \textbf{PB2} & \textbf{ASHA} & \textbf{CODA} \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel A: Compute time per $10^5$ sampled interactions (s)}} \\[1pt]",
    ]

    for learner in LEARNER_ORDER:
        for env in ENV_ORDER:
            vals = []
            for method in METHOD_ORDER:
                r = lookup.loc[(learner, env, method)]
                vals.append(
                    _format_median_iqr(
                        r["C_comp_median"], r["C_comp_q1"], r["C_comp_q3"], 1
                    )
                )
            lines.append(f"{learner} & {env} & " + " & ".join(vals) + r" \\")

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{6}{l}{\textit{Panel B: Total sampled environment interactions (million)}} \\[1pt]",
        ]
    )

    for learner in LEARNER_ORDER:
        for env in ENV_ORDER:
            vals = []
            for method in METHOD_ORDER:
                r = lookup.loc[(learner, env, method)]
                vals.append(
                    _format_median_iqr(
                        r["interactions_million_median"],
                        r["interactions_million_q1"],
                        r["interactions_million_q3"],
                        2,
                    )
                )
            lines.append(f"{learner} & {env} & " + " & ".join(vals) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sort_outputs(df: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    out = df.copy()
    out["learner"] = pd.Categorical(out["learner"], LEARNER_ORDER, ordered=True)
    out["environment"] = pd.Categorical(
        out["environment"], ENV_ORDER, ordered=True
    )
    if "method" in out.columns:
        out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)

    sort_cols = ["learner", "environment"]
    if "method" in out.columns:
        sort_cols.append("method")
    sort_cols.extend(extra)

    out = out.sort_values(sort_cols).reset_index(drop=True)
    for col in ["learner", "environment", "method"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_repo = script_path.parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce interaction-normalized computational cost and total "
            "HPO interaction counts from archived CODA metrics."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo,
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--ppo-metrics",
        type=Path,
        default=None,
        help=(
            "PPO metrics directory or ZIP. Default: "
            "<repo>/results/ppo/metrics"
        ),
    )
    parser.add_argument(
        "--sac-metrics",
        type=Path,
        default=None,
        help=(
            "SAC metrics directory or ZIP. Default: "
            "<repo>/results/sac/metrics"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<repo>/results/analysis/computational_cost"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    ppo_path = (
        args.ppo_metrics.resolve()
        if args.ppo_metrics is not None
        else repo_root / "results" / "ppo" / "metrics"
    )
    sac_path = (
        args.sac_metrics.resolve()
        if args.sac_metrics is not None
        else repo_root / "results" / "sac" / "metrics"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "results" / "analysis" / "computational_cost"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[dict] = []
    participant_rows: List[dict] = []

    for learner, source_path in [("PPO", ppo_path), ("SAC", sac_path)]:
        with MetricsSource(source_path) as source:
            files = _discover_metric_files(source, learner)

            if not files:
                raise RuntimeError(
                    f"No supported {learner} metric files found in {source_path}"
                )

            for metric_file in files:
                df = source.read_csv(metric_file.source_name)
                run_row, part_rows = summarize_hpo_run(df, metric_file)
                run_rows.append(run_row)
                participant_rows.extend(part_rows)

    seed_level = pd.DataFrame(run_rows)
    participant_level = pd.DataFrame(participant_rows)

    # Validate design coverage.
    expected_runs = len(LEARNER_ORDER) * len(ENV_ORDER) * len(METHOD_ORDER) * EXPECTED_SEEDS
    if len(seed_level) != expected_runs:
        print(
            f"WARNING: found {len(seed_level)} primary-comparison HPO runs; "
            f"expected {expected_runs}.",
            file=sys.stderr,
        )

    duplicate_mask = seed_level.duplicated(
        ["learner", "environment", "method", "training_seed"], keep=False
    )
    if duplicate_mask.any():
        raise RuntimeError(
            "Duplicate learner/environment/method/seed runs detected:\n"
            + seed_level.loc[
                duplicate_mask,
                ["learner", "environment", "method", "training_seed", "source_file"],
            ].to_string(index=False)
        )

    cost_summary = make_cost_summary(seed_level)
    relative_seed, relative_summary = make_relative_costs(seed_level)

    seed_level = _sort_outputs(seed_level, extra=["training_seed"])
    participant_level = _sort_outputs(
        participant_level, extra=["training_seed", "participant"]
    )
    cost_summary = _sort_outputs(cost_summary)

    relative_seed["learner"] = pd.Categorical(
        relative_seed["learner"], LEARNER_ORDER, ordered=True
    )
    relative_seed["environment"] = pd.Categorical(
        relative_seed["environment"], ENV_ORDER, ordered=True
    )
    relative_seed["baseline"] = pd.Categorical(
        relative_seed["baseline"], ["PBT", "PB2"], ordered=True
    )
    relative_seed = relative_seed.sort_values(
        ["learner", "environment", "baseline", "training_seed"]
    ).reset_index(drop=True)
    for col in ["learner", "environment", "baseline"]:
        relative_seed[col] = relative_seed[col].astype(str)

    relative_summary["learner"] = pd.Categorical(
        relative_summary["learner"], LEARNER_ORDER, ordered=True
    )
    relative_summary["environment"] = pd.Categorical(
        relative_summary["environment"], ENV_ORDER, ordered=True
    )
    relative_summary["baseline"] = pd.Categorical(
        relative_summary["baseline"], ["PBT", "PB2"], ordered=True
    )
    relative_summary = relative_summary.sort_values(
        ["learner", "environment", "baseline"]
    ).reset_index(drop=True)
    for col in ["learner", "environment", "baseline"]:
        relative_summary[col] = relative_summary[col].astype(str)

    # Save reproducible outputs.
    participant_level.to_csv(
        output_dir / "computational_cost_participant_level.csv", index=False
    )
    seed_level.to_csv(
        output_dir / "computational_cost_seed_level.csv", index=False
    )
    cost_summary.to_csv(
        output_dir / "computational_cost_summary.csv", index=False
    )
    relative_seed.to_csv(
        output_dir / "computational_cost_relative_seed_level.csv", index=False
    )
    relative_summary.to_csv(
        output_dir / "computational_cost_relative_summary.csv", index=False
    )
    write_latex_table(
        cost_summary,
        output_dir / "computational_cost_table.tex",
    )

    # Console summary.
    print("\nCODA computational and interaction-cost analysis")
    print("=" * 78)
    print(f"PPO metrics: {ppo_path}")
    print(f"SAC metrics: {sac_path}")
    print(f"Output dir : {output_dir}")
    print("\nDefinition:")
    print("  N_steps = sum of final timesteps_total over all participating trials")
    print("  t_agg   = sum of final time_total_s over all participating trials")
    print("  C_comp  = 1e5 * t_agg / N_steps")
    print("  Relative CODA cost is computed within matched seeds before aggregation.")

    print("\nMedian computational cost and interactions:")
    display = cost_summary[
        [
            "learner",
            "environment",
            "method",
            "n_seeds",
            "C_comp_median",
            "C_comp_q1",
            "C_comp_q3",
            "interactions_million_median",
            "interactions_million_q1",
            "interactions_million_q3",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMatched CODA relative compute differences (%):")
    print(
        relative_summary[
            [
                "learner",
                "environment",
                "baseline",
                "n_matched_seeds",
                "relative_difference_pct_median",
                "relative_difference_pct_q1",
                "relative_difference_pct_q3",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nAcross the 8 learner-environment settings:")
    for baseline in ["PBT", "PB2"]:
        values = relative_summary.loc[
            relative_summary["baseline"].eq(baseline),
            "relative_difference_pct_median",
        ]
        print(
            f"  CODA vs {baseline}: median setting-level range "
            f"{values.min():+.2f}% to {values.max():+.2f}%"
        )

    participant_mismatch = seed_level.loc[
        seed_level["participant_count_ok"].eq(False)
    ]
    if not participant_mismatch.empty:
        print(
            "\nWARNING: participant-count mismatches detected:\n"
            + participant_mismatch[
                [
                    "learner",
                    "environment",
                    "method",
                    "training_seed",
                    "n_participants",
                    "expected_participants",
                ]
            ].to_string(index=False),
            file=sys.stderr,
        )

    # Surface cumulative-time rollbacks without silently changing the metric.
    final_before_max = participant_level[
        participant_level["final_minus_max_time_s"] < -1e-9
    ]
    if not final_before_max.empty:
        print(
            f"\nNOTE: {len(final_before_max)} participant records have a final "
            "time_total_s below an earlier maximum. The reported metric uses the "
            "final report in causal order, consistent with the stated definition. "
            "See computational_cost_participant_level.csv for audit details."
        )

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
