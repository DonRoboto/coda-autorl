#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution-consistency audit for CODA.

This script reconstructs guided population-inheritance events from CODA
scheduler logs and matches each reconstructable event to the corresponding
exported learner report. It then audits:

    E_h = fraction of auditable events for which the complete requested
          outer configuration matches the learner-effective configuration.

    E_a = fraction of auditable events for which the requested learner-side
          actuator matches the learner-effective actuator.

The audit is performed separately for PPO and SAC.

Expected CODA archive layout
----------------------------
PPO training:
    metrics/<Environment>/metrics_CODA_FULL_seed<seed>.csv

SAC training:
    metrics/<Environment>/metrics_CODA_FULL_seed<seed>.csv

Scheduler logs:
    scheduler/<Environment>/scheduler_CODA_FULL_seed<seed>.csv

Event reconstruction
--------------------
1. Identify synthetic inheritance anchors.
   - SAC: use ``is_synthetic``.
   - PPO: final exported scheduler logs do not contain ``is_synthetic``;
     synthetic anchors are identified because ``guided_update`` is NaN.
2. For each anchor, select the first subsequent genuine scheduler row for
   the same receiver Trial in insertion order.
3. Map the Ray trial suffix ``_00000``, ``_00001``, ... to the exported
   training identifiers ``Agente_1``, ``Agente_2``, ...
4. Match the scheduler's post-inheritance executable configuration and
   actuator to an unused learner report from that agent with
   ``coda_guided_update == 1``.
5. If multiple learner reports share the same configuration, select the one
   whose ``timesteps_total`` is closest to the scheduler post-report time,
   with ``causal_order`` as deterministic tie-breaker.
6. Compare requested versus effective values in the matched learner report.

Numerical comparison
--------------------
- ``train_batch_size``: exact equality after integer rounding.
- floating-point settings: np.isclose(rtol=1e-7, atol=1e-12).

The script intentionally does NOT report an empirical E_memory because the
existing final training logs do not contain an application counter for each
lineage-generation token. The code-level generation guard can be inspected
separately, but that is not the same as an empirical event-level rate.

Example
-------
python analysis/audit_execution_consistency.py \
    --ppo-train results/ppo/ppo_train.zip \
    --ppo-scheduler results/ppo/ppo_scheduler.zip \
    --sac-train results/sac/sac_train.zip \
    --sac-scheduler results/sac/sac_scheduler.zip \
    --output-dir results/audits/execution_consistency
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


RTOL = 1e-7
ATOL = 1e-12


AUDIT_SPEC = {
    "PPO": {
        "scheduler_h": [
            "train_batch_size",
            "lr",
            "clip_param",
            "lambda",
        ],
        "training_h": [
            "train_batch_size",
            "lr",
            "clip_param",
            "lambda",
        ],
        "scheduler_actuator": "entropy_coeff",
        "training_actuator": "entropy_coeff",
    },
    "SAC": {
        "scheduler_h": [
            "train_batch_size",
            "tau",
            "optimization/actor_learning_rate",
            "optimization/critic_learning_rate",
        ],
        "training_h": [
            "train_batch_size",
            "tau",
            "actor_lr",
            "critic_lr",
        ],
        "scheduler_actuator": "target_entropy",
        "training_actuator": "target_entropy",
    },
}


# =============================================================================
# Input source: ZIP archive or extracted directory
# =============================================================================

@dataclass
class CSVSource:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self._zip: Optional[zipfile.ZipFile] = None
        if self.path.is_file():
            if self.path.suffix.lower() != ".zip":
                raise ValueError(
                    f"{self.path} is a file but not a .zip archive."
                )
            self._zip = zipfile.ZipFile(self.path, "r")

    def names(self) -> Iterable[str]:
        if self._zip is not None:
            return self._zip.namelist()

        return [
            str(p.relative_to(self.path)).replace("\\", "/")
            for p in self.path.rglob("*.csv")
        ]

    def read_csv(self, relative_name: str) -> pd.DataFrame:
        if self._zip is not None:
            with self._zip.open(relative_name) as f:
                return pd.read_csv(f)

        return pd.read_csv(self.path / relative_name)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "CSVSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# =============================================================================
# File discovery
# =============================================================================

def discover_full_coda_files(
    source: CSVSource,
    kind: str,
) -> Dict[Tuple[str, int], str]:
    """
    Return {(environment, seed): relative_path} for Full-CODA files.
    """
    if kind == "metrics":
        pattern = re.compile(
            r"^metrics/([^/]+)/metrics_CODA_FULL_seed(\d+)\.csv$"
        )
    elif kind == "scheduler":
        pattern = re.compile(
            r"^scheduler/([^/]+)/scheduler_CODA_FULL_seed(\d+)\.csv$"
        )
    else:
        raise ValueError("kind must be 'metrics' or 'scheduler'")

    found: Dict[Tuple[str, int], str] = {}

    for name in source.names():
        normalized = str(name).replace("\\", "/")
        match = pattern.match(normalized)
        if match is None:
            continue

        key = (match.group(1), int(match.group(2)))
        if key in found:
            raise RuntimeError(
                f"Duplicate Full-CODA {kind} file for {key}: "
                f"{found[key]} and {normalized}"
            )
        found[key] = normalized

    return found


# =============================================================================
# Numeric comparisons
# =============================================================================

def _numeric(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def settings_equal(
    requested,
    effective,
    parameter_name: str,
) -> bool:
    requested = _numeric(requested)
    effective = _numeric(effective)

    if not (np.isfinite(requested) and np.isfinite(effective)):
        return False

    if parameter_name == "train_batch_size":
        return int(round(requested)) == int(round(effective))

    return bool(
        np.isclose(
            requested,
            effective,
            rtol=RTOL,
            atol=ATOL,
        )
    )


def values_auditable(a, b) -> bool:
    return np.isfinite(_numeric(a)) and np.isfinite(_numeric(b))


# =============================================================================
# Scheduler-event reconstruction
# =============================================================================

def synthetic_mask(scheduler_df: pd.DataFrame) -> pd.Series:
    """
    Identify synthetic inheritance anchors.

    SAC final logs export `is_synthetic`.
    PPO final logs omit it; synthetic rows have NaN `guided_update`.
    """
    if "is_synthetic" in scheduler_df.columns:
        values = scheduler_df["is_synthetic"]

        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool)

        return (
            values.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "1.0"})
        )

    if "guided_update" not in scheduler_df.columns:
        raise KeyError(
            "Cannot identify PPO synthetic anchors: "
            "`guided_update` is missing."
        )

    return pd.to_numeric(
        scheduler_df["guided_update"],
        errors="coerce",
    ).isna()


def trial_to_agent(trial_name: str) -> Optional[str]:
    """
    Map Ray trial suffix:
        ..._00000 -> Agente_1
        ..._00001 -> Agente_2
        ...
    """
    match = re.search(r"_(\d+)$", str(trial_name))
    if match is None:
        return None

    return f"Agente_{int(match.group(1)) + 1}"


def reconstruct_scheduler_events(
    scheduler_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Reconstruct synthetic-anchor -> first subsequent genuine report.
    """
    df = scheduler_df.copy().reset_index(drop=True)
    df["_scheduler_row"] = np.arange(len(df), dtype=np.int64)

    synthetic = synthetic_mask(df).to_numpy(dtype=bool)

    events = []
    anchors = np.flatnonzero(synthetic)

    for anchor_row in anchors:
        trial = str(df.at[anchor_row, "Trial"])

        candidates = np.flatnonzero(
            (np.arange(len(df)) > anchor_row)
            & (~synthetic)
            & df["Trial"].astype(str).eq(trial).to_numpy()
        )

        if candidates.size == 0:
            continue

        post_row = int(candidates[0])
        post = df.iloc[post_row]

        guided = _numeric(post.get("guided_update", np.nan))
        is_guided = bool(np.isfinite(guided) and guided >= 0.5)

        events.append(
            {
                "trial": trial,
                "anchor_row": int(anchor_row),
                "post_row": post_row,
                "post_time": _numeric(post.get("Time", np.nan)),
                "agent": trial_to_agent(trial),
                "post_guided_update": float(is_guided),
            }
        )

    audit = {
        "anchors": int(len(anchors)),
        "resolved": int(len(events)),
        "no_post": int(len(anchors) - len(events)),
        "resolved_guided": int(
            sum(event["post_guided_update"] >= 0.5 for event in events)
        ),
    }

    return pd.DataFrame(events), audit


# =============================================================================
# Match scheduler events to training reports
# =============================================================================

def _desired_col(parameter: str) -> str:
    return f"custom_metrics/coda_desired_{parameter}"


def _effective_col(parameter: str) -> str:
    return f"custom_metrics/coda_effective_{parameter}"


def _mismatch_col(parameter: str) -> str:
    return f"custom_metrics/coda_mismatch_{parameter}"


def training_match_mask(
    post_scheduler_row: pd.Series,
    training_rows: pd.DataFrame,
    learner: str,
) -> np.ndarray:
    """
    Match the scheduler post-inheritance executable settings against the
    learner report's requested settings.
    """
    spec = AUDIT_SPEC[learner]
    mask = np.ones(len(training_rows), dtype=bool)

    for scheduler_name, training_name in zip(
        spec["scheduler_h"],
        spec["training_h"],
    ):
        column = _desired_col(training_name)
        if column not in training_rows.columns:
            raise KeyError(f"Missing training column: {column}")

        values = pd.to_numeric(
            training_rows[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        target = _numeric(post_scheduler_row[scheduler_name])

        if training_name == "train_batch_size":
            valid = np.isfinite(values) & np.isfinite(target)
            equal = np.zeros(len(values), dtype=bool)
            equal[valid] = (
                np.rint(values[valid]).astype(np.int64)
                == int(round(target))
            )
        else:
            equal = (
                np.isfinite(values)
                & np.isfinite(target)
                & np.isclose(
                    values,
                    target,
                    rtol=RTOL,
                    atol=ATOL,
                )
            )

        mask &= equal

    scheduler_actuator = spec["scheduler_actuator"]
    training_actuator = spec["training_actuator"]

    actuator_col = _desired_col(training_actuator)
    if actuator_col not in training_rows.columns:
        raise KeyError(f"Missing training column: {actuator_col}")

    values = pd.to_numeric(
        training_rows[actuator_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    target = _numeric(post_scheduler_row[scheduler_actuator])

    mask &= (
        np.isfinite(values)
        & np.isfinite(target)
        & np.isclose(
            values,
            target,
            rtol=RTOL,
            atol=ATOL,
        )
    )

    guided_col = "custom_metrics/coda_guided_update"
    if guided_col in training_rows.columns:
        guided = pd.to_numeric(
            training_rows[guided_col],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)

        mask &= guided >= 0.5

    return mask


def audit_training_report(
    row: pd.Series,
    learner: str,
) -> Tuple[float, float]:
    """
    Return event-level (E_h, E_a).

    NaN means that the requested/effective pair was not auditable.
    """
    spec = AUDIT_SPEC[learner]

    h_auditable = True
    h_matches = True

    for parameter in spec["training_h"]:
        requested = row.get(_desired_col(parameter), np.nan)
        effective = row.get(_effective_col(parameter), np.nan)

        auditable = values_auditable(requested, effective)
        h_auditable &= auditable

        if auditable:
            h_matches &= settings_equal(
                requested,
                effective,
                parameter,
            )
        else:
            h_matches = False

    actuator = spec["training_actuator"]
    requested_a = row.get(_desired_col(actuator), np.nan)
    effective_a = row.get(_effective_col(actuator), np.nan)

    a_auditable = values_auditable(requested_a, effective_a)
    a_matches = (
        a_auditable
        and settings_equal(
            requested_a,
            effective_a,
            actuator,
        )
    )

    E_h = (
        float(h_matches)
        if h_auditable
        else np.nan
    )

    E_a = (
        float(a_matches)
        if a_auditable
        else np.nan
    )

    return E_h, E_a


def audit_scheduler_anchor_to_post(
    anchor: pd.Series,
    post: pd.Series,
    learner: str,
) -> Tuple[float, float]:
    """
    Optional internal scheduler check:
    synthetic anchor settings should equal the first subsequent real settings.
    """
    spec = AUDIT_SPEC[learner]

    h_match = True
    for scheduler_name, training_name in zip(
        spec["scheduler_h"],
        spec["training_h"],
    ):
        h_match &= settings_equal(
            anchor.get(scheduler_name, np.nan),
            post.get(scheduler_name, np.nan),
            training_name,
        )

    a_match = settings_equal(
        anchor.get(spec["scheduler_actuator"], np.nan),
        post.get(spec["scheduler_actuator"], np.nan),
        spec["training_actuator"],
    )

    return float(h_match), float(a_match)


# =============================================================================
# One learner audit
# =============================================================================

def audit_learner(
    learner: str,
    train_source: CSVSource,
    scheduler_source: CSVSource,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    learner = learner.upper()

    if learner not in AUDIT_SPEC:
        raise ValueError(f"Unsupported learner: {learner}")

    training_files = discover_full_coda_files(
        train_source,
        kind="metrics",
    )
    scheduler_files = discover_full_coda_files(
        scheduler_source,
        kind="scheduler",
    )

    common = sorted(
        set(training_files).intersection(scheduler_files)
    )

    if not common:
        raise RuntimeError(
            f"No matching Full-CODA train/scheduler files found for {learner}."
        )

    missing_training = sorted(
        set(scheduler_files).difference(training_files)
    )
    missing_scheduler = sorted(
        set(training_files).difference(scheduler_files)
    )

    if missing_training:
        print(
            f"[{learner}] warning: scheduler files without training logs: "
            f"{missing_training}"
        )

    if missing_scheduler:
        print(
            f"[{learner}] warning: training logs without scheduler files: "
            f"{missing_scheduler}"
        )

    reconstruction_rows = []
    event_rows = []

    for environment, seed in common:
        scheduler_df = scheduler_source.read_csv(
            scheduler_files[(environment, seed)]
        ).reset_index(drop=True)

        training_df = train_source.read_csv(
            training_files[(environment, seed)]
        ).reset_index(drop=True)

        events, reconstruction = reconstruct_scheduler_events(
            scheduler_df
        )

        matched_count = 0
        no_train_match = 0
        multi_candidate_count = 0

        # Do not reuse the same training report for two inheritance events.
        used_training_indices = set()

        for event in events.itertuples(index=False):
            anchor = scheduler_df.iloc[int(event.anchor_row)]
            post = scheduler_df.iloc[int(event.post_row)]

            agent = event.agent
            if agent is None or "agente_id" not in training_df.columns:
                candidates = training_df.iloc[0:0].copy()
            else:
                agent_rows = training_df[
                    training_df["agente_id"].astype(str).eq(str(agent))
                ].copy()

                mask = training_match_mask(
                    post_scheduler_row=post,
                    training_rows=agent_rows,
                    learner=learner,
                )

                candidates = agent_rows.loc[mask].copy()

                if used_training_indices:
                    candidates = candidates[
                        ~candidates.index.isin(
                            used_training_indices
                        )
                    ]

            candidate_count = int(len(candidates))

            if candidate_count > 1:
                multi_candidate_count += 1

            if candidate_count == 0:
                no_train_match += 1

                event_rows.append(
                    {
                        "learner": learner,
                        "environment": environment,
                        "seed": int(seed),
                        "trial": event.trial,
                        "anchor_row": int(event.anchor_row),
                        "post_row": int(event.post_row),
                        "post_time": event.post_time,
                        "agent": agent,
                        "matched": 0,
                        "match_type": "none",
                        "candidate_count_before_tiebreak": 0,
                        "train_order": np.nan,
                        "train_time": np.nan,
                        "time_diff": np.nan,
                        "E_h_event": np.nan,
                        "E_a_event": np.nan,
                        "sched_h_match": np.nan,
                        "sched_a_match": np.nan,
                    }
                )
                continue

            # Deterministic selection among repeated reports with the same
            # active configuration: nearest reported progress, then causal order.
            train_time = pd.to_numeric(
                candidates["timesteps_total"],
                errors="coerce",
            )

            candidates["_time_diff"] = (
                train_time - float(event.post_time)
            ).abs()

            if "causal_order" not in candidates.columns:
                candidates["causal_order"] = np.arange(
                    len(candidates),
                    dtype=np.int64,
                )

            candidates = candidates.sort_values(
                ["_time_diff", "causal_order"],
                kind="stable",
            )

            selected_index = int(candidates.index[0])
            used_training_indices.add(selected_index)

            train_row = training_df.loc[selected_index]

            E_h, E_a = audit_training_report(
                row=train_row,
                learner=learner,
            )

            sched_h, sched_a = audit_scheduler_anchor_to_post(
                anchor=anchor,
                post=post,
                learner=learner,
            )

            matched_count += 1

            event_rows.append(
                {
                    "learner": learner,
                    "environment": environment,
                    "seed": int(seed),
                    "trial": event.trial,
                    "anchor_row": int(event.anchor_row),
                    "post_row": int(event.post_row),
                    "post_time": event.post_time,
                    "agent": agent,
                    "matched": 1,
                    "match_type": "config_seq",
                    "candidate_count_before_tiebreak": candidate_count,
                    "train_order": _numeric(
                        train_row.get("causal_order", np.nan)
                    ),
                    "train_time": _numeric(
                        train_row.get("timesteps_total", np.nan)
                    ),
                    "time_diff": abs(
                        _numeric(
                            train_row.get(
                                "timesteps_total",
                                np.nan,
                            )
                        )
                        - float(event.post_time)
                    ),
                    "E_h_event": E_h,
                    "E_a_event": E_a,
                    "sched_h_match": sched_h,
                    "sched_a_match": sched_a,
                }
            )

        reconstruction_rows.append(
            {
                "learner": learner,
                "environment": environment,
                "seed": int(seed),
                "anchors": reconstruction["anchors"],
                "resolved": reconstruction["resolved"],
                "no_post": reconstruction["no_post"],
                "resolved_guided": reconstruction["resolved_guided"],
                "matched_train": matched_count,
                "no_train_match": no_train_match,
                "multi_candidate_before_tiebreak": multi_candidate_count,
            }
        )

    return (
        pd.DataFrame(reconstruction_rows),
        pd.DataFrame(event_rows),
    )


# =============================================================================
# Summaries
# =============================================================================

def _q1(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def _q3(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def build_seed_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        learner,
        environment,
        seed,
    ), group in events.groupby(
        ["learner", "environment", "seed"],
        sort=True,
    ):
        reconstructed = int(len(group))

        train_matched = group["matched"].eq(1)

        h_auditable = (
            train_matched
            & pd.to_numeric(
                group["E_h_event"],
                errors="coerce",
            ).notna()
        )

        a_auditable = (
            train_matched
            & pd.to_numeric(
                group["E_a_event"],
                errors="coerce",
            ).notna()
        )

        auditable_joint = h_auditable & a_auditable

        E_h = (
            float(
                pd.to_numeric(
                    group.loc[
                        h_auditable,
                        "E_h_event",
                    ],
                    errors="coerce",
                ).mean()
            )
            if h_auditable.any()
            else np.nan
        )

        E_a = (
            float(
                pd.to_numeric(
                    group.loc[
                        a_auditable,
                        "E_a_event",
                    ],
                    errors="coerce",
                ).mean()
            )
            if a_auditable.any()
            else np.nan
        )

        rows.append(
            {
                "learner": learner,
                "environment": environment,
                "seed": int(seed),
                "reconstructed_events": reconstructed,
                "train_matched_events": int(train_matched.sum()),
                "joint_auditable_events": int(auditable_joint.sum()),
                "audit_coverage_pct": (
                    100.0 * auditable_joint.sum() / reconstructed
                    if reconstructed
                    else np.nan
                ),
                "E_h": E_h,
                "E_a": E_a,
            }
        )

    return pd.DataFrame(rows)


def build_environment_summary(
    events: pd.DataFrame,
    seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        learner,
        environment,
    ), group in events.groupby(
        ["learner", "environment"],
        sort=True,
    ):
        seed_group = seed_summary[
            seed_summary["learner"].eq(learner)
            & seed_summary["environment"].eq(environment)
        ]

        reconstructed = int(len(group))

        joint = (
            group["matched"].eq(1)
            & pd.to_numeric(
                group["E_h_event"],
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                group["E_a_event"],
                errors="coerce",
            ).notna()
        )

        auditable = int(joint.sum())

        E_h_pooled = (
            float(
                pd.to_numeric(
                    group.loc[joint, "E_h_event"],
                    errors="coerce",
                ).mean()
            )
            if auditable
            else np.nan
        )

        E_a_pooled = (
            float(
                pd.to_numeric(
                    group.loc[joint, "E_a_event"],
                    errors="coerce",
                ).mean()
            )
            if auditable
            else np.nan
        )

        valid_Eh = pd.to_numeric(
            seed_group["E_h"],
            errors="coerce",
        ).dropna()

        valid_Ea = pd.to_numeric(
            seed_group["E_a"],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "learner": learner,
                "environment": environment,
                "reconstructed_events": reconstructed,
                "auditable_events": auditable,
                "audit_coverage_pct": (
                    100.0 * auditable / reconstructed
                    if reconstructed
                    else np.nan
                ),
                "E_h_pooled": E_h_pooled,
                "E_a_pooled": E_a_pooled,
                "E_h_seed_median": (
                    float(valid_Eh.median())
                    if not valid_Eh.empty
                    else np.nan
                ),
                "E_h_seed_q1": (
                    _q1(valid_Eh)
                    if not valid_Eh.empty
                    else np.nan
                ),
                "E_h_seed_q3": (
                    _q3(valid_Eh)
                    if not valid_Eh.empty
                    else np.nan
                ),
                "E_a_seed_median": (
                    float(valid_Ea.median())
                    if not valid_Ea.empty
                    else np.nan
                ),
                "E_a_seed_q1": (
                    _q1(valid_Ea)
                    if not valid_Ea.empty
                    else np.nan
                ),
                "E_a_seed_q3": (
                    _q3(valid_Ea)
                    if not valid_Ea.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_overall_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for learner, group in events.groupby(
        "learner",
        sort=True,
    ):
        reconstructed = int(len(group))

        joint = (
            group["matched"].eq(1)
            & pd.to_numeric(
                group["E_h_event"],
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                group["E_a_event"],
                errors="coerce",
            ).notna()
        )

        auditable = int(joint.sum())

        rows.append(
            {
                "learner": learner,
                "reconstructed_events": reconstructed,
                "auditable_events": auditable,
                "audit_coverage_pct": (
                    100.0 * auditable / reconstructed
                    if reconstructed
                    else np.nan
                ),
                "E_h": (
                    float(
                        pd.to_numeric(
                            group.loc[
                                joint,
                                "E_h_event",
                            ],
                            errors="coerce",
                        ).mean()
                    )
                    if auditable
                    else np.nan
                ),
                "E_a": (
                    float(
                        pd.to_numeric(
                            group.loc[
                                joint,
                                "E_a_event",
                            ],
                            errors="coerce",
                        ).mean()
                    )
                    if auditable
                    else np.nan
                ),
            }
        )

    # Optional combined row.
    reconstructed = int(len(events))

    joint = (
        events["matched"].eq(1)
        & pd.to_numeric(
            events["E_h_event"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            events["E_a_event"],
            errors="coerce",
        ).notna()
    )

    auditable = int(joint.sum())

    rows.append(
        {
            "learner": "ALL",
            "reconstructed_events": reconstructed,
            "auditable_events": auditable,
            "audit_coverage_pct": (
                100.0 * auditable / reconstructed
                if reconstructed
                else np.nan
            ),
            "E_h": (
                float(
                    pd.to_numeric(
                        events.loc[
                            joint,
                            "E_h_event",
                        ],
                        errors="coerce",
                    ).mean()
                )
                if auditable
                else np.nan
            ),
            "E_a": (
                float(
                    pd.to_numeric(
                        events.loc[
                            joint,
                            "E_a_event",
                        ],
                        errors="coerce",
                    ).mean()
                )
                if auditable
                else np.nan
            ),
        }
    )

    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct CODA guided inheritance events and audit "
            "requested-versus-effective execution consistency."
        )
    )

    parser.add_argument(
        "--ppo-train",
        type=Path,
        required=True,
        help="PPO training ZIP or extracted directory.",
    )
    parser.add_argument(
        "--ppo-scheduler",
        type=Path,
        required=True,
        help="PPO scheduler ZIP or extracted directory.",
    )
    parser.add_argument(
        "--sac-train",
        type=Path,
        required=True,
        help="SAC training ZIP or extracted directory.",
    )
    parser.add_argument(
        "--sac-scheduler",
        type=Path,
        required=True,
        help="SAC scheduler ZIP or extracted directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/audits/execution_consistency"
        ),
        help="Directory for audit CSV files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reconstruction_parts = []
    event_parts = []

    with CSVSource(args.ppo_train) as ppo_train, \
         CSVSource(args.ppo_scheduler) as ppo_scheduler, \
         CSVSource(args.sac_train) as sac_train, \
         CSVSource(args.sac_scheduler) as sac_scheduler:

        reconstruction, events = audit_learner(
            learner="PPO",
            train_source=ppo_train,
            scheduler_source=ppo_scheduler,
        )
        reconstruction_parts.append(reconstruction)
        event_parts.append(events)

        reconstruction, events = audit_learner(
            learner="SAC",
            train_source=sac_train,
            scheduler_source=sac_scheduler,
        )
        reconstruction_parts.append(reconstruction)
        event_parts.append(events)

    reconstruction = pd.concat(
        reconstruction_parts,
        ignore_index=True,
    )

    events = pd.concat(
        event_parts,
        ignore_index=True,
    )

    seed_summary = build_seed_summary(events)

    environment_summary = build_environment_summary(
        events,
        seed_summary,
    )

    overall_summary = build_overall_summary(events)

    # -------------------------------------------------------------------------
    # Save reproducible outputs
    # -------------------------------------------------------------------------

    reconstruction.to_csv(
        args.output_dir
        / "coda_inheritance_event_reconstruction_audit.csv",
        index=False,
    )

    events.to_csv(
        args.output_dir
        / "coda_inheritance_event_execution_audit.csv",
        index=False,
    )

    seed_summary.to_csv(
        args.output_dir
        / "coda_inheritance_execution_consistency_seed_summary.csv",
        index=False,
    )

    environment_summary.to_csv(
        args.output_dir
        / "coda_inheritance_execution_consistency_summary.csv",
        index=False,
    )

    overall_summary.to_csv(
        args.output_dir
        / "coda_inheritance_execution_consistency_overall.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\nCODA inheritance execution-consistency audit")
    print("=" * 72)

    print("\nReconstruction:")
    print(
        reconstruction.groupby("learner")[
            ["anchors", "resolved", "no_post"]
        ].sum()
    )

    print("\nEnvironment summary:")
    print(
        environment_summary[
            [
                "learner",
                "environment",
                "reconstructed_events",
                "auditable_events",
                "audit_coverage_pct",
                "E_h_pooled",
                "E_a_pooled",
            ]
        ].to_string(index=False)
    )

    print("\nOverall:")
    print(overall_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Sanity checks: mismatches should never be silently ignored.
    # -------------------------------------------------------------------------

    auditable = events["matched"].eq(1)

    h_failures = events[
        auditable
        & pd.to_numeric(
            events["E_h_event"],
            errors="coerce",
        ).eq(0.0)
    ]

    a_failures = events[
        auditable
        & pd.to_numeric(
            events["E_a_event"],
            errors="coerce",
        ).eq(0.0)
    ]

    if not h_failures.empty:
        print(
            f"\nWARNING: {len(h_failures)} auditable events "
            "contain an outer-configuration mismatch."
        )

    if not a_failures.empty:
        print(
            f"\nWARNING: {len(a_failures)} auditable events "
            "contain an actuator mismatch."
        )

    unmatched = events[events["matched"].eq(0)]
    if not unmatched.empty:
        print(
            f"\nNote: {len(unmatched)} reconstructable events "
            "could not be jointly matched to an exported learner report. "
            "They are excluded from E_h/E_a rather than counted as failures."
        )

    print(
        f"\nOutputs written to: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
