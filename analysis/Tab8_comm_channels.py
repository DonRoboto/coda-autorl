#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:30:46 2026

@author: yor5
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PPO_ZIP = Path("../results/ppo/ppo_scheduler.zip")
SAC_ZIP = Path("../results/sac/sac_scheduler.zip")

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

EXPECTED_SEEDS = 10

# Operational definitions
EPS = 1e-12

# "Practically near the upper diagnostic boundary"
S_NEAR_SATURATION = 0.95

# Exact numerical saturation
S_EXACT_SAT_ATOL = 1e-10

# PPO: Delta a_max = 0.004
PPO_MAX_INCREMENT = 0.004

# SAC:
# baseline = -d_a
# Delta a_max = 0.1 * d_a
ACTION_DIMENSIONS = {
    "HalfCheetah-v5": 6,
    "Hopper-v5": 3,
    "Swimmer-v5": 2,
    "Walker2d-v5": 6,
}

OUTPUT_EVENTS = "coda_channel_event_level.csv"
OUTPUT_AUDIT = "coda_event_reconstruction_audit.csv"
OUTPUT_SEEDS = "coda_channel_seed_level.csv"
OUTPUT_SUMMARY = "coda_channel_activity_summary.csv"

OUTPUT_MAIN_LATEX = "coda_channel_activity_main.tex"
OUTPUT_SUPP_LATEX = "coda_channel_activity_supplement.tex"


# ============================================================
# GENERIC HELPERS
# ============================================================

def as_numeric(df, column):
    """Return column as numeric Series, or NaN Series if absent."""
    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def scalar_numeric(value):
    return pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]


def safe_fraction(numerator, denominator):
    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def parse_scheduler_member(member):
    """
    Expected:
    scheduler/HalfCheetah-v5/scheduler_CODA_FULL_seed1042.csv
    """
    pattern = (
        r"scheduler/"
        r"(?P<environment>[^/]+)/"
        r"scheduler_CODA_FULL_seed"
        r"(?P<seed>\d+)\.csv$"
    )

    match = re.search(
        pattern,
        member,
    )

    if not match:
        raise ValueError(
            f"Could not parse scheduler path: {member}"
        )

    return (
        match.group("environment"),
        int(match.group("seed")),
    )


# ============================================================
# SYNTHETIC-ANCHOR IDENTIFICATION
# ============================================================

def identify_synthetic_rows(df, learner):
    """
    SAC exports is_synthetic explicitly.

    PPO does not export that flag in the final scheduler logs.
    In the observed PPO files, synthetic inherited anchors have
    NaN across all scheduler/O2I audit variables, whereas real
    reports contain finite zero/nonzero values.
    """

    if "is_synthetic" in df.columns:
        raw = df["is_synthetic"]

        # Handle bool, 0/1, and text
        if raw.dtype == bool:
            return raw.fillna(False)

        normalized = (
            raw.astype(str)
            .str.strip()
            .str.lower()
        )

        return normalized.isin(
            [
                "true",
                "1",
                "1.0",
            ]
        )

    # PPO reconstruction
    audit_columns = [
        "guided_update",
        "gp_data_count",
        "o2i_uncertainty_raw",
        "o2i_uncertainty_thresholded",
        "o2i_warmup_gain",
        "o2i_uncertainty_effective",
        "entropy_increment",
    ]

    present = [
        col
        for col in audit_columns
        if col in df.columns
    ]

    if len(present) < 5:
        raise RuntimeError(
            f"{learner}: insufficient scheduler columns "
            "to reconstruct synthetic anchors."
        )

    return df[present].isna().all(axis=1)


# ============================================================
# ACTUATOR NORMALIZATION
# ============================================================

def physical_normalized_magnitude(
    row,
    learner,
    environment,
):
    """
    Independently reconstruct the normalized actuator displacement.

    PPO:
        entropy_increment / 0.004

    SAC:
        actuator_increment / (0.1 * action_dimension)

    Under the experimental mapping this should agree with U_eff,
    up to floating-point tolerance.
    """

    if learner == "PPO":

        increment = scalar_numeric(
            row.get(
                "entropy_increment",
                np.nan,
            )
        )

        if not np.isfinite(increment):
            return np.nan

        return float(
            increment
            / PPO_MAX_INCREMENT
        )

    if learner == "SAC":

        increment = scalar_numeric(
            row.get(
                "actuator_increment",
                np.nan,
            )
        )

        if not np.isfinite(increment):
            return np.nan

        d_a = ACTION_DIMENSIONS[
            environment
        ]

        max_increment = (
            0.1 * d_a
        )

        return float(
            increment
            / max_increment
        )

    raise ValueError(
        f"Unknown learner: {learner}"
    )


# ============================================================
# EVENT RECONSTRUCTION
# ============================================================

def reconstruct_events(
    df,
    learner,
    environment,
    seed,
):
    """
    Reconstruct one execution event as:

        synthetic inherited anchor
                ->
        first subsequent real report
        from the same Trial

    We do NOT count every training report as an O2I event.
    """

    df = (
        df.copy()
        .reset_index(drop=True)
    )

    df["_row_order"] = np.arange(
        len(df),
        dtype=int,
    )

    df["_synthetic"] = (
        identify_synthetic_rows(
            df,
            learner,
        )
    )

    events = []
    audit = []

    anchor_indices = (
        df.index[
            df["_synthetic"]
        ]
        .tolist()
    )

    for anchor_idx in anchor_indices:

        trial = df.loc[
            anchor_idx,
            "Trial"
        ]

        # ----------------------------------------------------
        # First subsequent real report from the same receiver
        # ----------------------------------------------------

        candidates = df.index[
            (df.index > anchor_idx)
            &
            (df["Trial"] == trial)
            &
            (~df["_synthetic"])
        ]

        if len(candidates) == 0:

            audit.append({
                "learner":
                    learner,

                "environment":
                    environment,

                "training_seed":
                    seed,

                "anchor_row":
                    anchor_idx,

                "Trial":
                    trial,

                "status":
                    "no_post_anchor_report",
            })

            continue

        report_idx = int(
            candidates[0]
        )

        # ----------------------------------------------------
        # Make sure another synthetic anchor for this trial
        # did not appear first.
        # ----------------------------------------------------

        between = df[
            (df.index > anchor_idx)
            &
            (df.index < report_idx)
            &
            (df["Trial"] == trial)
            &
            df["_synthetic"]
        ]

        if not between.empty:

            audit.append({
                "learner":
                    learner,

                "environment":
                    environment,

                "training_seed":
                    seed,

                "anchor_row":
                    anchor_idx,

                "Trial":
                    trial,

                "status":
                    "ambiguous_multiple_anchors",
            })

            continue

        row = df.loc[
            report_idx
        ]

        guided = scalar_numeric(
            row.get(
                "guided_update",
                np.nan,
            )
        )

        gp_count = scalar_numeric(
            row.get(
                "gp_data_count",
                np.nan,
            )
        )

        raw = scalar_numeric(
            row.get(
                "o2i_uncertainty_raw",
                np.nan,
            )
        )

        thresholded = scalar_numeric(
            row.get(
                "o2i_uncertainty_thresholded",
                np.nan,
            )
        )

        warmup = scalar_numeric(
            row.get(
                "o2i_warmup_gain",
                np.nan,
            )
        )

        effective = scalar_numeric(
            row.get(
                "o2i_uncertainty_effective",
                np.nan,
            )
        )

        # ----------------------------------------------------
        # O2I operational state
        # ----------------------------------------------------

        if (
            not np.isfinite(guided)
            or guided < 0.5
        ):
            event_status = "not_guided"

        elif (
            not np.isfinite(raw)
            or raw <= EPS
        ):
            event_status = (
                "no_excess_uncertainty"
            )

        elif (
            not np.isfinite(thresholded)
            or thresholded <= EPS
        ):
            event_status = (
                "below_threshold"
            )

        elif (
            not np.isfinite(warmup)
            or warmup <= EPS
        ):
            event_status = (
                "warmup_blocked"
            )

        elif (
            np.isfinite(effective)
            and effective > EPS
        ):
            event_status = "active"

        else:
            event_status = (
                "other_zero_effective"
            )

        # ----------------------------------------------------
        # Magnitude
        #
        # Primary operational quantity:
        # U_eff
        #
        # Secondary audit:
        # actuator increment / maximum allowed increment
        # ----------------------------------------------------

        normalized_from_u = (
            float(effective)
            if np.isfinite(effective)
            else np.nan
        )

        normalized_from_actuator = (
            physical_normalized_magnitude(
                row,
                learner,
                environment,
            )
        )

        consistency_error = np.nan

        if (
            np.isfinite(
                normalized_from_u
            )
            and
            np.isfinite(
                normalized_from_actuator
            )
        ):
            consistency_error = abs(
                normalized_from_u
                -
                normalized_from_actuator
            )

        event = {
            "learner":
                learner,

            "environment":
                environment,

            "training_seed":
                seed,

            "Trial":
                trial,

            "anchor_row":
                anchor_idx,

            "anchor_time":
                scalar_numeric(
                    df.loc[
                        anchor_idx,
                        "Time"
                    ]
                ),

            "report_row":
                report_idx,

            "report_time":
                scalar_numeric(
                    row.get(
                        "Time",
                        np.nan
                    )
                ),

            "guided_update":
                guided,

            "gp_data_count":
                gp_count,

            "u_raw":
                raw,

            "u_thresholded":
                thresholded,

            "warmup_gain":
                warmup,

            "u_effective":
                effective,

            "normalized_magnitude":
                normalized_from_u,

            "normalized_from_actuator":
                normalized_from_actuator,

            "actuator_consistency_error":
                consistency_error,

            "event_status":
                event_status,
        }

        # ----------------------------------------------------
        # Learner-specific actuator audit values
        # ----------------------------------------------------

        if learner == "PPO":

            event[
                "actuator_increment"
            ] = scalar_numeric(
                row.get(
                    "entropy_increment",
                    np.nan,
                )
            )

            event[
                "applied_actuator"
            ] = scalar_numeric(
                row.get(
                    "entropy_coeff",
                    np.nan,
                )
            )

        else:

            event[
                "actuator_increment"
            ] = scalar_numeric(
                row.get(
                    "actuator_increment",
                    np.nan,
                )
            )

            event[
                "applied_actuator"
            ] = scalar_numeric(
                row.get(
                    "target_entropy",
                    np.nan,
                )
            )

        events.append(
            event
        )

        audit.append({
            "learner":
                learner,

            "environment":
                environment,

            "training_seed":
                seed,

            "anchor_row":
                anchor_idx,

            "Trial":
                trial,

            "status":
                "resolved",
        })

    return (
        pd.DataFrame(events),
        pd.DataFrame(audit),
        df,
    )


# ============================================================
# I2O OPERATIONAL SUMMARY
# ============================================================

def summarize_i2o(df):
    """
    I2O is summarized over REAL learner reports.

    Seed-level quantities:
      - valid diagnostic fraction
      - median S
      - Q1(S), Q3(S), IQR(S)
      - exact S=1 rate
      - practical near-saturation S>=0.95 rate
    """

    real = df[
        ~df["_synthetic"]
    ].copy()

    validity = as_numeric(
        real,
        "policy_update_state_valid",
    )

    state = as_numeric(
        real,
        "policy_update_state",
    )

    valid_mask = (
        np.isfinite(validity)
        &
        (validity >= 0.5)
        &
        np.isfinite(state)
    )

    n_reports = len(
        real
    )

    n_valid = int(
        valid_mask.sum()
    )

    valid_rate = safe_fraction(
        n_valid,
        n_reports,
    )

    if n_valid == 0:

        return {
            "i2o_n_reports":
                n_reports,

            "i2o_n_valid":
                0,

            "i2o_valid_rate":
                valid_rate,

            "i2o_state_median":
                np.nan,

            "i2o_state_q1":
                np.nan,

            "i2o_state_q3":
                np.nan,

            "i2o_state_iqr":
                np.nan,

            "i2o_exact_saturation_rate":
                np.nan,

            "i2o_near_saturation_rate":
                np.nan,
        }

    s = (
        state[
            valid_mask
        ]
        .astype(float)
    )

    s_median = float(
        s.median()
    )

    s_q1 = float(
        s.quantile(0.25)
    )

    s_q3 = float(
        s.quantile(0.75)
    )

    s_iqr = float(
        s_q3 - s_q1
    )

    exact_saturation = float(
        np.mean(
            np.isclose(
                s,
                1.0,
                rtol=0.0,
                atol=S_EXACT_SAT_ATOL,
            )
        )
    )

    near_saturation = float(
        np.mean(
            s >= S_NEAR_SATURATION
        )
    )

    return {
        "i2o_n_reports":
            n_reports,

        "i2o_n_valid":
            n_valid,

        "i2o_valid_rate":
            valid_rate,

        "i2o_state_median":
            s_median,

        "i2o_state_q1":
            s_q1,

        "i2o_state_q3":
            s_q3,

        "i2o_state_iqr":
            s_iqr,

        "i2o_exact_saturation_rate":
            exact_saturation,

        "i2o_near_saturation_rate":
            near_saturation,
    }


# ============================================================
# SEED-LEVEL EVENT SUMMARY
# ============================================================

def summarize_events_for_seed(
    events,
):
    """
    Summaries are calculated WITHIN the training seed.
    The training seed remains the independent replicate.
    """

    if events.empty:
        return {
            "n_reconstructed_events":
                0,

            "n_guided_events":
                0,

            "o2i_active_rate":
                np.nan,

            "no_excess_rate":
                np.nan,

            "threshold_blocked_rate":
                np.nan,

            "warmup_blocked_rate":
                np.nan,

            "other_zero_rate":
                np.nan,

            "active_magnitude_median":
                np.nan,

            "active_magnitude_q1":
                np.nan,

            "active_magnitude_q3":
                np.nan,

            "guided_u_raw_median":
                np.nan,

            "guided_u_thresholded_median":
                np.nan,

            "guided_warmup_median":
                np.nan,

            "max_actuator_consistency_error":
                np.nan,
        }

    guided = events[
        (
            pd.to_numeric(
                events["guided_update"],
                errors="coerce"
            )
            >= 0.5
        )
    ].copy()

    n_guided = len(
        guided
    )

    def event_rate(status):

        if n_guided == 0:
            return np.nan

        return float(
            np.mean(
                guided[
                    "event_status"
                ]
                == status
            )
        )

    active = guided[
        guided[
            "event_status"
        ]
        == "active"
    ].copy()

    active_magnitude = pd.to_numeric(
        active[
            "normalized_magnitude"
        ],
        errors="coerce",
    )

    active_magnitude = active_magnitude[
        np.isfinite(
            active_magnitude
        )
    ]

    if active_magnitude.empty:

        mag_med = np.nan
        mag_q1 = np.nan
        mag_q3 = np.nan

    else:

        mag_med = float(
            active_magnitude.median()
        )

        mag_q1 = float(
            active_magnitude.quantile(
                0.25
            )
        )

        mag_q3 = float(
            active_magnitude.quantile(
                0.75
            )
        )

    consistency = pd.to_numeric(
        events[
            "actuator_consistency_error"
        ],
        errors="coerce",
    )

    consistency = consistency[
        np.isfinite(
            consistency
        )
    ]

    max_consistency_error = (
        float(
            consistency.max()
        )
        if not consistency.empty
        else np.nan
    )

    guided_raw = pd.to_numeric(
        guided["u_raw"],
        errors="coerce",
    )

    guided_thr = pd.to_numeric(
        guided["u_thresholded"],
        errors="coerce",
    )

    guided_warmup = pd.to_numeric(
        guided["warmup_gain"],
        errors="coerce",
    )

    return {
        "n_reconstructed_events":
            len(events),

        "n_guided_events":
            n_guided,

        "o2i_active_rate":
            event_rate(
                "active"
            ),

        "no_excess_rate":
            event_rate(
                "no_excess_uncertainty"
            ),

        "threshold_blocked_rate":
            event_rate(
                "below_threshold"
            ),

        "warmup_blocked_rate":
            event_rate(
                "warmup_blocked"
            ),

        "other_zero_rate":
            event_rate(
                "other_zero_effective"
            ),

        "active_magnitude_median":
            mag_med,

        "active_magnitude_q1":
            mag_q1,

        "active_magnitude_q3":
            mag_q3,

        "guided_u_raw_median":
            (
                float(
                    guided_raw.median()
                )
                if guided_raw.notna().any()
                else np.nan
            ),

        "guided_u_thresholded_median":
            (
                float(
                    guided_thr.median()
                )
                if guided_thr.notna().any()
                else np.nan
            ),

        "guided_warmup_median":
            (
                float(
                    guided_warmup.median()
                )
                if guided_warmup.notna().any()
                else np.nan
            ),

        "max_actuator_consistency_error":
            max_consistency_error,
    }


# ============================================================
# PROCESS COMPLETE ZIP
# ============================================================

def process_zip(
    zip_path,
    learner,
):
    if not zip_path.exists():
        raise FileNotFoundError(
            zip_path.resolve()
        )

    all_events = []
    all_audit = []
    seed_rows = []

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zf:

        members = [
            name
            for name in zf.namelist()
            if (
                name.startswith(
                    "scheduler/"
                )
                and
                "scheduler_CODA_FULL_seed"
                in name
                and
                name.endswith(".csv")
            )
        ]

        print(
            f"{learner}: "
            f"{len(members)} Full-CODA "
            "scheduler files"
        )

        for member in sorted(
            members
        ):

            (
                environment,
                seed,
            ) = parse_scheduler_member(
                member
            )

            with zf.open(
                member
            ) as f:

                df = pd.read_csv(
                    f
                )

            (
                events,
                audit,
                augmented_df,
            ) = reconstruct_events(
                df,
                learner,
                environment,
                seed,
            )

            i2o_stats = summarize_i2o(
                augmented_df
            )

            event_stats = (
                summarize_events_for_seed(
                    events
                )
            )

            all_events.append(
                events
            )

            all_audit.append(
                audit
            )

            seed_rows.append({
                "learner":
                    learner,

                "environment":
                    environment,

                "training_seed":
                    seed,

                **event_stats,
                **i2o_stats,
            })

    return (
        pd.concat(
            all_events,
            ignore_index=True,
        ),

        pd.concat(
            all_audit,
            ignore_index=True,
        ),

        pd.DataFrame(
            seed_rows
        ),
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

(
    ppo_events,
    ppo_audit,
    ppo_seed,
) = process_zip(
    PPO_ZIP,
    "PPO",
)

(
    sac_events,
    sac_audit,
    sac_seed,
) = process_zip(
    SAC_ZIP,
    "SAC",
)


events = pd.concat(
    [
        ppo_events,
        sac_events,
    ],
    ignore_index=True,
)

audit = pd.concat(
    [
        ppo_audit,
        sac_audit,
    ],
    ignore_index=True,
)

seed_summary = pd.concat(
    [
        ppo_seed,
        sac_seed,
    ],
    ignore_index=True,
)


# ============================================================
# VALIDATION
# ============================================================

seed_counts = (
    seed_summary
    .groupby(
        [
            "learner",
            "environment",
        ]
    )[
        "training_seed"
    ]
    .nunique()
)

print(
    "\n"
    + "=" * 80
)

print(
    "SEEDS PER LEARNER/ENVIRONMENT"
)

print(
    "=" * 80
)

print(
    seed_counts
)


bad_seed_counts = seed_counts[
    seed_counts != EXPECTED_SEEDS
]

if not bad_seed_counts.empty:

    raise RuntimeError(
        "\nIncomplete design:\n"
        f"{bad_seed_counts}"
    )


# ------------------------------------------------------------
# Event reconstruction quality
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 80
)

print(
    "EVENT RECONSTRUCTION AUDIT"
)

print(
    "=" * 80
)

audit_counts = (
    audit["status"]
    .value_counts(
        dropna=False
    )
)

print(
    audit_counts
)


if (
    "ambiguous_multiple_anchors"
    in audit_counts.index
    and
    audit_counts[
        "ambiguous_multiple_anchors"
    ] > 0
):
    raise RuntimeError(
        "Ambiguous event reconstruction detected."
    )


# ------------------------------------------------------------
# Event states
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 80
)

print(
    "EVENT STATUS COUNTS"
)

print(
    "=" * 80
)

print(
    events
    .groupby(
        [
            "learner",
            "environment",
            "event_status",
        ]
    )
    .size()
)


# ------------------------------------------------------------
# Actuator mapping consistency
# ------------------------------------------------------------

finite_consistency = pd.to_numeric(
    events[
        "actuator_consistency_error"
    ],
    errors="coerce",
)

finite_consistency = finite_consistency[
    np.isfinite(
        finite_consistency
    )
]

print(
    "\nMaximum |U_eff - normalized actuator increment|:"
)

if finite_consistency.empty:

    print(
        "No finite actuator consistency comparisons."
    )

else:

    print(
        finite_consistency.max()
    )


# ============================================================
# AGGREGATE OVER INDEPENDENT TRAINING SEEDS
# ============================================================

AGGREGATE_METRICS = [
    # Event-level operational quantities
    "n_reconstructed_events",
    "n_guided_events",

    "o2i_active_rate",
    "active_magnitude_median",

    "warmup_blocked_rate",
    "threshold_blocked_rate",
    "no_excess_rate",
    "other_zero_rate",

    "guided_u_raw_median",
    "guided_u_thresholded_median",
    "guided_warmup_median",

    # I2O
    "i2o_valid_rate",
    "i2o_state_median",
    "i2o_state_iqr",

    "i2o_near_saturation_rate",
    "i2o_exact_saturation_rate",

    # Implementation audit
    "max_actuator_consistency_error",
]


aggregate_rows = []

for (
    learner,
    environment,
), group in seed_summary.groupby(
    [
        "learner",
        "environment",
    ]
):

    result = {
        "learner":
            learner,

        "environment":
            environment,

        "n_seeds":
            group[
                "training_seed"
            ].nunique(),
    }

    for metric in AGGREGATE_METRICS:

        values = pd.to_numeric(
            group[
                metric
            ],
            errors="coerce",
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if values.empty:

            result[
                f"{metric}_median"
            ] = np.nan

            result[
                f"{metric}_q1"
            ] = np.nan

            result[
                f"{metric}_q3"
            ] = np.nan

            continue

        result[
            f"{metric}_median"
        ] = float(
            values.median()
        )

        result[
            f"{metric}_q1"
        ] = float(
            values.quantile(
                0.25
            )
        )

        result[
            f"{metric}_q3"
        ] = float(
            values.quantile(
                0.75
            )
        )

    aggregate_rows.append(
        result
    )


summary = pd.DataFrame(
    aggregate_rows
)


# ============================================================
# ORDER
# ============================================================

learner_order = {
    "PPO": 0,
    "SAC": 1,
}

environment_order = {
    env: idx
    for idx, env
    in enumerate(
        ENVIRONMENTS
    )
}

summary["_learner_order"] = (
    summary[
        "learner"
    ]
    .map(
        learner_order
    )
)

summary["_environment_order"] = (
    summary[
        "environment"
    ]
    .map(
        environment_order
    )
)

summary = (
    summary
    .sort_values(
        [
            "_learner_order",
            "_environment_order",
        ]
    )
    .drop(
        columns=[
            "_learner_order",
            "_environment_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE NUMERICAL OUTPUTS
# ============================================================

events.to_csv(
    OUTPUT_EVENTS,
    index=False,
)

audit.to_csv(
    OUTPUT_AUDIT,
    index=False,
)

seed_summary.to_csv(
    OUTPUT_SEEDS,
    index=False,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# PRINT MAIN SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "CHANNEL ACTIVITY SUMMARY"
)

print(
    "=" * 120
)

display_columns = [
    "learner",
    "environment",
    "n_seeds",

    "n_guided_events_median",

    "o2i_active_rate_median",
    "active_magnitude_median_median",

    "warmup_blocked_rate_median",

    "i2o_valid_rate_median",
    "i2o_state_median_median",
    "i2o_state_iqr_median",

    "i2o_near_saturation_rate_median",
    "i2o_exact_saturation_rate_median",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# LATEX HELPERS
# ============================================================

def median_iqr(
    row,
    metric,
    scale=1.0,
    digits=1,
):
    med = (
        row[
            f"{metric}_median"
        ]
        * scale
    )

    q1 = (
        row[
            f"{metric}_q1"
        ]
        * scale
    )

    q3 = (
        row[
            f"{metric}_q3"
        ]
        * scale
    )

    if not (
        np.isfinite(med)
        and
        np.isfinite(q1)
        and
        np.isfinite(q3)
    ):
        return "--"

    return (
        f"{med:.{digits}f} "
        f"[{q1:.{digits}f}, "
        f"{q3:.{digits}f}]"
    )


# ============================================================
# MAIN-PAPER LATEX TABLE
# ============================================================

main_caption = (
    "Operational activity of the Full-CODA communication "
    "channels. Each statistic is first computed within an "
    "independent training seed and then reported as the "
    r"median $[Q_1,Q_3]$ across ten seeds. O2I activity "
    "is evaluated over reconstructed guided inheritance "
    "events. Active magnitude is the effective uncertainty "
    r"$U^{\mathrm{eff}}$, which equals the applied actuator "
    "increment normalized by its learner-specific maximum. "
    r"$S\geq0.95$ denotes practical near-saturation of the "
    "I2O diagnostic.}"
)

main_lines = [
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{\justifying",
    main_caption,
    r"\label{tab:channel-activity}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2.5pt}",
    r"\renewcommand{\arraystretch}{1.08}",
    r"\begin{tabular}{@{}llccccccc@{}}",
    r"\toprule",
    (
        r"\textbf{Learner} & "
        r"\textbf{Environment} & "
        r"\textbf{Guided events} & "
        r"\textbf{O2I active (\%)} & "
        r"\textbf{Active magnitude} & "
        r"\textbf{Warm-up blocked (\%)} & "
        r"\textbf{I2O valid (\%)} & "
        r"\textbf{Median $S$} & "
        r"\textbf{$S\geq0.95$ (\%)} \\"
    ),
    r"\midrule",
]


previous_learner = None

for _, row in summary.iterrows():

    learner = row["learner"]

    if (
        previous_learner is not None
        and learner != previous_learner
    ):
        main_lines.append(
            r"\midrule"
        )

    main_lines.append(
        f"{learner} & "
        f"{row['environment']} & "
        f"{median_iqr(row, 'n_guided_events', 1.0, 1)} & "
        f"{median_iqr(row, 'o2i_active_rate', 100.0, 1)} & "
        f"{median_iqr(row, 'active_magnitude_median', 1.0, 2)} & "
        f"{median_iqr(row, 'warmup_blocked_rate', 100.0, 1)} & "
        f"{median_iqr(row, 'i2o_valid_rate', 100.0, 1)} & "
        f"{median_iqr(row, 'i2o_state_median', 1.0, 3)} & "
        f"{median_iqr(row, 'i2o_near_saturation_rate', 100.0, 1)} "
        r"\\"
    )

    previous_learner = learner


main_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
])


# Optional safety check
for i, item in enumerate(main_lines):
    if not isinstance(item, str):
        raise TypeError(
            f"main_lines[{i}] is {type(item).__name__}, not str: {item}"
        )


main_latex = "\n".join(
    main_lines
)

Path(
    OUTPUT_MAIN_LATEX
).write_text(
    main_latex,
    encoding="utf-8",
)


# ============================================================
# SUPPLEMENTARY LATEX TABLE
# ============================================================

supp_caption = (
    "Detailed operational decomposition of Full-CODA "
    "communication activity. Statistics are computed "
    "within each training seed and summarized across "
    r"ten seeds as median $[Q_1,Q_3]$. O2I inactivity "
    "categories are expressed relative to reconstructed "
    "guided events. Exact saturation denotes numerical "
    r"$S=1$, whereas near saturation denotes $S\geq0.95$.}"
)

supp_lines = [
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{\justifying",
    supp_caption,
    r"\label{tab:channel-activity-supp}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2pt}",
    r"\renewcommand{\arraystretch}{1.08}",
    r"\begin{tabular}{@{}llcccccccc@{}}",
    r"\toprule",
    (
        r"\textbf{Learner} & "
        r"\textbf{Environment} & "
        r"\textbf{$U^{raw}$} & "
        r"\textbf{$\omega$} & "
        r"\textbf{No excess (\%)} & "
        r"\textbf{Threshold (\%)} & "
        r"\textbf{Warm-up (\%)} & "
        r"\textbf{IQR($S$)} & "
        r"\textbf{$S\geq.95$ (\%)} & "
        r"\textbf{$S=1$ (\%)} \\"
    ),
    r"\midrule",
]


previous_learner = None

for _, row in summary.iterrows():

    learner = row["learner"]

    if (
        previous_learner is not None
        and learner != previous_learner
    ):
        supp_lines.append(
            r"\midrule"
        )

    supp_lines.append(
        f"{learner} & "
        f"{row['environment']} & "
        f"{median_iqr(row, 'guided_u_raw_median', 1.0, 3)} & "
        f"{median_iqr(row, 'guided_warmup_median', 1.0, 3)} & "
        f"{median_iqr(row, 'no_excess_rate', 100.0, 1)} & "
        f"{median_iqr(row, 'threshold_blocked_rate', 100.0, 1)} & "
        f"{median_iqr(row, 'warmup_blocked_rate', 100.0, 1)} & "
        f"{median_iqr(row, 'i2o_state_iqr', 1.0, 3)} & "
        f"{median_iqr(row, 'i2o_near_saturation_rate', 100.0, 1)} & "
        f"{median_iqr(row, 'i2o_exact_saturation_rate', 100.0, 1)} "
        r"\\"
    )

    previous_learner = learner


supp_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
])


for i, item in enumerate(supp_lines):
    if not isinstance(item, str):
        raise TypeError(
            f"supp_lines[{i}] is {type(item).__name__}, not str: {item}"
        )


supp_latex = "\n".join(
    supp_lines
)

Path(
    OUTPUT_SUPP_LATEX
).write_text(
    supp_latex,
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "MAIN-PAPER LATEX TABLE"
)

print(
    "=" * 120
    + "\n"
)

print(
    main_latex
)

print(
    "\n"
    + "=" * 120
)

print(
    "SUPPLEMENTARY LATEX TABLE"
)

print(
    "=" * 120
    + "\n"
)

print(
    supp_latex
)

print(
    "\nGenerated files:"
)

for path in [
    OUTPUT_EVENTS,
    OUTPUT_AUDIT,
    OUTPUT_SEEDS,
    OUTPUT_SUMMARY,
    OUTPUT_MAIN_LATEX,
    OUTPUT_SUPP_LATEX,
]:
    print(
        f"  {path}"
    )