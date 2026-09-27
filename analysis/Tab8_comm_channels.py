#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table 8: Full-CODA operational communication-channel activity.

This script reconstructs guided inheritance events from Full-CODA
scheduler logs and summarizes the realized activity of the O2I and I2O
communication channels.

For each learner (PPO/SAC), environment, and training seed, the script:

1. Reads the archived Full-CODA scheduler logs.
2. Identifies synthetic inheritance anchors.
3. Reconstructs one operational event as:

       synthetic inherited anchor
           ->
       first subsequent genuine scheduler report
       from the same receiver

4. Classifies O2I event state using the logged uncertainty pipeline:
       no excess uncertainty,
       below threshold,
       warm-up blocked,
       active,
       other zero-effective.

5. Audits the normalized actuator displacement against U_eff.
6. Summarizes I2O diagnostic validity and realized diagnostic state S.
7. Computes seed-level summaries first.
8. Aggregates the ten independent training seeds using median [Q1, Q3].
9. Saves auditable event-level, reconstruction, seed-level, and
   across-seed numerical artifacts.
10. Generates LaTeX table bodies for the main-paper and supplementary
    operational summaries.

Expected repository layout
--------------------------

coda-autorl/
├── analysis/
│   └── Tab8_comm_channels.py
└── results/
    ├── ppo/
    │   └── scheduler.zip
    └── sac/
        └── scheduler.zip

Outputs
-------

results/analysis/communication_channels/
├── coda_channel_event_level.csv
├── coda_event_reconstruction_audit.csv
├── coda_channel_seed_level.csv
├── coda_channel_seed_counts.csv
├── coda_channel_activity_summary.csv
├── coda_channel_activity_main_table_body.tex
└── coda_channel_activity_supplement_table_body.tex
"""

from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

# Works when this file is stored under:
#     <repo_root>/analysis/Tab8_comm_channels.py
#
# The fallback also allows execution from an interactive session.
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent

PPO_ZIP = (
    REPO_ROOT
    / "results"
    / "ppo"
    / "ppo_scheduler.zip"
)

SAC_ZIP = (
    REPO_ROOT
    / "results"
    / "sac"
    / "sac_scheduler.zip"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "communication_channels"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_EVENTS = (
    OUTPUT_DIR
    / "coda_channel_event_level.csv"
)

OUTPUT_AUDIT = (
    OUTPUT_DIR
    / "coda_event_reconstruction_audit.csv"
)

OUTPUT_SEEDS = (
    OUTPUT_DIR
    / "coda_channel_seed_level.csv"
)

OUTPUT_SEED_COUNTS = (
    OUTPUT_DIR
    / "coda_channel_seed_counts.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "coda_channel_activity_summary.csv"
)

OUTPUT_MAIN_LATEX = (
    OUTPUT_DIR
    / "coda_channel_activity_main_table_body.tex"
)

OUTPUT_SUPP_LATEX = (
    OUTPUT_DIR
    / "coda_channel_activity_supplement_table_body.tex"
)


# ============================================================
# Analysis configuration
# ============================================================

ENVIRONMENTS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

LEARNERS = [
    "PPO",
    "SAC",
]

EXPECTED_SEEDS = 10

# Operational threshold used to treat effectively-zero quantities.
EPS = 1e-12

# "Practically near the upper diagnostic boundary".
S_NEAR_SATURATION = 0.95

# Exact numerical saturation.
S_EXACT_SAT_ATOL = 1e-10

# PPO: Delta a_max = 0.004.
PPO_MAX_INCREMENT = 0.004

# SAC:
# baseline = -d_a
# Delta a_max = 0.1 * action_dimension
ACTION_DIMENSIONS = {
    "HalfCheetah-v5": 6,
    "Hopper-v5": 3,
    "Swimmer-v5": 2,
    "Walker2d-v5": 6,
}


# ============================================================
# Generic helpers
# ============================================================

def validate_zip_path(
    zip_path,
):
    """
    Validate one scheduler ZIP before analysis.
    """
    zip_path = Path(
        zip_path
    )

    if not zip_path.exists():
        raise FileNotFoundError(
            "Scheduler archive not found:\n"
            f"{zip_path}"
        )

    if not zipfile.is_zipfile(
        zip_path
    ):
        raise ValueError(
            f"Not a valid ZIP archive: {zip_path}"
        )


def as_numeric(
    df,
    column,
):
    """
    Return a column as numeric Series, or a NaN Series if absent.
    """
    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[
            column
        ],
        errors="coerce",
    )


def scalar_numeric(
    value,
):
    """
    Convert one scalar-like value to numeric, returning NaN if invalid.
    """
    return pd.to_numeric(
        pd.Series(
            [
                value
            ]
        ),
        errors="coerce",
    ).iloc[
        0
    ]


def safe_fraction(
    numerator,
    denominator,
):
    """
    Safe scalar fraction.
    """
    if denominator == 0:
        return np.nan

    return float(
        numerator
        / denominator
    )


def parse_scheduler_member(
    member,
):
    """
    Parse environment and training seed from a Full-CODA scheduler path.

    The function matches the basename and searches the environment among
    the path components, so it tolerates an additional top-level folder
    inside the archive.
    """
    name = PurePosixPath(
        member
    ).name

    match = re.fullmatch(
        r"scheduler_CODA_FULL_seed(?P<seed>\d+)\.csv",
        name,
    )

    if match is None:

        raise ValueError(
            "Could not parse Full-CODA scheduler filename: "
            f"{member}"
        )

    parts = PurePosixPath(
        member
    ).parts

    environment_matches = [
        part
        for part in parts
        if part in ENVIRONMENTS
    ]

    if len(
        environment_matches
    ) != 1:

        raise ValueError(
            "Could not uniquely infer environment from scheduler path: "
            f"{member}"
        )

    return (
        environment_matches[
            0
        ],
        int(
            match.group(
                "seed"
            )
        ),
    )


# ============================================================
# Synthetic-anchor identification
# ============================================================

def identify_synthetic_rows(
    df,
    learner,
):
    """
    Identify synthetic inherited-reference rows.

    SAC exports `is_synthetic` explicitly.

    PPO does not export that flag in the final scheduler logs. In the
    archived PPO scheduler files, synthetic inherited anchors have NaN
    across the scheduler/O2I audit variables, whereas genuine reports
    contain finite zero/nonzero values.
    """
    if (
        "is_synthetic"
        in df.columns
    ):

        raw = df[
            "is_synthetic"
        ]

        if raw.dtype == bool:

            return raw.fillna(
                False
            )

        normalized = (
            raw
            .astype(str)
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
        column
        for column in audit_columns
        if column in df.columns
    ]

    if len(
        present
    ) < 5:

        raise RuntimeError(
            f"{learner}: insufficient scheduler columns "
            "to reconstruct synthetic anchors."
        )

    return (
        df[
            present
        ]
        .isna()
        .all(
            axis=1
        )
    )


# ============================================================
# Actuator normalization
# ============================================================

def physical_normalized_magnitude(
    row,
    learner,
    environment,
):
    """
    Independently reconstruct normalized actuator displacement.

    PPO:
        entropy_increment / 0.004

    SAC:
        actuator_increment / (0.1 * action_dimension)

    Under the experimental mapping, this should agree with U_eff up to
    floating-point tolerance.
    """
    if learner == "PPO":

        increment = scalar_numeric(
            row.get(
                "entropy_increment",
                np.nan,
            )
        )

        if not np.isfinite(
            increment
        ):
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

        if not np.isfinite(
            increment
        ):
            return np.nan

        d_a = ACTION_DIMENSIONS[
            environment
        ]

        max_increment = (
            0.1
            * d_a
        )

        return float(
            increment
            / max_increment
        )

    raise ValueError(
        f"Unknown learner: {learner}"
    )


# ============================================================
# Event reconstruction
# ============================================================

def reconstruct_events(
    df,
    learner,
    environment,
    seed,
):
    """
    Reconstruct one operational event as:

        synthetic inherited anchor
            ->
        first subsequent genuine scheduler report
        from the same Trial.

    Every training report is NOT counted as an O2I event.
    """
    required_columns = {
        "Trial",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "Scheduler CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    df = (
        df.copy()
        .reset_index(
            drop=True
        )
    )

    df[
        "_row_order"
    ] = np.arange(
        len(
            df
        ),
        dtype=int,
    )

    df[
        "_synthetic"
    ] = (
        identify_synthetic_rows(
            df,
            learner,
        )
    )

    events = []
    audit = []

    anchor_indices = (
        df.index[
            df[
                "_synthetic"
            ]
        ]
        .tolist()
    )

    for anchor_idx in anchor_indices:

        trial = df.loc[
            anchor_idx,
            "Trial",
        ]

        # ----------------------------------------------------
        # First subsequent genuine report from same receiver
        # ----------------------------------------------------

        candidates = df.index[
            (df.index > anchor_idx)
            &
            (df[
                "Trial"
            ] == trial)
            &
            (
                ~df[
                    "_synthetic"
                ]
            )
        ]

        if len(
            candidates
        ) == 0:

            audit.append(
                {
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
                }
            )

            continue

        report_idx = int(
            candidates[
                0
            ]
        )

        # ----------------------------------------------------
        # Reject ambiguous repeated synthetic anchors
        # ----------------------------------------------------

        between = df[
            (df.index > anchor_idx)
            &
            (df.index < report_idx)
            &
            (df[
                "Trial"
            ] == trial)
            &
            df[
                "_synthetic"
            ]
        ]

        if not between.empty:

            audit.append(
                {
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
                }
            )

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
            not np.isfinite(
                guided
            )
            or guided < 0.5
        ):

            event_status = (
                "not_guided"
            )

        elif (
            not np.isfinite(
                raw
            )
            or raw <= EPS
        ):

            event_status = (
                "no_excess_uncertainty"
            )

        elif (
            not np.isfinite(
                thresholded
            )
            or thresholded <= EPS
        ):

            event_status = (
                "below_threshold"
            )

        elif (
            not np.isfinite(
                warmup
            )
            or warmup <= EPS
        ):

            event_status = (
                "warmup_blocked"
            )

        elif (
            np.isfinite(
                effective
            )
            and effective > EPS
        ):

            event_status = (
                "active"
            )

        else:

            event_status = (
                "other_zero_effective"
            )

        # ----------------------------------------------------
        # Primary normalized magnitude: U_eff
        # Secondary check: physical actuator displacement
        # ----------------------------------------------------

        normalized_from_u = (
            float(
                effective
            )
            if np.isfinite(
                effective
            )
            else np.nan
        )

        normalized_from_actuator = (
            physical_normalized_magnitude(
                row,
                learner,
                environment,
            )
        )

        consistency_error = (
            np.nan
        )

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
                - normalized_from_actuator
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
                        "Time",
                    ]
                    if "Time" in df.columns
                    else np.nan
                ),

            "report_row":
                report_idx,

            "report_time":
                scalar_numeric(
                    row.get(
                        "Time",
                        np.nan,
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

        audit.append(
            {
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
            }
        )

    return (
        pd.DataFrame(
            events
        ),
        pd.DataFrame(
            audit
        ),
        df,
    )


# ============================================================
# I2O operational summary
# ============================================================

def summarize_i2o(
    df,
):
    """
    Summarize I2O over genuine learner reports.

    Seed-level quantities:
      - valid diagnostic fraction,
      - median S,
      - Q1(S),
      - Q3(S),
      - IQR(S),
      - exact S=1 rate,
      - practical near-saturation S>=0.95 rate.
    """
    real = df[
        ~df[
            "_synthetic"
        ]
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
        np.isfinite(
            validity
        )
        &
        (
            validity
            >= 0.5
        )
        &
        np.isfinite(
            state
        )
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
        .astype(
            float
        )
    )

    s_median = float(
        s.median()
    )

    s_q1 = float(
        s.quantile(
            0.25
        )
    )

    s_q3 = float(
        s.quantile(
            0.75
        )
    )

    s_iqr = float(
        s_q3
        - s_q1
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
            s
            >= S_NEAR_SATURATION
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
# Seed-level event summary
# ============================================================

def summarize_events_for_seed(
    events,
):
    """
    Compute operational O2I summaries within one independent training seed.
    """
    empty_result = {
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

    if events.empty:
        return empty_result

    guided = events[
        pd.to_numeric(
            events[
                "guided_update"
            ],
            errors="coerce",
        )
        >= 0.5
    ].copy()

    n_guided = len(
        guided
    )

    def event_rate(
        status,
    ):
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
        guided[
            "u_raw"
        ],
        errors="coerce",
    )

    guided_thr = pd.to_numeric(
        guided[
            "u_thresholded"
        ],
        errors="coerce",
    )

    guided_warmup = pd.to_numeric(
        guided[
            "warmup_gain"
        ],
        errors="coerce",
    )

    return {
        "n_reconstructed_events":
            len(
                events
            ),

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
# Process one complete scheduler ZIP
# ============================================================

def process_zip(
    zip_path,
    learner,
):
    """
    Process all Full-CODA scheduler CSVs for one learner.
    """
    validate_zip_path(
        zip_path
    )

    all_events = []
    all_audit = []
    seed_rows = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        members = [
            name
            for name in zf.namelist()
            if (
                "scheduler_CODA_FULL_seed"
                in PurePosixPath(
                    name
                ).name
                and name.endswith(
                    ".csv"
                )
            )
        ]

        print(
            f"{learner}: "
            f"{len(members)} Full-CODA scheduler files"
        )

        if len(
            members
        ) == 0:

            raise RuntimeError(
                f"No Full-CODA scheduler files found in {zip_path}"
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
            ) as file:

                df = pd.read_csv(
                    file
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

            event_stats = summarize_events_for_seed(
                events
            )

            if not events.empty:
                all_events.append(
                    events
                )

            if not audit.empty:
                all_audit.append(
                    audit
                )

            seed_rows.append(
                {
                    "learner":
                        learner,

                    "environment":
                        environment,

                    "training_seed":
                        seed,

                    **event_stats,
                    **i2o_stats,
                }
            )

    events_all = (
        pd.concat(
            all_events,
            ignore_index=True,
        )
        if all_events
        else pd.DataFrame()
    )

    audit_all = (
        pd.concat(
            all_audit,
            ignore_index=True,
        )
        if all_audit
        else pd.DataFrame()
    )

    return (
        events_all,
        audit_all,
        pd.DataFrame(
            seed_rows
        ),
    )


# ============================================================
# Design validation
# ============================================================

def build_seed_counts(
    seed_summary,
):
    """
    Count independent training seeds per learner/environment.
    """
    seed_counts = (
        seed_summary
        .groupby(
            [
                "learner",
                "environment",
            ],
            as_index=False,
        )
        .agg(
            n_training_seeds=(
                "training_seed",
                "nunique",
            )
        )
    )

    return seed_counts


def validate_seed_counts(
    seed_counts,
):
    """
    Require ten Full-CODA seeds in every learner/environment condition.
    """
    expected = pd.DataFrame(
        [
            (
                learner,
                environment,
            )
            for learner in LEARNERS
            for environment in ENVIRONMENTS
        ],
        columns=[
            "learner",
            "environment",
        ],
    )

    checked = expected.merge(
        seed_counts,
        on=[
            "learner",
            "environment",
        ],
        how="left",
        validate="one_to_one",
    )

    checked[
        "n_training_seeds"
    ] = (
        checked[
            "n_training_seeds"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    bad = checked[
        checked[
            "n_training_seeds"
        ]
        != EXPECTED_SEEDS
    ]

    if not bad.empty:

        print(
            "\nIncomplete Full-CODA operational design:"
        )

        print(
            bad.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_SEEDS} training seeds "
            "for every learner/environment condition."
        )

    return checked


# ============================================================
# Across-seed aggregation
# ============================================================

AGGREGATE_METRICS = [
    # Event-level operational quantities.
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

    # I2O.
    "i2o_valid_rate",
    "i2o_state_median",
    "i2o_state_iqr",
    "i2o_near_saturation_rate",
    "i2o_exact_saturation_rate",

    # Implementation audit.
    "max_actuator_consistency_error",
]


def aggregate_across_seeds(
    seed_summary,
):
    """
    Aggregate seed-level operational quantities using median [Q1, Q3].
    """
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

    learner_order = {
        "PPO": 0,
        "SAC": 1,
    }

    environment_order = {
        environment: index
        for index, environment
        in enumerate(
            ENVIRONMENTS
        )
    }

    summary[
        "_learner_order"
    ] = (
        summary[
            "learner"
        ]
        .map(
            learner_order
        )
    )

    summary[
        "_environment_order"
    ] = (
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

    return summary


# ============================================================
# LaTeX helpers
# ============================================================

def median_iqr(
    row,
    metric,
    scale=1.0,
    digits=1,
):
    """
    Format one seed-aggregated operational metric as median [Q1, Q3].
    """
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
        np.isfinite(
            med
        )
        and
        np.isfinite(
            q1
        )
        and
        np.isfinite(
            q3
        )
    ):

        return "--"

    return (
        f"{med:.{digits}f} "
        f"[{q1:.{digits}f}, "
        f"{q3:.{digits}f}]"
    )


def build_main_table_body(
    summary,
):
    """
    Build the body of the main-paper operational Table 8.
    """
    lines = []

    previous_learner = None

    for _, row in summary.iterrows():

        learner = row[
            "learner"
        ]

        if (
            previous_learner is not None
            and learner
            != previous_learner
        ):

            lines.append(
                r"\midrule"
            )

            lines.append(
                ""
            )

        lines.append(
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

        previous_learner = (
            learner
        )

    return (
        "\n".join(
            lines
        )
        .rstrip()
    )


def build_supplement_table_body(
    summary,
):
    """
    Build the supplementary operational decomposition table body.
    """
    lines = []

    previous_learner = None

    for _, row in summary.iterrows():

        learner = row[
            "learner"
        ]

        if (
            previous_learner is not None
            and learner
            != previous_learner
        ):

            lines.append(
                r"\midrule"
            )

            lines.append(
                ""
            )

        lines.append(
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

        previous_learner = (
            learner
        )

    return (
        "\n".join(
            lines
        )
        .rstrip()
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Process PPO and SAC Full-CODA scheduler logs
    # --------------------------------------------------------

    print(
        "Processing PPO Full-CODA scheduler logs..."
    )

    (
        ppo_events,
        ppo_audit,
        ppo_seed,
    ) = process_zip(
        PPO_ZIP,
        learner="PPO",
    )

    print(
        "\nProcessing SAC Full-CODA scheduler logs..."
    )

    (
        sac_events,
        sac_audit,
        sac_seed,
    ) = process_zip(
        SAC_ZIP,
        learner="SAC",
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

    # --------------------------------------------------------
    # Validate independent training-seed design
    # --------------------------------------------------------

    seed_counts = (
        build_seed_counts(
            seed_summary
        )
    )

    seed_counts = (
        validate_seed_counts(
            seed_counts
        )
    )

    print(
        "\nTraining seeds per learner/environment:"
    )

    print(
        seed_counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Event reconstruction audit
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )

    print(
        "EVENT RECONSTRUCTION AUDIT"
    )

    print(
        "=" * 88
    )

    audit_counts = (
        audit[
            "status"
        ]
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
        ]
        > 0
    ):

        raise RuntimeError(
            "Ambiguous event reconstruction detected."
        )

    # --------------------------------------------------------
    # Operational event-state counts
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )

    print(
        "EVENT STATUS COUNTS"
    )

    print(
        "=" * 88
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

    # --------------------------------------------------------
    # Actuator mapping consistency
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Aggregate across independent training seeds
    # --------------------------------------------------------

    summary = (
        aggregate_across_seeds(
            seed_summary
        )
    )

    # --------------------------------------------------------
    # Save auditable numerical artifacts
    # --------------------------------------------------------

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

    seed_counts.to_csv(
        OUTPUT_SEED_COUNTS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # Generate LaTeX table bodies
    # --------------------------------------------------------

    main_latex = (
        build_main_table_body(
            summary
        )
    )

    supp_latex = (
        build_supplement_table_body(
            summary
        )
    )

    OUTPUT_MAIN_LATEX.write_text(
        main_latex
        + "\n",
        encoding="utf-8",
    )

    OUTPUT_SUPP_LATEX.write_text(
        supp_latex
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

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
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "MAIN-PAPER LATEX TABLE BODY"
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
        "SUPPLEMENTARY LATEX TABLE BODY"
    )

    print(
        "=" * 120
        + "\n"
    )

    print(
        supp_latex
    )

    # --------------------------------------------------------
    # Report generated files
    # --------------------------------------------------------

    print(
        "\nGenerated outputs:"
    )

    print(
        f"  Event-level data        : {OUTPUT_EVENTS}"
    )

    print(
        f"  Reconstruction audit    : {OUTPUT_AUDIT}"
    )

    print(
        f"  Seed-level summary      : {OUTPUT_SEEDS}"
    )

    print(
        f"  Seed counts             : {OUTPUT_SEED_COUNTS}"
    )

    print(
        f"  Across-seed summary     : {OUTPUT_SUMMARY}"
    )

    print(
        f"  Main table body         : {OUTPUT_MAIN_LATEX}"
    )

    print(
        f"  Supplement table body   : {OUTPUT_SUPP_LATEX}"
    )

    print(
        "\nDone."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()