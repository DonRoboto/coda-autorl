



from __future__ import annotations

"""
Select one training-side champion per (method, environment, training seed)
and evaluate its FINAL checkpoint on held-out MuJoCo episodes.

Selection is performed using TRAINING DATA ONLY. Test returns are never used
for checkpoint selection.

In addition to held-out return, this version records locomotion/control-quality
metrics for Hopper-v5, HalfCheetah-v5, Walker2d-v5, Swimmer-v5, Ant-v5, and
Humanoid-v5:

- root-body orientation deviation from the episode-reset orientation;
- root-body angular-velocity magnitude;
- normalized action smoothness;
- normalized action effort;
- full-horizon survival/completion and termination indicators;
- optional forward displacement and mean forward speed when x_position is
  exposed by the environment.

No composite stability score is constructed. The metrics are stored separately
so return remains the primary outcome and physical/control stability can be
analyzed without arbitrary weights.

This final-campaign version adds four safeguards that are essential for the
PBT/PB2/ASHA/CODA comparison:

1. A preflight audit rejects incomplete population runs before any test starts.
2. Champion scoring uses the final causal branch AND the final executed
   configuration segment.
3. A candidate needs at least 100k interactions after its last configuration
   change, in addition to at least three finite terminal observations.
4. Episode results are saved per method/environment/training-seed so the test
   campaign can be resumed safely.

IMPORTANT:
- Run this file from the project root so custom callback/module classes used by
  RLlib checkpoints are importable.
- Confirm RUN_NAMES exactly match the final campaign folders.
- PBT, PB2, ASHA, and CODA must have saved terminal checkpoints with
  CheckpointConfig(checkpoint_at_end=True).
- Set ASHA_EXPECTED_NUM_TRIALS to the single value frozen for the final ASHA
  campaign. Leave it as None only if exact ASHA trial-count validation is not
  possible from the archived outputs.
"""

# -----------------------------------------------------------------------------
# CPU limits must be set before NumPy / Ray / PyTorch imports.
# -----------------------------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.algorithm import Algorithm


# =============================================================================
# REPOSITORY / EVALUATION CONFIGURATION
# =============================================================================

# Intended location:
#     <repo>/evaluation/heldout_test_ppo.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from configs.environments import ENVIRONMENTS  # noqa: E402
from configs.seeds import (  # noqa: E402
    TRAINING_SEEDS,
    HELDOUT_RETURN_SEEDS,
)
from configs.ppo_config import (  # noqa: E402
    POPULATION_SIZE,
    MAX_TIMESTEPS_PER_WORKER,
    ASHA_NUM_SAMPLES,
    TERMINAL_WINDOW_STEPS,
    MIN_TERMINAL_SUPPORT_STEPS,
    MIN_POST_CONFIG_SUPPORT_STEPS,
    MIN_TERMINAL_POINTS,
)

RUN_NAMES: Dict[str, str] = {
    "PBT": "PBT_PB2SPACE",
    "PB2": "PB2_PB2SPACE",
    "ASHA": "ASHA_PB2SPACE",
    "CODA": "CODA_KL_DUQ_DONORREL_PB2SPACE",
    "CODA-I2O": "CODA_I2O_KL_DUQ_DONORREL_PB2SPACE",
    "CODA-O2I": "CODA_O2I_KL_DUQ_DONORREL_PB2SPACE",
}

EXPECTED_POPULATION_SIZE: Dict[str, int] = {
    "PBT": POPULATION_SIZE,
    "PB2": POPULATION_SIZE,
    "CODA": POPULATION_SIZE,
    "CODA-I2O": POPULATION_SIZE,
    "CODA-O2I": POPULATION_SIZE,
}

ASHA_EXPECTED_NUM_TRIALS: Optional[int] = ASHA_NUM_SAMPLES

TARGET_TRAINING_STEPS = MAX_TIMESTEPS_PER_WORKER

CONFIG_COLUMNS: Tuple[str, ...] = (
    "config/train_batch_size",
    "config/lambda",
    "config/clip_param",
    "config/lr",
    "config/entropy_coeff",
)

ASHA_REQUIRE_FULL_BUDGET = True
ALLOW_NONCANONICAL_METRICS = False

# Held-out protocol.
N_TEST_EPISODES = len(HELDOUT_RETURN_SEEDS)
TEST_EPISODE_SEEDS = list(HELDOUT_RETURN_SEEDS)
EXPLORE = False

# Safe smoke mode by default.
SMOKE_TEST = os.environ.get("CODA_PPO_TEST_SMOKE", "1") != "0"
SMOKE_ENV = os.environ.get("CODA_PPO_TEST_ENV", "Hopper-v5")
SMOKE_TRAINING_SEED = int(
    os.environ.get("CODA_PPO_TEST_TRAINING_SEED", str(TRAINING_SEEDS[0]))
)
SMOKE_METHODS = ["CODA"]
SMOKE_TEST_EPISODES = int(
    os.environ.get("CODA_PPO_TEST_EPISODES", "10")
)

TEST_OUTPUT_ROOT = RESULTS_ROOT / (
    "heldout_stability_ppo_smoke"
    if SMOKE_TEST
    else "heldout_stability_ppo_final"
)
CASE_OUTPUT_ROOT = TEST_OUTPUT_ROOT / "episodes_by_case"

RESUME_COMPLETED_CASES = True
REQUIRE_PREFLIGHT_SUCCESS = True
FAIL_FAST_DURING_EVALUATION = False


# =============================================================================
# CONSTANTS
# =============================================================================

METRIC_COL = "env_runners/episode_return_mean"
TIMESTEP_COL = "timesteps_total"
AGENT_COL = "agente_id"

logger = logging.getLogger("champion_test")


STABILITY_METRIC_VERSION = "root_motion_action_v1"


@dataclass(frozen=True)
class StabilitySpec:
    """Observation coordinates used for environment-agnostic root stability.

    orientation_kind:
        ``planar`` uses one angular coordinate and wrapped angular distance.
        ``quaternion`` uses a wxyz quaternion and geodesic angular distance.

    All indices assume the default Gymnasium-v5 observation construction used
    by the archived checkpoints (current x/y positions excluded where the
    environment excludes them by default).
    """

    orientation_kind: str
    orientation_indices: Tuple[int, ...]
    angular_velocity_indices: Tuple[int, ...]
    root_label: str
    minimum_observation_size: int


STABILITY_SPECS: Dict[str, StabilitySpec] = {
    "Hopper-v5": StabilitySpec(
        orientation_kind="planar",
        orientation_indices=(1,),
        angular_velocity_indices=(7,),
        root_label="torso",
        minimum_observation_size=11,
    ),
    "HalfCheetah-v5": StabilitySpec(
        orientation_kind="planar",
        orientation_indices=(1,),
        angular_velocity_indices=(10,),
        root_label="front_tip",
        minimum_observation_size=17,
    ),
    "Walker2d-v5": StabilitySpec(
        orientation_kind="planar",
        orientation_indices=(1,),
        angular_velocity_indices=(10,),
        root_label="torso",
        minimum_observation_size=17,
    ),
    "Swimmer-v5": StabilitySpec(
        orientation_kind="planar",
        orientation_indices=(0,),
        angular_velocity_indices=(5,),
        root_label="front_tip",
        minimum_observation_size=8,
    ),
    "Ant-v5": StabilitySpec(
        orientation_kind="quaternion",
        orientation_indices=(1, 2, 3, 4),
        angular_velocity_indices=(16, 17, 18),
        root_label="torso",
        minimum_observation_size=19,
    ),
    "Humanoid-v5": StabilitySpec(
        orientation_kind="quaternion",
        orientation_indices=(1, 2, 3, 4),
        angular_velocity_indices=(25, 26, 27),
        root_label="torso",
        minimum_observation_size=28,
    ),
}


PRIMARY_STABILITY_COLUMNS: Tuple[str, ...] = (
    "root_orientation_rms_rad",
    "root_angular_velocity_rms_rad_s",
    "normalized_action_smoothness_mse",
    "normalized_action_effort_mse",
    "episode_completion_fraction",
    "survived_full_horizon",
    "ended_by_termination",
)


@dataclass(frozen=True)
class EpisodeEvaluation:
    episode_return: float
    episode_length: int
    root_orientation_rms_rad: float
    root_orientation_mean_abs_rad: float
    root_orientation_p95_abs_rad: float
    root_angular_velocity_rms_rad_s: float
    root_angular_velocity_mean_abs_rad_s: float
    root_angular_velocity_p95_abs_rad_s: float
    normalized_action_smoothness_mse: float
    normalized_action_effort_mse: float
    action_bound_violation_fraction: float
    episode_completion_fraction: float
    survived_full_horizon: int
    ended_by_termination: int
    ended_by_truncation: int
    forward_displacement: float
    mean_forward_speed: float
    environment_dt: float
    max_episode_steps: int


@dataclass(frozen=True)
class TerminalScore:
    score: float
    window_start: float
    window_end: float
    n_points: int
    last_training_return: float
    execution_segment_start: float
    post_config_support: float
    terminal_progress: float


# =============================================================================
# PATH / DATA HELPERS
# =============================================================================


def _ensure_project_on_path() -> None:
    """Make repository and custom scheduler modules importable on restore."""
    for candidate in (REPO_ROOT, SRC_ROOT):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


def _metrics_path(env_name: str, run_name: str, seed: int) -> Path:
    """Return the exact final metrics path, rejecting silent ambiguity."""
    expected = (
        RESULTS_ROOT
        / "metrics"
        / env_name
        / f"metrics_{run_name}_seed{seed}.csv"
    )
    if expected.exists():
        return expected

    if not ALLOW_NONCANONICAL_METRICS:
        raise FileNotFoundError(
            f"Canonical metrics file not found: {expected}. "
            "Rename the final file to the canonical name or explicitly enable "
            "ALLOW_NONCANONICAL_METRICS after removing old copies."
        )

    parent = expected.parent
    matches = sorted(parent.glob(f"metrics_{run_name}_seed{seed}*.csv"))

    if len(matches) == 1:
        logger.warning("Using noncanonical metrics file: %s", matches[0])
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous metrics files for {run_name}, {env_name}, seed={seed}:\n"
            + "\n".join(f"  - {p}" for p in matches)
        )
    raise FileNotFoundError(f"Metrics file not found: {expected}")


def _checkpoint_dir(env_name: str, run_name: str, seed: int, agent_id: str) -> Path:
    path = (
        RESULTS_ROOT
        / "champions"
        / env_name
        / f"{run_name}_seed{seed}"
        / agent_id
    )
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    if not any(path.iterdir()):
        raise RuntimeError(f"Checkpoint directory is empty: {path}")
    return path


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _chronological_worker_history(df: pd.DataFrame) -> pd.DataFrame:
    """Recover causal execution order within one worker/trial."""
    out = df.copy()
    out["__row_order"] = np.arange(len(out), dtype=np.int64)

    if "causal_order" in out.columns:
        order = _safe_numeric(out["causal_order"])
        if order.notna().all() and not order.duplicated().any():
            out["__order"] = order
            return (
                out.sort_values(["__order", "__row_order"], kind="stable")
                .drop(columns=["__order"])
                .reset_index(drop=True)
            )

    # Backward-compatible fallbacks. Final campaign files should use
    # causal_order; these paths are retained only for older baseline files.
    if "time_total_s" in out.columns:
        order = _safe_numeric(out["time_total_s"])
        if order.notna().any():
            out["__order"] = order
            return (
                out.sort_values(["__order", "__row_order"], kind="stable")
                .drop(columns=["__order"])
                .reset_index(drop=True)
            )

    if "training_iteration" in out.columns:
        order = _safe_numeric(out["training_iteration"])
        if order.notna().any():
            out["__order"] = order
            return (
                out.sort_values(["__order", "__row_order"], kind="stable")
                .drop(columns=["__order"])
                .reset_index(drop=True)
            )

    return out.sort_values("__row_order", kind="stable").reset_index(drop=True)


def _last_finite_value(series: pd.Series) -> float:
    values = _safe_numeric(series)
    values = values[np.isfinite(values)]
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _worker_terminal_progress(worker_df: pd.DataFrame) -> float:
    hist = _chronological_worker_history(worker_df)
    if TIMESTEP_COL not in hist.columns:
        return float("nan")
    return _last_finite_value(hist[TIMESTEP_COL])


def _final_causal_branch(worker_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the final branch after the latest observed timestep rollback."""
    hist = _chronological_worker_history(worker_df)
    if TIMESTEP_COL not in hist.columns:
        return hist

    timesteps = _safe_numeric(hist[TIMESTEP_COL])
    reset_positions = np.flatnonzero(
        (timesteps.diff() < 0).fillna(False).to_numpy()
    )
    if len(reset_positions) == 0:
        return hist.reset_index(drop=True)

    start = int(reset_positions[-1])
    return hist.iloc[start:].reset_index(drop=True)


def _values_differ(previous: float, current: float, column: str) -> bool:
    prev_finite = np.isfinite(previous)
    curr_finite = np.isfinite(current)

    if not prev_finite and not curr_finite:
        return False
    if prev_finite != curr_finite:
        return True

    if column == "config/train_batch_size":
        return int(round(previous)) != int(round(current))

    return not bool(np.isclose(previous, current, rtol=1e-10, atol=1e-12))


def _last_config_change_position(branch: pd.DataFrame) -> int:
    """Return the row where the final executed configuration segment begins."""
    present = [column for column in CONFIG_COLUMNS if column in branch.columns]
    if not present or len(branch) <= 1:
        return 0

    numeric = {
        column: _safe_numeric(branch[column]).to_numpy(dtype=float)
        for column in present
    }

    last_change = 0
    for row in range(1, len(branch)):
        changed = any(
            _values_differ(values[row - 1], values[row], column)
            for column, values in numeric.items()
        )
        if changed:
            last_change = row

    return last_change


def _final_execution_segment(worker_df: pd.DataFrame) -> pd.DataFrame:
    """Return the final causal branch restricted to the final configuration."""
    branch = _final_causal_branch(worker_df)
    if branch.empty:
        return branch
    start = _last_config_change_position(branch)
    return branch.iloc[start:].reset_index(drop=True)


def _terminal_time_weighted_score(
    worker_df: pd.DataFrame,
    window_steps: int = TERMINAL_WINDOW_STEPS,
) -> TerminalScore:
    """Time-weighted return over the final window of the final causal branch.

    The score itself follows the pre-specified final-branch criterion.  The
    final configuration segment is tracked separately and is used as an
    eligibility safeguard: a checkpoint must have at least
    MIN_POST_CONFIG_SUPPORT_STEPS of observed training after the latest change
    in any executed hyperparameter/actuator coordinate.
    """
    terminal_progress = _worker_terminal_progress(worker_df)
    branch = _final_causal_branch(worker_df)

    empty = TerminalScore(
        score=np.nan,
        window_start=np.nan,
        window_end=np.nan,
        n_points=0,
        last_training_return=np.nan,
        execution_segment_start=np.nan,
        post_config_support=np.nan,
        terminal_progress=terminal_progress,
    )

    if branch.empty or METRIC_COL not in branch or TIMESTEP_COL not in branch:
        return empty

    branch_timesteps = _safe_numeric(branch[TIMESTEP_COL])
    finite_branch_t = branch_timesteps[np.isfinite(branch_timesteps)]
    if finite_branch_t.empty:
        return empty

    last_change_position = _last_config_change_position(branch)
    config_start_value = _safe_numeric(
        branch.iloc[last_change_position : last_change_position + 1][TIMESTEP_COL]
    )
    finite_config_start = config_start_value[np.isfinite(config_start_value)]
    if finite_config_start.empty:
        return empty
    execution_segment_start = float(finite_config_start.iloc[0])

    returns = _safe_numeric(branch[METRIC_COL])
    valid = (
        branch_timesteps.notna()
        & returns.notna()
        & np.isfinite(branch_timesteps)
        & np.isfinite(returns)
    )
    xy = pd.DataFrame({"T": branch_timesteps[valid], "R": returns[valid]})
    if xy.empty:
        return empty

    # Keep the latest causally observed return at duplicated training counters.
    xy = xy.reset_index(drop=True).groupby("T", as_index=False, sort=True).last()

    t_end = float(xy["T"].iloc[-1])
    t_first_finite = float(xy["T"].iloc[0])
    t_start = max(t_first_finite, t_end - float(window_steps))
    last_return = float(xy["R"].iloc[-1])
    post_config_support = float(t_end - execution_segment_start)

    if len(xy) == 1 or not (t_end > t_start):
        return TerminalScore(
            score=last_return,
            window_start=t_start,
            window_end=t_end,
            n_points=1,
            last_training_return=last_return,
            execution_segment_start=execution_segment_start,
            post_config_support=post_config_support,
            terminal_progress=terminal_progress,
        )

    T = xy["T"].to_numpy(dtype=float)
    R = xy["R"].to_numpy(dtype=float)
    r_start = float(np.interp(t_start, T, R))

    inside = (T > t_start) & (T <= t_end)
    T_window = np.concatenate([[t_start], T[inside]])
    R_window = np.concatenate([[r_start], R[inside]])

    if len(T_window) < 2 or T_window[-1] <= T_window[0]:
        score = last_return
    else:
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(R_window, T_window))
        else:
            auc = float(np.trapz(R_window, T_window))
        score = auc / float(T_window[-1] - T_window[0])

    return TerminalScore(
        score=score,
        window_start=t_start,
        window_end=t_end,
        n_points=int(len(T_window)),
        last_training_return=last_return,
        execution_segment_start=execution_segment_start,
        post_config_support=post_config_support,
        terminal_progress=terminal_progress,
    )

def _asha_is_full_budget(worker_df: pd.DataFrame) -> bool:
    if "asha_reached_max_resource" in worker_df.columns:
        flag = _safe_numeric(worker_df["asha_reached_max_resource"])
        if flag.notna().any():
            return bool(flag.max() >= 1)

    terminal = _worker_terminal_progress(worker_df)
    return bool(np.isfinite(terminal) and terminal >= TARGET_TRAINING_STEPS)


def _validate_causal_order(agent_id: str, worker_df: pd.DataFrame) -> List[str]:
    problems: List[str] = []
    if "causal_order" not in worker_df.columns:
        problems.append(f"{agent_id}: causal_order column is missing")
        return problems

    order = _safe_numeric(worker_df["causal_order"])
    if order.isna().any():
        problems.append(f"{agent_id}: causal_order contains non-numeric values")
    if order.duplicated().any():
        problems.append(f"{agent_id}: causal_order contains duplicates")
    return problems


def _validate_training_run(
    method: str,
    env_name: str,
    seed: int,
    run_name: str,
    df: pd.DataFrame,
) -> None:
    """Reject incomplete training runs before champion selection or testing."""
    problems: List[str] = []

    required = {AGENT_COL, METRIC_COL, TIMESTEP_COL}
    missing = required - set(df.columns)
    if missing:
        problems.append(f"missing required columns: {sorted(missing)}")

    if problems:
        raise RuntimeError(
            f"Invalid training run {method}/{env_name}/seed={seed}: "
            + "; ".join(problems)
        )

    groups = list(df.groupby(AGENT_COL, sort=False))
    agent_ids = [str(agent_id) for agent_id, _ in groups]

    expected = EXPECTED_POPULATION_SIZE.get(method)
    if method == "ASHA" and ASHA_EXPECTED_NUM_TRIALS is not None:
        expected = ASHA_EXPECTED_NUM_TRIALS

    if expected is not None and len(groups) != expected:
        problems.append(
            f"expected {expected} workers/trials, found {len(groups)}: {agent_ids}"
        )

    full_budget_asha = 0
    for agent_id, worker_df in groups:
        agent_id = str(agent_id)
        problems.extend(_validate_causal_order(agent_id, worker_df))
        terminal = _worker_terminal_progress(worker_df)

        if method in EXPECTED_POPULATION_SIZE:
            if not np.isfinite(terminal) or terminal < TARGET_TRAINING_STEPS:
                problems.append(
                    f"{agent_id}: terminal progress={terminal}, expected at least "
                    f"{TARGET_TRAINING_STEPS}"
                )
            try:
                _checkpoint_dir(env_name, run_name, seed, agent_id)
            except Exception as exc:
                problems.append(f"{agent_id}: {exc}")

        elif method == "ASHA" and _asha_is_full_budget(worker_df):
            full_budget_asha += 1
            try:
                _checkpoint_dir(env_name, run_name, seed, agent_id)
            except Exception as exc:
                problems.append(f"{agent_id}: full-budget checkpoint invalid: {exc}")

    if method == "ASHA" and ASHA_REQUIRE_FULL_BUDGET and full_budget_asha < 1:
        problems.append("ASHA has no full-budget trial with an available checkpoint")

    if problems:
        raise RuntimeError(
            f"Invalid training run {method}/{env_name}/seed={seed}:\n  - "
            + "\n  - ".join(problems)
        )


def select_training_champion(
    method: str,
    env_name: str,
    training_seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Select one champion using training information only."""
    run_name = RUN_NAMES[method]
    path = _metrics_path(env_name, run_name, training_seed)
    df = pd.read_csv(path)

    _validate_training_run(method, env_name, training_seed, run_name, df)

    rows: List[dict] = []

    for agent_id, worker_df in df.groupby(AGENT_COL, sort=False):
        agent_id = str(agent_id)
        eligible = True
        reasons: List[str] = []

        if method == "ASHA" and ASHA_REQUIRE_FULL_BUDGET:
            if not _asha_is_full_budget(worker_df):
                eligible = False
                reasons.append("ASHA trial did not reach full budget")

        try:
            checkpoint = _checkpoint_dir(
                env_name, run_name, training_seed, agent_id
            )
            checkpoint_exists = True
        except Exception as exc:
            checkpoint = Path("")
            checkpoint_exists = False
            eligible = False
            reasons.append(str(exc))

        terminal = _terminal_time_weighted_score(worker_df)
        window_support = (
            terminal.window_end - terminal.window_start
            if np.isfinite(terminal.window_start)
            and np.isfinite(terminal.window_end)
            else np.nan
        )

        if not np.isfinite(terminal.score):
            eligible = False
            reasons.append("terminal training score unavailable")
        if not np.isfinite(window_support) or window_support < MIN_TERMINAL_SUPPORT_STEPS:
            eligible = False
            reasons.append("insufficient terminal scoring-window support")
        if terminal.n_points < MIN_TERMINAL_POINTS:
            eligible = False
            reasons.append("insufficient terminal observations")
        if (
            not np.isfinite(terminal.post_config_support)
            or terminal.post_config_support < MIN_POST_CONFIG_SUPPORT_STEPS
        ):
            eligible = False
            reasons.append("insufficient support after last configuration change")

        rows.append(
            {
                "method": method,
                "run_name": run_name,
                "environment": env_name,
                "training_seed": training_seed,
                "agent_id": agent_id,
                "eligible": int(eligible),
                "eligibility_reason": "eligible" if eligible else "; ".join(reasons),
                "training_terminal_score": terminal.score,
                "terminal_window_start": terminal.window_start,
                "terminal_window_end": terminal.window_end,
                "terminal_window_support": window_support,
                "terminal_points": terminal.n_points,
                "last_training_return": terminal.last_training_return,
                "execution_segment_start": terminal.execution_segment_start,
                "post_config_support_steps": terminal.post_config_support,
                "terminal_progress": terminal.terminal_progress,
                "checkpoint_exists": int(checkpoint_exists),
                "checkpoint_path": str(checkpoint) if checkpoint_exists else "",
                "metrics_path": str(path),
            }
        )

    candidates = pd.DataFrame(rows)
    eligible_df = candidates[candidates["eligible"] == 1].copy()

    if eligible_df.empty:
        raise RuntimeError(
            f"No eligible champion candidate for {method}, {env_name}, "
            f"training_seed={training_seed}."
        )

    eligible_df = eligible_df.sort_values(
        ["training_terminal_score", "last_training_return", "agent_id"],
        ascending=[False, False, True],
        kind="stable",
    )

    winner_index = eligible_df.index[0]
    candidates["selected_champion"] = 0
    candidates.loc[winner_index, "selected_champion"] = 1
    winner = candidates.loc[winner_index].copy()
    return candidates, winner


# =============================================================================
# CHECKPOINT EVALUATION
# =============================================================================


def _restore_algorithm(checkpoint_path: Path) -> Algorithm:
    logger.info("Restoring checkpoint: %s", checkpoint_path)
    return Algorithm.from_checkpoint(str(checkpoint_path))


def _validate_algorithm_environment(
    algo: Algorithm,
    env: gym.Env,
    env_name: str,
) -> None:
    configured_env = getattr(algo.config, "env", None)
    if isinstance(configured_env, str) and configured_env != env_name:
        raise RuntimeError(
            f"Checkpoint environment mismatch: checkpoint={configured_env}, "
            f"requested={env_name}"
        )

    if env_name not in STABILITY_SPECS:
        raise KeyError(
            f"No stability specification is defined for {env_name!r}. "
            f"Supported environments: {sorted(STABILITY_SPECS)}"
        )

    policy = algo.get_policy()
    if policy is None:
        raise RuntimeError("Restored Algorithm has no default policy")

    policy_action_shape = getattr(policy.action_space, "shape", None)
    env_action_shape = getattr(env.action_space, "shape", None)
    if policy_action_shape != env_action_shape:
        raise RuntimeError(
            f"Action-space mismatch: checkpoint={policy_action_shape}, "
            f"environment={env_action_shape}"
        )

    observation_filter = getattr(algo.config, "observation_filter", None)
    if observation_filter not in (None, "MeanStdFilter"):
        raise RuntimeError(
            f"Unexpected restored observation filter: {observation_filter!r}"
        )

    spec = STABILITY_SPECS[env_name]
    observation_shape = getattr(env.observation_space, "shape", None)
    if (
        observation_shape is None
        or len(observation_shape) != 1
        or int(observation_shape[0]) < spec.minimum_observation_size
    ):
        raise RuntimeError(
            f"Unexpected observation shape for {env_name}: {observation_shape}; "
            f"need at least ({spec.minimum_observation_size},) for the "
            "stability-coordinate mapping."
        )


def _flat_finite_observation(obs, *, context: str) -> np.ndarray:
    array = np.asarray(obs, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise FloatingPointError(f"Non-finite or empty observation: {context}")
    return array


def _wrap_to_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _normalized_quaternion(values: np.ndarray, *, context: str) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise FloatingPointError(f"Invalid quaternion in {context}: {quaternion}")
    return quaternion / norm


def _orientation_reference(obs: np.ndarray, spec: StabilitySpec) -> np.ndarray:
    values = obs[list(spec.orientation_indices)]
    if spec.orientation_kind == "planar":
        return np.asarray([float(values[0])], dtype=np.float64)
    if spec.orientation_kind == "quaternion":
        return _normalized_quaternion(values, context="episode reset")
    raise ValueError(f"Unknown orientation kind: {spec.orientation_kind!r}")


def _root_orientation_deviation(
    obs: np.ndarray,
    reference: np.ndarray,
    spec: StabilitySpec,
) -> float:
    values = obs[list(spec.orientation_indices)]
    if spec.orientation_kind == "planar":
        return abs(_wrap_to_pi(float(values[0]) - float(reference[0])))

    current = _normalized_quaternion(values, context="episode step")
    # q and -q represent the same orientation.  abs(dot) makes the geodesic
    # distance invariant to this sign ambiguity.
    dot = float(np.clip(abs(np.dot(current, reference)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _root_angular_velocity_magnitude(
    obs: np.ndarray,
    spec: StabilitySpec,
) -> float:
    values = obs[list(spec.angular_velocity_indices)]
    magnitude = float(np.linalg.norm(values))
    if not np.isfinite(magnitude):
        raise FloatingPointError("Non-finite root angular velocity")
    return magnitude


def _normalize_action(action: np.ndarray, action_space: gym.spaces.Box) -> np.ndarray:
    action_flat = np.asarray(action, dtype=np.float64).reshape(-1)
    low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)

    if action_flat.size != low.size:
        raise RuntimeError(
            f"Action size mismatch: action={action_flat.size}, space={low.size}"
        )

    midpoint = 0.5 * (high + low)
    half_range = 0.5 * (high - low)
    if (
        not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(half_range <= 0.0)
    ):
        raise RuntimeError("Stability analysis requires finite Box action bounds")

    return (action_flat - midpoint) / half_range


def _optional_info_float(info: object, key: str) -> float:
    if not isinstance(info, dict) or key not in info:
        return float("nan")
    try:
        value = float(info[key])
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _rms(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array))))


def _one_test_episode(
    algo: Algorithm,
    env: gym.Env,
    env_name: str,
    episode_seed: int,
) -> EpisodeEvaluation:
    spec = STABILITY_SPECS[env_name]
    obs, reset_info = env.reset(seed=int(episode_seed))
    obs_array = _flat_finite_observation(
        obs, context=f"reset, seed={episode_seed}"
    )
    orientation_ref = _orientation_reference(obs_array, spec)

    try:
        env.action_space.seed(int(episode_seed))
    except Exception:
        pass

    terminated = False
    truncated = False
    total_reward = 0.0
    episode_length = 0

    orientation_deviations: List[float] = []
    angular_velocity_magnitudes: List[float] = []
    action_efforts: List[float] = []
    action_changes: List[float] = []
    previous_normalized_action: Optional[np.ndarray] = None
    bound_violation_components = 0
    total_action_components = 0

    initial_x = _optional_info_float(reset_info, "x_position")
    final_x = initial_x
    final_info: dict = dict(reset_info) if isinstance(reset_info, dict) else {}

    while not (terminated or truncated):
        # On RLlib's old API stack, Algorithm.compute_actions() applies the
        # restored preprocessor and MeanStdFilter with update=False. Do not
        # replace this call with compute_single_action() for this campaign.
        action_dict = algo.compute_actions(
            {"eval_agent": obs},
            explore=EXPLORE,
        )
        raw_action = np.asarray(action_dict["eval_agent"], dtype=np.float64)
        action = raw_action.reshape(env.action_space.shape)

        if not np.isfinite(action).all():
            raise FloatingPointError(
                f"Non-finite action in held-out episode seed={episode_seed}, "
                f"step={episode_length}"
            )

        normalized_action = _normalize_action(action, env.action_space)
        action_efforts.append(float(np.mean(np.square(normalized_action))))
        if previous_normalized_action is not None:
            action_changes.append(
                float(
                    np.mean(
                        np.square(normalized_action - previous_normalized_action)
                    )
                )
            )
        previous_normalized_action = normalized_action.copy()

        low = np.asarray(env.action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float64).reshape(-1)
        action_flat = action.reshape(-1)
        violations = (action_flat < low - 1e-8) | (action_flat > high + 1e-8)
        bound_violation_components += int(np.count_nonzero(violations))
        total_action_components += int(action_flat.size)

        obs, reward, terminated, truncated, step_info = env.step(action)
        final_info = step_info if isinstance(step_info, dict) else {}

        if not np.isfinite(float(reward)):
            raise FloatingPointError(
                f"Non-finite reward in held-out episode seed={episode_seed}, "
                f"step={episode_length}"
            )

        obs_raw = np.asarray(obs, dtype=np.float64).reshape(-1)
        observation_is_finite = bool(obs_raw.size > 0 and np.isfinite(obs_raw).all())
        if not observation_is_finite and not (terminated or truncated):
            raise FloatingPointError(
                f"Non-finite nonterminal observation in test seed={episode_seed}, "
                f"step={episode_length}"
            )

        # A terminal state may be non-finite in a numerically failed environment.
        # Such a policy should not silently receive a physical-stability score.
        if not observation_is_finite:
            raise FloatingPointError(
                f"Non-finite terminal observation in test seed={episode_seed}, "
                f"step={episode_length}"
            )

        orientation_deviations.append(
            _root_orientation_deviation(obs_raw, orientation_ref, spec)
        )
        angular_velocity_magnitudes.append(
            _root_angular_velocity_magnitude(obs_raw, spec)
        )

        x_value = _optional_info_float(final_info, "x_position")
        if np.isfinite(x_value):
            final_x = x_value

        total_reward += float(reward)
        episode_length += 1

    if not np.isfinite(total_reward):
        raise FloatingPointError(
            f"Non-finite episode return for test seed {episode_seed}"
        )

    orientation = np.asarray(orientation_deviations, dtype=np.float64)
    angular_velocity = np.asarray(angular_velocity_magnitudes, dtype=np.float64)
    if (
        orientation.size != episode_length
        or angular_velocity.size != episode_length
        or not np.isfinite(orientation).all()
        or not np.isfinite(angular_velocity).all()
    ):
        raise FloatingPointError("Incomplete or non-finite stability trajectory")

    max_episode_steps_value = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    max_episode_steps = (
        int(max_episode_steps_value)
        if max_episode_steps_value is not None
        else int(episode_length)
    )
    completion_fraction = (
        min(float(episode_length) / float(max_episode_steps), 1.0)
        if max_episode_steps > 0
        else float("nan")
    )
    survived_full_horizon = int(
        max_episode_steps > 0
        and episode_length >= max_episode_steps
        and not bool(terminated)
    )

    dt_value = getattr(getattr(env, "unwrapped", env), "dt", np.nan)
    try:
        environment_dt = float(dt_value)
    except (TypeError, ValueError):
        environment_dt = float("nan")
    if not np.isfinite(environment_dt) or environment_dt <= 0.0:
        environment_dt = float("nan")

    forward_displacement = (
        float(final_x - initial_x)
        if np.isfinite(initial_x) and np.isfinite(final_x)
        else float("nan")
    )
    mean_forward_speed = (
        forward_displacement / (float(episode_length) * environment_dt)
        if np.isfinite(forward_displacement)
        and np.isfinite(environment_dt)
        and episode_length > 0
        else float("nan")
    )

    smoothness = (
        float(np.mean(action_changes)) if action_changes else 0.0
    )
    effort = float(np.mean(action_efforts)) if action_efforts else float("nan")
    violation_fraction = (
        float(bound_violation_components) / float(total_action_components)
        if total_action_components > 0
        else float("nan")
    )

    mandatory = {
        "orientation_rms": _rms(orientation),
        "angular_velocity_rms": _rms(angular_velocity),
        "action_smoothness": smoothness,
        "action_effort": effort,
        "completion_fraction": completion_fraction,
    }
    if not all(np.isfinite(value) for value in mandatory.values()):
        raise FloatingPointError(
            f"Non-finite episode stability metric: {mandatory}"
        )

    return EpisodeEvaluation(
        episode_return=float(total_reward),
        episode_length=int(episode_length),
        root_orientation_rms_rad=mandatory["orientation_rms"],
        root_orientation_mean_abs_rad=float(np.mean(np.abs(orientation))),
        root_orientation_p95_abs_rad=float(np.quantile(np.abs(orientation), 0.95)),
        root_angular_velocity_rms_rad_s=mandatory["angular_velocity_rms"],
        root_angular_velocity_mean_abs_rad_s=float(
            np.mean(np.abs(angular_velocity))
        ),
        root_angular_velocity_p95_abs_rad_s=float(
            np.quantile(np.abs(angular_velocity), 0.95)
        ),
        normalized_action_smoothness_mse=mandatory["action_smoothness"],
        normalized_action_effort_mse=mandatory["action_effort"],
        action_bound_violation_fraction=violation_fraction,
        episode_completion_fraction=mandatory["completion_fraction"],
        survived_full_horizon=survived_full_horizon,
        ended_by_termination=int(bool(terminated)),
        ended_by_truncation=int(bool(truncated)),
        forward_displacement=forward_displacement,
        mean_forward_speed=mean_forward_speed,
        environment_dt=environment_dt,
        max_episode_steps=max_episode_steps,
    )


def evaluate_checkpoint(
    method: str,
    env_name: str,
    training_seed: int,
    champion: pd.Series,
    test_seeds: Sequence[int],
) -> pd.DataFrame:
    checkpoint_path = Path(str(champion["checkpoint_path"]))
    algo: Optional[Algorithm] = None
    env: Optional[gym.Env] = None

    try:
        algo = _restore_algorithm(checkpoint_path)
        env = gym.make(env_name)
        _validate_algorithm_environment(algo, env, env_name)

        rows: List[dict] = []
        for episode_index, test_seed in enumerate(test_seeds, start=1):
            evaluation = _one_test_episode(
                algo, env, env_name, int(test_seed)
            )
            rows.append(
                {
                    "method": method,
                    "run_name": RUN_NAMES[method],
                    "environment": env_name,
                    "training_seed": training_seed,
                    "champion_agent": champion["agent_id"],
                    "training_terminal_score": champion["training_terminal_score"],
                    "last_training_return": champion["last_training_return"],
                    "execution_segment_start": champion["execution_segment_start"],
                    "post_config_support_steps": champion[
                        "post_config_support_steps"
                    ],
                    "checkpoint_path": str(checkpoint_path),
                    "explore": EXPLORE,
                    "test_episode": episode_index,
                    "test_seed": int(test_seed),
                    "stability_metric_version": STABILITY_METRIC_VERSION,
                    "stability_root_body": STABILITY_SPECS[env_name].root_label,
                    "test_return": evaluation.episode_return,
                    "test_episode_length": evaluation.episode_length,
                    "root_orientation_rms_rad": evaluation.root_orientation_rms_rad,
                    "root_orientation_mean_abs_rad": (
                        evaluation.root_orientation_mean_abs_rad
                    ),
                    "root_orientation_p95_abs_rad": (
                        evaluation.root_orientation_p95_abs_rad
                    ),
                    "root_angular_velocity_rms_rad_s": (
                        evaluation.root_angular_velocity_rms_rad_s
                    ),
                    "root_angular_velocity_mean_abs_rad_s": (
                        evaluation.root_angular_velocity_mean_abs_rad_s
                    ),
                    "root_angular_velocity_p95_abs_rad_s": (
                        evaluation.root_angular_velocity_p95_abs_rad_s
                    ),
                    "normalized_action_smoothness_mse": (
                        evaluation.normalized_action_smoothness_mse
                    ),
                    "normalized_action_effort_mse": (
                        evaluation.normalized_action_effort_mse
                    ),
                    "action_bound_violation_fraction": (
                        evaluation.action_bound_violation_fraction
                    ),
                    "episode_completion_fraction": (
                        evaluation.episode_completion_fraction
                    ),
                    "survived_full_horizon": evaluation.survived_full_horizon,
                    "ended_by_termination": evaluation.ended_by_termination,
                    "ended_by_truncation": evaluation.ended_by_truncation,
                    "forward_displacement": evaluation.forward_displacement,
                    "mean_forward_speed": evaluation.mean_forward_speed,
                    "environment_dt": evaluation.environment_dt,
                    "max_episode_steps": evaluation.max_episode_steps,
                }
            )

        episodes = pd.DataFrame(rows)
        expected_seeds = list(map(int, test_seeds))
        observed_seeds = episodes["test_seed"].astype(int).tolist()
        if observed_seeds != expected_seeds:
            raise RuntimeError("Held-out episode seeds were not evaluated exactly once")
        if len(episodes) != len(expected_seeds):
            raise RuntimeError("Incorrect number of held-out episodes")
        if not np.isfinite(episodes["test_return"]).all():
            raise FloatingPointError("Held-out test contains non-finite returns")

        for column in PRIMARY_STABILITY_COLUMNS:
            values = pd.to_numeric(episodes[column], errors="coerce")
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"Held-out stability output contains non-finite {column}"
                )

        return episodes

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if algo is not None:
            try:
                algo.stop()
            except Exception:
                pass


# =============================================================================
# OUTPUT / RESUME HELPERS
# =============================================================================


def _case_output_path(method: str, env_name: str, training_seed: int) -> Path:
    return CASE_OUTPUT_ROOT / env_name / f"{method}_seed{training_seed}.csv"


def _completed_case_is_compatible(
    path: Path,
    champion: pd.Series,
    test_seeds: Sequence[int],
) -> bool:
    if not path.exists():
        return False
    try:
        existing = pd.read_csv(path)
    except Exception:
        return False

    required = {
        "champion_agent",
        "checkpoint_path",
        "test_seed",
        "test_return",
        "explore",
        "stability_metric_version",
        *PRIMARY_STABILITY_COLUMNS,
    }
    if not required.issubset(existing.columns):
        return False
    if len(existing) != len(test_seeds):
        return False
    if existing["champion_agent"].astype(str).nunique() != 1:
        return False
    if str(existing["champion_agent"].iloc[0]) != str(champion["agent_id"]):
        return False
    if existing["checkpoint_path"].astype(str).nunique() != 1:
        return False
    if str(existing["checkpoint_path"].iloc[0]) != str(champion["checkpoint_path"]):
        return False
    if existing["test_seed"].astype(int).tolist() != list(map(int, test_seeds)):
        return False
    if existing["stability_metric_version"].astype(str).nunique() != 1:
        return False
    if str(existing["stability_metric_version"].iloc[0]) != STABILITY_METRIC_VERSION:
        return False
    if not np.isfinite(pd.to_numeric(existing["test_return"], errors="coerce")).all():
        return False
    for column in PRIMARY_STABILITY_COLUMNS:
        if not np.isfinite(pd.to_numeric(existing[column], errors="coerce")).all():
            return False
    if not (existing["explore"].astype(str).str.lower().isin(["false", "0"]).all()):
        return False
    return True


def _seed_level_test_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 100 episode-level measurements into one result per training seed."""
    if episodes.empty:
        return pd.DataFrame()

    keys = [
        "method",
        "run_name",
        "environment",
        "training_seed",
        "champion_agent",
        "checkpoint_path",
    ]

    def q25(values: pd.Series) -> float:
        return float(values.quantile(0.25))

    def q75(values: pd.Series) -> float:
        return float(values.quantile(0.75))

    summary = (
        episodes.groupby(keys, dropna=False, as_index=False)
        .agg(
            training_terminal_score=("training_terminal_score", "first"),
            last_training_return=("last_training_return", "first"),
            post_config_support_steps=("post_config_support_steps", "first"),
            stability_metric_version=("stability_metric_version", "first"),
            stability_root_body=("stability_root_body", "first"),
            n_test_episodes=("test_return", "size"),
            test_mean_return=("test_return", "mean"),
            test_median_return=("test_return", "median"),
            test_std_return=("test_return", "std"),
            test_p25_return=("test_return", q25),
            test_p75_return=("test_return", q75),
            test_min_return=("test_return", "min"),
            test_max_return=("test_return", "max"),
            test_mean_episode_length=("test_episode_length", "mean"),
            mean_root_orientation_rms_rad=("root_orientation_rms_rad", "mean"),
            median_root_orientation_rms_rad=("root_orientation_rms_rad", "median"),
            mean_root_angular_velocity_rms_rad_s=(
                "root_angular_velocity_rms_rad_s",
                "mean",
            ),
            median_root_angular_velocity_rms_rad_s=(
                "root_angular_velocity_rms_rad_s",
                "median",
            ),
            mean_normalized_action_smoothness_mse=(
                "normalized_action_smoothness_mse",
                "mean",
            ),
            median_normalized_action_smoothness_mse=(
                "normalized_action_smoothness_mse",
                "median",
            ),
            mean_normalized_action_effort_mse=(
                "normalized_action_effort_mse",
                "mean",
            ),
            median_normalized_action_effort_mse=(
                "normalized_action_effort_mse",
                "median",
            ),
            full_horizon_survival_rate=("survived_full_horizon", "mean"),
            termination_rate=("ended_by_termination", "mean"),
            mean_episode_completion_fraction=(
                "episode_completion_fraction",
                "mean",
            ),
            mean_forward_displacement=("forward_displacement", "mean"),
            mean_forward_speed=("mean_forward_speed", "mean"),
            mean_action_bound_violation_fraction=(
                "action_bound_violation_fraction",
                "mean",
            ),
        )
    )
    summary["test_iqr_return"] = (
        summary["test_p75_return"] - summary["test_p25_return"]
    )
    summary["test_sem_return"] = summary["test_std_return"] / np.sqrt(
        summary["n_test_episodes"]
    )
    return summary


def _method_environment_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive aggregation across independent training seeds."""
    if seed_summary.empty:
        return pd.DataFrame()

    def q25(values: pd.Series) -> float:
        return float(values.quantile(0.25))

    def q75(values: pd.Series) -> float:
        return float(values.quantile(0.75))

    result = (
        seed_summary.groupby(["method", "run_name", "environment"], as_index=False)
        .agg(
            n_training_seeds=("training_seed", "nunique"),
            median_test_mean_return=("test_mean_return", "median"),
            p25_test_mean_return=("test_mean_return", q25),
            p75_test_mean_return=("test_mean_return", q75),
            mean_test_mean_return=("test_mean_return", "mean"),
            median_root_orientation_rms_rad=(
                "mean_root_orientation_rms_rad",
                "median",
            ),
            p25_root_orientation_rms_rad=(
                "mean_root_orientation_rms_rad",
                q25,
            ),
            p75_root_orientation_rms_rad=(
                "mean_root_orientation_rms_rad",
                q75,
            ),
            median_root_angular_velocity_rms_rad_s=(
                "mean_root_angular_velocity_rms_rad_s",
                "median",
            ),
            p25_root_angular_velocity_rms_rad_s=(
                "mean_root_angular_velocity_rms_rad_s",
                q25,
            ),
            p75_root_angular_velocity_rms_rad_s=(
                "mean_root_angular_velocity_rms_rad_s",
                q75,
            ),
            median_normalized_action_smoothness_mse=(
                "mean_normalized_action_smoothness_mse",
                "median",
            ),
            p25_normalized_action_smoothness_mse=(
                "mean_normalized_action_smoothness_mse",
                q25,
            ),
            p75_normalized_action_smoothness_mse=(
                "mean_normalized_action_smoothness_mse",
                q75,
            ),
            median_normalized_action_effort_mse=(
                "mean_normalized_action_effort_mse",
                "median",
            ),
            median_full_horizon_survival_rate=(
                "full_horizon_survival_rate",
                "median",
            ),
            median_termination_rate=("termination_rate", "median"),
            median_episode_completion_fraction=(
                "mean_episode_completion_fraction",
                "median",
            ),
            median_mean_forward_speed=("mean_forward_speed", "median"),
        )
    )
    result["iqr_test_mean_return"] = (
        result["p75_test_mean_return"] - result["p25_test_mean_return"]
    )
    result["iqr_root_orientation_rms_rad"] = (
        result["p75_root_orientation_rms_rad"]
        - result["p25_root_orientation_rms_rad"]
    )
    result["iqr_root_angular_velocity_rms_rad_s"] = (
        result["p75_root_angular_velocity_rms_rad_s"]
        - result["p25_root_angular_velocity_rms_rad_s"]
    )
    result["iqr_normalized_action_smoothness_mse"] = (
        result["p75_normalized_action_smoothness_mse"]
        - result["p25_normalized_action_smoothness_mse"]
    )
    return result


def _write_aggregate_outputs(
    candidates: Sequence[pd.DataFrame],
    episodes: Sequence[pd.DataFrame],
    failures: Sequence[dict],
) -> None:
    candidates_df = (
        pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    )
    episodes_df = (
        pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame()
    )
    seed_summary = _seed_level_test_summary(episodes_df)
    method_summary = _method_environment_summary(seed_summary)

    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(
        TEST_OUTPUT_ROOT / "training_champion_candidates.csv", index=False
    )
    episodes_df.to_csv(
        TEST_OUTPUT_ROOT / "heldout_stability_episodes.csv", index=False
    )
    seed_summary.to_csv(
        TEST_OUTPUT_ROOT / "heldout_stability_seed_summary.csv", index=False
    )
    method_summary.to_csv(
        TEST_OUTPUT_ROOT / "heldout_stability_method_environment_summary.csv",
        index=False,
    )
    pd.DataFrame(failures).to_csv(
        TEST_OUTPUT_ROOT / "heldout_stability_failures.csv", index=False
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    _ensure_project_on_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    overlap = set(TRAINING_SEEDS) & set(TEST_EPISODE_SEEDS)
    if overlap:
        raise ValueError(f"Test seeds overlap training seeds: {sorted(overlap)}")

    if SMOKE_TEST:
        methods = SMOKE_METHODS
        environments = [SMOKE_ENV]
        training_seeds = [SMOKE_TRAINING_SEED]
        test_seeds = TEST_EPISODE_SEEDS[:SMOKE_TEST_EPISODES]
    else:
        methods = list(RUN_NAMES.keys())
        environments = ENVIRONMENTS
        training_seeds = TRAINING_SEEDS
        test_seeds = TEST_EPISODE_SEEDS

    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CASE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    protocol = {
        "run_names": RUN_NAMES,
        "environments": environments,
        "training_seeds": training_seeds,
        "terminal_window_steps": TERMINAL_WINDOW_STEPS,
        "minimum_terminal_support_steps": MIN_TERMINAL_SUPPORT_STEPS,
        "minimum_post_config_support_steps": MIN_POST_CONFIG_SUPPORT_STEPS,
        "minimum_terminal_points": MIN_TERMINAL_POINTS,
        "target_training_steps": TARGET_TRAINING_STEPS,
        "config_columns_used_for_final_segment": list(CONFIG_COLUMNS),
        "asha_require_full_budget": ASHA_REQUIRE_FULL_BUDGET,
        "asha_expected_num_trials": ASHA_EXPECTED_NUM_TRIALS,
        "n_test_episodes": len(test_seeds),
        "test_episode_seeds": list(map(int, test_seeds)),
        "explore": EXPLORE,
        "stability_metric_version": STABILITY_METRIC_VERSION,
        "stability_environment_specs": {
            name: asdict(spec) for name, spec in STABILITY_SPECS.items()
        },
        "stability_definitions": {
            "root_orientation_rms_rad": (
                "RMS geodesic orientation deviation from the episode-reset "
                "root-body orientation; wrapped angle for planar agents and "
                "quaternion geodesic distance for Ant/Humanoid."
            ),
            "root_angular_velocity_rms_rad_s": (
                "RMS magnitude of root-body angular velocity from the default "
                "environment observation."
            ),
            "normalized_action_smoothness_mse": (
                "Mean squared change between consecutive actions after affine "
                "normalization by each action coordinate's Box bounds."
            ),
            "normalized_action_effort_mse": (
                "Mean squared normalized action magnitude."
            ),
            "survived_full_horizon": (
                "One when the episode reaches env.spec.max_episode_steps without "
                "environment termination; otherwise zero."
            ),
            "interpretation": (
                "Return remains primary. Lower angular motion/action metrics "
                "indicate less motion or smoother control, not automatically "
                "better task performance; comparisons are within environment."
            ),
        },
        "champion_selection": (
            "Highest time-weighted mean training return over up to the final "
            "200k interactions of the final causal branch; requires at least "
            "100k interactions after the last configuration change and at least "
            "three finite observations."
        ),
        "statistical_unit": "independent training seed",
    }
    (TEST_OUTPUT_ROOT / "test_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "test_episode": np.arange(1, len(test_seeds) + 1),
            "test_seed": test_seeds,
        }
    ).to_csv(TEST_OUTPUT_ROOT / "heldout_episode_seeds.csv", index=False)

    # ------------------------------------------------------------------
    # Preflight: validate every requested training run and freeze champions
    # before test data are generated.
    # ------------------------------------------------------------------
    all_candidates: List[pd.DataFrame] = []
    champions: Dict[Tuple[str, str, int], pd.Series] = {}
    preflight_failures: List[dict] = []

    for env_name in environments:
        for training_seed in training_seeds:
            for method in methods:
                try:
                    candidates, champion = select_training_champion(
                        method, env_name, training_seed
                    )
                    all_candidates.append(candidates)
                    champions[(method, env_name, training_seed)] = champion
                except Exception as exc:
                    preflight_failures.append(
                        {
                            "stage": "preflight",
                            "method": method,
                            "environment": env_name,
                            "training_seed": training_seed,
                            "error": repr(exc),
                        }
                    )

    _write_aggregate_outputs(all_candidates, [], preflight_failures)

    if preflight_failures and REQUIRE_PREFLIGHT_SUCCESS:
        raise RuntimeError(
            f"Preflight failed for {len(preflight_failures)} cases. Review "
            f"{TEST_OUTPUT_ROOT / 'heldout_stability_failures.csv'} before testing."
        )

    # ------------------------------------------------------------------
    # Held-out evaluation.
    # ------------------------------------------------------------------
    all_episodes: List[pd.DataFrame] = []
    failures: List[dict] = list(preflight_failures)

    ray.init(
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
        log_to_driver=False,
        include_dashboard=False,
    )

    started = time.time()
    total = len(champions)
    completed = 0

    try:
        for env_name in environments:
            for training_seed in training_seeds:
                for method in methods:
                    key = (method, env_name, training_seed)
                    if key not in champions:
                        continue

                    completed += 1
                    champion = champions[key]
                    print("\n" + "=" * 80)
                    print(
                        f"[{completed}/{total}] {method} | {env_name} | "
                        f"training seed={training_seed} | champion={champion['agent_id']}"
                    )
                    print("=" * 80)

                    case_path = _case_output_path(method, env_name, training_seed)
                    case_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        if RESUME_COMPLETED_CASES and _completed_case_is_compatible(
                            case_path, champion, test_seeds
                        ):
                            episodes = pd.read_csv(case_path)
                            logger.info("Resuming completed case: %s", case_path)
                        else:
                            episodes = evaluate_checkpoint(
                                method,
                                env_name,
                                training_seed,
                                champion,
                                test_seeds,
                            )
                            episodes.to_csv(case_path, index=False)

                        all_episodes.append(episodes)
                        print(
                            f"Held-out test ({len(episodes)} episodes): "
                            f"return_mean={episodes['test_return'].mean():.3f} | "
                            f"angular_rms={episodes['root_angular_velocity_rms_rad_s'].mean():.4f} | "
                            f"orientation_rms={episodes['root_orientation_rms_rad'].mean():.4f} | "
                            f"action_smoothness={episodes['normalized_action_smoothness_mse'].mean():.6f} | "
                            f"survival={episodes['survived_full_horizon'].mean():.3f}"
                        )

                    except Exception as exc:
                        logger.exception(
                            "Evaluation failed: %s | %s | seed=%s",
                            method,
                            env_name,
                            training_seed,
                        )
                        failures.append(
                            {
                                "stage": "evaluation",
                                "method": method,
                                "environment": env_name,
                                "training_seed": training_seed,
                                "error": repr(exc),
                            }
                        )

                    # Persist after every case so an interruption does not lose
                    # completed evaluations.
                    _write_aggregate_outputs(all_candidates, all_episodes, failures)

                    if failures and FAIL_FAST_DURING_EVALUATION:
                        raise RuntimeError("Evaluation failed; see failure CSV")

    finally:
        if ray.is_initialized():
            ray.shutdown()

    elapsed = time.time() - started
    _write_aggregate_outputs(all_candidates, all_episodes, failures)

    print("\n" + "=" * 80)
    print("HELD-OUT RETURN + STABILITY CAMPAIGN COMPLETE")
    print(f"Output: {TEST_OUTPUT_ROOT}")
    print(f"Elapsed: {elapsed / 60.0:.1f} min")
    print(f"Failures: {len(failures)}")
    print("=" * 80)

    if failures:
        print(
            "The campaign produced partial outputs, but it is not statistically "
            "complete until every requested matched case succeeds."
        )


if __name__ == "__main__":
    main()