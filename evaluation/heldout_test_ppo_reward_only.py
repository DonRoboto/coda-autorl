



from __future__ import annotations

"""
Select one training-side champion per (method, environment, training seed)
and evaluate its FINAL checkpoint on held-out MuJoCo episodes.

Selection is performed using TRAINING DATA ONLY. Test returns are never used
for checkpoint selection.

PPO NOTE:
- This file evaluates the old-stack PPO checkpoints produced by the refactored
  experiments/ppo/train_pbt.py, train_pb2.py, train_asha.py, and train_coda.py.
- The checkpoint-inference path remains Algorithm.from_checkpoint() followed by
  Algorithm.compute_actions(..., explore=False), preserving the restored PPO
  preprocessor/MeanStdFilter path.
- Shared environments, training seeds, held-out seeds, population size, ASHA
  trial count, training horizon, and champion-selection thresholds are imported
  from configs/ rather than duplicated here.
- Run this file from the repository root. The repository and src/ directories
  are added to sys.path before checkpoint restoration so custom CODA modules are
  importable.

This reward-only version evaluates held-out episodic return and episode length
only. It intentionally omits locomotion/control-quality metrics so the full
held-out campaign can be executed with less Python-side bookkeeping.

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
- PBT+PPO, PB2+PPO, ASHA+PPO, and CODA+PPO must have saved terminal checkpoints with
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
import sys
import time
from dataclasses import dataclass
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
#     <repo>/evaluation/heldout_test_ppo_reward_only.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Final repository layout keeps PPO and SAC artifacts isolated. Override this
# only when intentionally evaluating an archived/migrated result tree.
_RESULTS_ROOT_OVERRIDE = os.environ.get("PPO_RESULTS_ROOT", "").strip()
if _RESULTS_ROOT_OVERRIDE:
    _root = Path(_RESULTS_ROOT_OVERRIDE).expanduser()
    if not _root.is_absolute():
        _root = REPO_ROOT / _root
    RESULTS_ROOT = _root.resolve()
else:
    RESULTS_ROOT = REPO_ROOT / "results" / "ppo"

for candidate in (REPO_ROOT, SRC_ROOT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from configs.environments import ENVIRONMENTS  # noqa: E402
from configs.seeds import (  # noqa: E402
    TRAINING_SEEDS,
    HELDOUT_RETURN_SEEDS,
)
from configs import ppo_config as ppo_cfg  # noqa: E402

POPULATION_SIZE = int(ppo_cfg.POPULATION_SIZE)
MAX_TIMESTEPS_PER_WORKER = int(ppo_cfg.MAX_TIMESTEPS_PER_WORKER)
TERMINAL_WINDOW_STEPS = int(ppo_cfg.TERMINAL_WINDOW_STEPS)
MIN_TERMINAL_SUPPORT_STEPS = int(ppo_cfg.MIN_TERMINAL_SUPPORT_STEPS)
MIN_POST_CONFIG_SUPPORT_STEPS = int(ppo_cfg.MIN_POST_CONFIG_SUPPORT_STEPS)
MIN_TERMINAL_POINTS = int(ppo_cfg.MIN_TERMINAL_POINTS)

# The current centralized PPO config freezes ASHA_NUM_SAMPLES. getattr keeps
# older archives readable; if absent, exact ASHA trial-count validation is
# disabled while full-budget validation remains mandatory.
ASHA_NUM_SAMPLES = getattr(ppo_cfg, "ASHA_NUM_SAMPLES", None)

# ---------------------------------------------------------------------------
# Training-run names produced by experiments/ppo/*.py
# ---------------------------------------------------------------------------

RUN_NAMES: Dict[str, str] = {
    "PBT": "PBT_PPO_HPO",
    "PB2": "PB2_PPO_HPO",
    "ASHA": "ASHA_PPO_HPO",
    "CODA": "CODA_FULL",
    "CODA-I2O": "CODA_I2O",
    "CODA-O2I": "CODA_O2I",
}

EXPECTED_POPULATION_SIZE: Dict[str, int] = {
    "PBT": POPULATION_SIZE,
    "PB2": POPULATION_SIZE,
    "CODA": POPULATION_SIZE,
    "CODA-I2O": POPULATION_SIZE,
    "CODA-O2I": POPULATION_SIZE,
}

ASHA_EXPECTED_NUM_TRIALS: Optional[int] = (
    int(ASHA_NUM_SAMPLES) if ASHA_NUM_SAMPLES is not None else None
)

# ---------------------------------------------------------------------------
# Training-side champion-selection protocol
# ---------------------------------------------------------------------------

TARGET_TRAINING_STEPS = MAX_TIMESTEPS_PER_WORKER

# PPO execution coordinates used to identify the final executed configuration
# segment. entropy_coeff is included because it is the CODA-PPO O2I actuator.
# It remains fixed at zero for PBT/PB2/ASHA and CODA-I2O, so it does not create
# artificial configuration changes for those methods.
CONFIG_COLUMNS: Tuple[str, ...] = (
    "config/train_batch_size",
    "config/lambda",
    "config/clip_param",
    "config/lr",
    "config/entropy_coeff",
)

ASHA_REQUIRE_FULL_BUDGET = True

# Do not silently accept files such as metrics_...(1).csv in the final campaign.
ALLOW_NONCANONICAL_METRICS = False

# ---------------------------------------------------------------------------
# Held-out policy-evaluation protocol
# ---------------------------------------------------------------------------

TEST_EPISODE_SEEDS = list(HELDOUT_RETURN_SEEDS)
N_TEST_EPISODES = len(TEST_EPISODE_SEEDS)
EXPLORE = False

# Safe smoke evaluation is enabled by default.
#
# Full campaign:
#   PPO_HELDOUT_SMOKE=0 python -u evaluation/heldout_test_ppo_reward_only.py
SMOKE_TEST = os.environ.get("PPO_HELDOUT_SMOKE", "1") != "0"
# SMOKE_ENV = os.environ.get("PPO_HELDOUT_SMOKE_ENV", "Walker2d-v5")
# SMOKE_TRAINING_SEED = int(
#     os.environ.get(
#         "PPO_HELDOUT_SMOKE_TRAINING_SEED",
#         str(TRAINING_SEEDS[0]),
#     )
# )

def _parse_csv_strings(raw: str) -> List[str]:
    return [
        token.strip()
        for token in str(raw).split(",")
        if token.strip()
    ]


def _parse_csv_ints(raw: str) -> List[int]:
    return [
        int(token.strip())
        for token in str(raw).split(",")
        if token.strip()
    ]


SMOKE_ENVS = _parse_csv_strings(
    os.environ.get(
        "PPO_HELDOUT_SMOKE_ENVS",
        "Walker2d-v5",
    )
)

SMOKE_TRAINING_SEEDS = _parse_csv_ints(
    os.environ.get(
        "PPO_HELDOUT_SMOKE_TRAINING_SEEDS",
        str(TRAINING_SEEDS[0]),
    )
)


# SMOKE_METHODS = [
#     os.environ.get("PPO_HELDOUT_SMOKE_METHOD", "CODA-I2O").strip()
# ]
SMOKE_METHODS = _parse_csv_strings(
    os.environ.get(
        "PPO_HELDOUT_SMOKE_METHODS",
        "CODA-I2O",
    )
)

SMOKE_TEST_EPISODES = int(
    os.environ.get("PPO_HELDOUT_SMOKE_EPISODES", "100")
)

# Full final comparison by default.
DEFAULT_METHODS: Tuple[str, ...] = (
    "PBT",
    "PB2",
    "ASHA",
    "CODA",
    "CODA-I2O",
    "CODA-O2I",
)


def _parse_methods(raw: str, default: Sequence[str]) -> List[str]:
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    selected = tokens if tokens else list(default)
    return list(dict.fromkeys(selected))

REQUESTED_METHODS = _parse_methods(
    os.environ.get("PPO_HELDOUT_METHODS", ""),
    DEFAULT_METHODS,
)

# Resume completed cases only if champion, checkpoint, test seeds, and metric
# protocol all still match.
RESUME_COMPLETED_CASES = True

# No held-out episode is generated until all requested training cases pass
# preflight and champions are frozen from training data only.
REQUIRE_PREFLIGHT_SUCCESS = True

# Preserve already completed cases when one checkpoint fails.
FAIL_FAST_DURING_EVALUATION = False

# Keep reward-only outputs separate from previous stability campaigns.
TEST_OUTPUT_ROOT = RESULTS_ROOT / (
    "heldout_reward_ppo_smoke"
    if SMOKE_TEST
    else "heldout_reward_ppo_final"
)
CASE_OUTPUT_ROOT = TEST_OUTPUT_ROOT / "episodes_by_case"


# =============================================================================
# CONSTANTS
# =============================================================================

METRIC_COL = "env_runners/episode_return_mean"
TIMESTEP_COL = "timesteps_total"
AGENT_COL = "agente_id"

logger = logging.getLogger("champion_test")


EVALUATION_PROTOCOL_VERSION = "reward_only_v1"


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
    """Make repository and CODA package modules importable during restore."""
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
# CHECKPOINT EVALUATION -- REWARD ONLY
# =============================================================================


def _restore_algorithm(checkpoint_path: Path) -> Algorithm:
    logger.info("Restoring checkpoint: %s", checkpoint_path)
    return Algorithm.from_checkpoint(str(checkpoint_path))


def _validate_algorithm_environment(
    algo: Algorithm,
    env: gym.Env,
    env_name: str,
) -> None:
    """Validate only properties needed for reward evaluation."""
    configured_env = getattr(algo.config, "env", None)
    if isinstance(configured_env, str) and configured_env != env_name:
        raise RuntimeError(
            f"Checkpoint environment mismatch: checkpoint={configured_env}, "
            f"requested={env_name}"
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


def _one_test_episode(
    algo: Algorithm,
    env: gym.Env,
    episode_seed: int,
) -> Tuple[float, int]:
    """Run one deterministic held-out episode and return (return, length)."""
    obs, _ = env.reset(seed=int(episode_seed))

    try:
        env.action_space.seed(int(episode_seed))
    except Exception:
        pass

    terminated = False
    truncated = False
    total_reward = 0.0
    episode_length = 0

    while not (terminated or truncated):
        # Preserve the original old-stack PPO inference path.
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

        obs, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        if not np.isfinite(reward):
            raise FloatingPointError(
                f"Non-finite reward in held-out episode seed={episode_seed}, "
                f"step={episode_length}"
            )

        total_reward += reward
        episode_length += 1

    if not np.isfinite(total_reward):
        raise FloatingPointError(
            f"Non-finite episode return for test seed {episode_seed}"
        )

    return float(total_reward), int(episode_length)


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
            episode_return, episode_length = _one_test_episode(
                algo, env, int(test_seed)
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
                    "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                    "test_episode": episode_index,
                    "test_seed": int(test_seed),
                    "test_return": episode_return,
                    "test_episode_length": episode_length,
                }
            )

        episodes = pd.DataFrame(rows)
        expected_seeds = list(map(int, test_seeds))
        observed_seeds = episodes["test_seed"].astype(int).tolist()
        if observed_seeds != expected_seeds:
            raise RuntimeError("Held-out episode seeds were not evaluated exactly once")
        if len(episodes) != len(expected_seeds):
            raise RuntimeError("Incorrect number of held-out episodes")
        if not np.isfinite(
            pd.to_numeric(episodes["test_return"], errors="coerce")
        ).all():
            raise FloatingPointError("Held-out test contains non-finite returns")

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
        "test_episode_length",
        "explore",
        "evaluation_protocol_version",
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
    if existing["evaluation_protocol_version"].astype(str).nunique() != 1:
        return False
    if (
        str(existing["evaluation_protocol_version"].iloc[0])
        != EVALUATION_PROTOCOL_VERSION
    ):
        return False
    if not np.isfinite(
        pd.to_numeric(existing["test_return"], errors="coerce")
    ).all():
        return False
    if not (existing["explore"].astype(str).str.lower().isin(["false", "0"]).all()):
        return False
    return True


def _seed_level_test_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out episodes into one result per independent training seed."""
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
            evaluation_protocol_version=(
                "evaluation_protocol_version",
                "first",
            ),
            n_test_episodes=("test_return", "size"),
            test_mean_return=("test_return", "mean"),
            test_median_return=("test_return", "median"),
            test_std_return=("test_return", "std"),
            test_p25_return=("test_return", q25),
            test_p75_return=("test_return", q75),
            test_min_return=("test_return", "min"),
            test_max_return=("test_return", "max"),
            test_mean_episode_length=("test_episode_length", "mean"),
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
    """Descriptive reward aggregation across independent training seeds."""
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
            median_test_median_return=("test_median_return", "median"),
            median_test_episode_length=("test_mean_episode_length", "median"),
        )
    )
    result["iqr_test_mean_return"] = (
        result["p75_test_mean_return"] - result["p25_test_mean_return"]
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
        TEST_OUTPUT_ROOT / "heldout_test_episodes.csv", index=False
    )
    seed_summary.to_csv(
        TEST_OUTPUT_ROOT / "heldout_test_seed_summary.csv", index=False
    )
    method_summary.to_csv(
        TEST_OUTPUT_ROOT / "heldout_test_method_environment_summary.csv",
        index=False,
    )
    pd.DataFrame(failures).to_csv(
        TEST_OUTPUT_ROOT / "heldout_failures.csv", index=False
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

    requested_for_validation = SMOKE_METHODS if SMOKE_TEST else REQUESTED_METHODS
    unknown_methods = [
        method for method in requested_for_validation if method not in RUN_NAMES
    ]
    if unknown_methods:
        raise KeyError(
            "Unknown PPO held-out method(s): "
            f"{unknown_methods}. Available: {sorted(RUN_NAMES)}"
        )

    if SMOKE_TEST:
        unknown_envs = [
            env_name
            for env_name in SMOKE_ENVS
            if env_name not in ENVIRONMENTS
        ]
    
        if unknown_envs:
            raise KeyError(
                f"Unknown PPO held-out smoke environments: {unknown_envs}. "
                f"Available: {list(ENVIRONMENTS)}"
            )
    
        unknown_seeds = [
            seed
            for seed in SMOKE_TRAINING_SEEDS
            if seed not in TRAINING_SEEDS
        ]
    
        if unknown_seeds:
            raise KeyError(
                f"Unknown PPO training seeds: {unknown_seeds}. "
                f"Available: {list(TRAINING_SEEDS)}"
            )
    
        methods = SMOKE_METHODS
        environments = list(SMOKE_ENVS)
        training_seeds = list(SMOKE_TRAINING_SEEDS)
        test_seeds = TEST_EPISODE_SEEDS[:SMOKE_TEST_EPISODES]
    
    else:
        methods = REQUESTED_METHODS
        environments = list(ENVIRONMENTS)
        training_seeds = list(TRAINING_SEEDS)
        test_seeds = list(TEST_EPISODE_SEEDS)

    if ASHA_EXPECTED_NUM_TRIALS is None and "ASHA" in methods:
        logger.warning(
            "configs.ppo_config does not expose ASHA_NUM_SAMPLES; exact ASHA "
            "trial-count validation is disabled. Full-budget validation remains mandatory."
        )

    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CASE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    protocol = {
        "learner": "PPO",
        "run_names": {method: RUN_NAMES[method] for method in methods},
        "all_supported_run_names": RUN_NAMES,
        "results_root": str(RESULTS_ROOT),
        "configuration_source": {
            "environments": "configs/environments.py",
            "seeds": "configs/seeds.py",
            "ppo": "configs/ppo_config.py",
        },
        "environments": environments,
        "training_seeds": training_seeds,
        "terminal_window_steps": TERMINAL_WINDOW_STEPS,
        "minimum_terminal_support_steps": MIN_TERMINAL_SUPPORT_STEPS,
        "minimum_post_config_support_steps": MIN_POST_CONFIG_SUPPORT_STEPS,
        "minimum_terminal_points": MIN_TERMINAL_POINTS,
        "target_training_steps": TARGET_TRAINING_STEPS,
        "config_columns_used_for_final_segment": list(CONFIG_COLUMNS),
        "ppo_execution_coordinates": {
            "outer_hpo": ["train_batch_size", "lambda", "clip_param", "lr"],
            "coda_o2i_actuator": "entropy_coeff",
        },
        "asha_require_full_budget": ASHA_REQUIRE_FULL_BUDGET,
        "asha_expected_num_trials": ASHA_EXPECTED_NUM_TRIALS,
        "n_test_episodes": len(test_seeds),
        "test_episode_seeds": list(map(int, test_seeds)),
        "explore": EXPLORE,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "heldout_outcomes": ["episodic_return", "episode_length"],
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
            f"{TEST_OUTPUT_ROOT / 'heldout_failures.csv'} before testing."
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
                            f"Held-out reward test ({len(episodes)} episodes): "
                            f"return_mean={episodes['test_return'].mean():.3f} | "
                            f"return_median={episodes['test_return'].median():.3f} | "
                            f"episode_length_mean={episodes['test_episode_length'].mean():.1f}"
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
    print("HELD-OUT REWARD-ONLY CAMPAIGN COMPLETE")
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