from __future__ import annotations

"""
Select one training-side champion per (method, environment, training seed)
and evaluate its FINAL checkpoint on held-out MuJoCo episodes.

Selection is performed using TRAINING DATA ONLY.
Test returns are never used to choose the checkpoint.

Designed for the PB2SPACE campaign used by:
    PBT, PB2, ASHA, CODA
with RLlib's old API stack and MeanStdFilter.

IMPORTANT:
- Put this file in the project root (or run it from there), so any custom
  callback/module referenced by the RLlib checkpoints is importable.
- Confirm RUN_NAMES below match the names of your final campaign folders.
- PBT, PB2, and CODA training runs must have been created with
  CheckpointConfig(checkpoint_at_end=True).  This evaluation script cannot
  reconstruct a terminal learner state if training did not save it.
"""

# -----------------------------------------------------------------------------
# CPU limits should be set before NumPy / Ray / PyTorch imports.
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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.algorithm import Algorithm


# =============================================================================
# USER CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"

# Final campaign names. Change ONLY if your folders/files use different tags.
# Example alternative for CODA: "CODA_HIM_PB2SPACE".
RUN_NAMES: Dict[str, str] = {
    "PBT": "PBT_PB2SPACE",
    "PB2": "PB2_PB2SPACE",
    "ASHA": "ASHA_PB2SPACE",
    "CODA": "CODA_PB2SPACE",
}

ENVIRONMENTS = [
    "Hopper-v5",
    # "HalfCheetah-v5",
    # "Walker2d-v5",    
    # "Swimmer-v5",
    # "Ant-v5",
    # "Humanoid-v5",
]

TRAINING_SEEDS = [
    1042,
    2854,
    3910,
    4721,
    5603,
    6198,
    7433,
    8256,
    9107,
    9845,
]

# -----------------------------------------------------------------------------
# Training-side champion criterion
# -----------------------------------------------------------------------------
# We select the FINAL checkpoint of the worker/trial with the largest average
# return over the last 200k interactions of its FINAL CAUSAL BRANCH.
#
# The score is a time-weighted average (terminal AUC / terminal support), using
# linear interpolation. This avoids favoring workers that simply report more
# training iterations because their batch sizes are smaller.
#
# No historical maximum and no test return is used for selection.
TERMINAL_WINDOW_STEPS = 200_000
# Require enough evidence on the final causal branch so that a late checkpoint
# restoration followed by only one or two observations cannot win by chance.
MIN_TERMINAL_SUPPORT_STEPS = 100_000
MIN_TERMINAL_POINTS = 3
TARGET_TRAINING_STEPS = 2_000_000

# ASHA: only full-budget trials are eligible for champion selection.
ASHA_REQUIRE_FULL_BUDGET = True

# -----------------------------------------------------------------------------
# Held-out test protocol
# -----------------------------------------------------------------------------
# Recommended primary test size: 100 episodes per selected checkpoint.
N_TEST_EPISODES = 100

# Same held-out reset seeds for EVERY algorithm and EVERY training seed.
# These do not overlap with TRAINING_SEEDS.
TEST_EPISODE_SEEDS = list(range(100_000, 100_000 + N_TEST_EPISODES))

EXPLORE = False

# Useful for a quick validation before launching the entire campaign.
SMOKE_TEST = True
SMOKE_ENV = "Hopper-v5"
SMOKE_TRAINING_SEED = 2854 
#SMOKE_METHODS = ["PBT", "PB2", "ASHA", "CODA"]
SMOKE_METHODS = ["CODA"]
SMOKE_TEST_EPISODES = 100

# Output directory.
TEST_OUTPUT_ROOT = RESULTS_ROOT / "heldout_test"

# Fail immediately if a required metrics/checkpoint file is missing.
STRICT = True


# =============================================================================
# CONSTANTS
# =============================================================================

METRIC_COL = "env_runners/episode_return_mean"
TIMESTEP_COL = "timesteps_total"
AGENT_COL = "agente_id"

logger = logging.getLogger("champion_test")


# =============================================================================
# PATH / DATA HELPERS
# =============================================================================

def _ensure_project_on_path() -> None:
    """Make custom runner/callback modules importable during checkpoint restore."""
    for candidate in [PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
        s = str(candidate)
        if s not in sys.path:
            sys.path.insert(0, s)


def _metrics_path(env_name: str, run_name: str, seed: int) -> Path:
    """Return exact metrics path, refusing silent ambiguity."""
    expected = (
        RESULTS_ROOT
        / "metrics"
        / env_name
        / f"metrics_{run_name}_seed{seed}.csv"
    )
    if expected.exists():
        return expected

    # Helpful fallback for accidental suffixes such as "(1)" or copied files.
    parent = expected.parent
    matches = sorted(parent.glob(f"metrics_{run_name}_seed{seed}*.csv"))

    if len(matches) == 1:
        logger.warning("Using noncanonical metrics file: %s", matches[0])
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous metrics files for {run_name}, {env_name}, seed={seed}:\n"
            + "\n".join(f"  - {p}" for p in matches)
            + "\nRemove/rename old copies or set RUN_NAMES to the exact final campaign."
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
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    return path


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _chronological_worker_history(df: pd.DataFrame) -> pd.DataFrame:
    """Recover the actual causal execution order within one worker/trial.

    Final-campaign files explicitly store ``causal_order``.  Prefer it over
    training_iteration or timesteps_total, both of which may decrease or repeat
    after population checkpoint inheritance.  The remaining fields are kept as
    backward-compatible fallbacks for older metric files.
    """
    out = df.copy()
    out["__row_order"] = np.arange(len(out), dtype=np.int64)

    if "causal_order" in out.columns:
        order = _safe_numeric(out["causal_order"])
        if order.notna().all():
            out["__order"] = order
            return (
                out.sort_values(["__order", "__row_order"], kind="stable")
                .drop(columns=["__order"])
                .reset_index(drop=True)
            )

    # Backward-compatible fallbacks for older outputs.
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


def _final_causal_branch(worker_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the final monotone timesteps_total branch of a worker."""
    hist = _chronological_worker_history(worker_df)

    if TIMESTEP_COL not in hist.columns:
        return hist

    t = _safe_numeric(hist[TIMESTEP_COL])
    reset_positions = np.flatnonzero((t.diff() < 0).fillna(False).to_numpy())

    if len(reset_positions) == 0:
        return hist.reset_index(drop=True)

    # reset_positions contains the row index of the first observation AFTER
    # a checkpoint-induced timestep decrease. Keep the latest such branch.
    start = int(reset_positions[-1])
    return hist.iloc[start:].reset_index(drop=True)


def _terminal_time_weighted_score(
    worker_df: pd.DataFrame,
    window_steps: int = TERMINAL_WINDOW_STEPS,
) -> Tuple[float, float, float, int, float]:
    """
    Time-weighted average return over the terminal interaction window.

    Returns:
        score,
        window_start,
        window_end,
        n_observations_used,
        last_training_return
    """
    branch = _final_causal_branch(worker_df)

    if METRIC_COL not in branch.columns or TIMESTEP_COL not in branch.columns:
        return np.nan, np.nan, np.nan, 0, np.nan

    x = _safe_numeric(branch[TIMESTEP_COL])
    y = _safe_numeric(branch[METRIC_COL])

    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    xy = pd.DataFrame({"T": x[valid], "R": y[valid]})
    if xy.empty:
        return np.nan, np.nan, np.nan, 0, np.nan

    # Within the final branch, duplicated T values can occur. Keep the latest
    # observation at a duplicated training counter.
    xy = xy.reset_index(drop=True)
    xy = xy.groupby("T", as_index=False, sort=True).last()

    t_end = float(xy["T"].iloc[-1])
    t_first = float(xy["T"].iloc[0])
    t_start = max(t_first, t_end - float(window_steps))
    last_return = float(xy["R"].iloc[-1])

    # Degenerate one-point branch: fall back to its last observed training return.
    if len(xy) == 1 or not (t_end > t_start):
        return last_return, t_start, t_end, 1, last_return

    T = xy["T"].to_numpy(dtype=float)
    R = xy["R"].to_numpy(dtype=float)

    # Interpolate a value exactly at t_start if the terminal window begins
    # between two reported observations.
    r_start = float(np.interp(t_start, T, R))

    inside = (T > t_start) & (T <= t_end)
    T_win = np.concatenate([[t_start], T[inside]])
    R_win = np.concatenate([[r_start], R[inside]])

    if len(T_win) < 2 or T_win[-1] <= T_win[0]:
        score = last_return
    else:
        # np.trapezoid was introduced after np.trapz; support both.
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(R_win, T_win))
        else:
            auc = float(np.trapz(R_win, T_win))
        score = auc / float(T_win[-1] - T_win[0])

    return score, t_start, t_end, int(len(T_win)), last_return


def _asha_is_full_budget(worker_df: pd.DataFrame) -> bool:
    if "asha_reached_max_resource" in worker_df.columns:
        flag = _safe_numeric(worker_df["asha_reached_max_resource"])
        if flag.notna().any():
            return bool(flag.max() >= 1)

    if TIMESTEP_COL not in worker_df.columns:
        return False

    t = _safe_numeric(worker_df[TIMESTEP_COL])
    return bool(t.notna().any() and float(t.max()) >= TARGET_TRAINING_STEPS)


def select_training_champion(
    method: str,
    env_name: str,
    training_seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Select champion using only training metrics."""
    run_name = RUN_NAMES[method]
    path = _metrics_path(env_name, run_name, training_seed)
    df = pd.read_csv(path)

    required = {AGENT_COL, METRIC_COL, TIMESTEP_COL}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

    rows = []

    for agent_id, worker_df in df.groupby(AGENT_COL, sort=False):
        eligible = True
        eligibility_reason = "eligible"

        if method == "ASHA" and ASHA_REQUIRE_FULL_BUDGET:
            eligible = _asha_is_full_budget(worker_df)
            if not eligible:
                eligibility_reason = "ASHA trial did not reach full budget"

        try:
            checkpoint = _checkpoint_dir(
                env_name,
                run_name,
                training_seed,
                str(agent_id),
            )
            checkpoint_exists = True
        except FileNotFoundError:
            checkpoint = Path("")
            checkpoint_exists = False
            eligible = False
            eligibility_reason = "checkpoint missing"

        score, t0, t1, n_pts, last_return = _terminal_time_weighted_score(worker_df)

        terminal_support = (
            float(t1 - t0)
            if np.isfinite(t0) and np.isfinite(t1)
            else np.nan
        )

        if not np.isfinite(score):
            eligible = False
            eligibility_reason = "terminal training score unavailable"
        elif (
            not np.isfinite(terminal_support)
            or terminal_support < MIN_TERMINAL_SUPPORT_STEPS
        ):
            eligible = False
            eligibility_reason = "insufficient terminal branch support"
        elif n_pts < MIN_TERMINAL_POINTS:
            eligible = False
            eligibility_reason = "insufficient terminal observations"

        rows.append(
            {
                "method": method,
                "run_name": run_name,
                "environment": env_name,
                "training_seed": training_seed,
                "agent_id": str(agent_id),
                "eligible": int(eligible),
                "eligibility_reason": eligibility_reason,
                "training_terminal_score": score,
                "terminal_window_start": t0,
                "terminal_window_end": t1,
                "terminal_window_support": terminal_support,
                "terminal_points": n_pts,
                "last_training_return": last_return,
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

    # Deterministic tie-breaking:
    # 1) larger terminal training score
    # 2) larger last training return
    # 3) lexical agent_id
    eligible_df = eligible_df.sort_values(
        ["training_terminal_score", "last_training_return", "agent_id"],
        ascending=[False, False, True],
        kind="stable",
    )

    winner_idx = eligible_df.index[0]
    candidates["selected_champion"] = 0
    candidates.loc[winner_idx, "selected_champion"] = 1

    winner = candidates.loc[winner_idx].copy()
    return candidates, winner


# =============================================================================
# CHECKPOINT EVALUATION
# =============================================================================

def _restore_algorithm(checkpoint_path: Path) -> Algorithm:
    """Restore an RLlib Algorithm checkpoint without changing its learned state."""
    logger.info("Restoring checkpoint: %s", checkpoint_path)
    algo = Algorithm.from_checkpoint(str(checkpoint_path))

    # The training campaign explicitly used RLlib's old API stack. The test loop
    # below intentionally uses Algorithm.compute_actions() (not raw-policy action
    # computation), because this path applies the restored observation filter with
    # update=False before inference.
    return algo


def _one_test_episode(
    algo: Algorithm,
    env: gym.Env,
    episode_seed: int,
) -> Tuple[float, int]:
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
        # Single-agent environment represented as a one-item dict because the
        # old-stack compute_actions path explicitly applies preprocessors and the
        # restored MeanStdFilter with update=False.
        action_dict = algo.compute_actions(
            {"eval_agent": obs},
            explore=EXPLORE,
        )
        action = action_dict["eval_agent"]

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        episode_length += 1

    return total_reward, episode_length


def evaluate_checkpoint(
    method: str,
    env_name: str,
    training_seed: int,
    champion: pd.Series,
    test_seeds: Iterable[int],
) -> pd.DataFrame:
    checkpoint_path = Path(str(champion["checkpoint_path"]))
    algo: Optional[Algorithm] = None
    env: Optional[gym.Env] = None

    try:
        algo = _restore_algorithm(checkpoint_path)
        env = gym.make(env_name)

        rows = []
        for episode_idx, test_seed in enumerate(test_seeds, start=1):
            ret, length = _one_test_episode(algo, env, int(test_seed))
            rows.append(
                {
                    "method": method,
                    "run_name": RUN_NAMES[method],
                    "environment": env_name,
                    "training_seed": training_seed,
                    "champion_agent": champion["agent_id"],
                    "training_terminal_score": champion["training_terminal_score"],
                    "last_training_return": champion["last_training_return"],
                    "checkpoint_path": str(checkpoint_path),
                    "explore": EXPLORE,
                    "test_episode": episode_idx,
                    "test_seed": int(test_seed),
                    "test_return": float(ret),
                    "test_episode_length": int(length),
                }
            )

        return pd.DataFrame(rows)

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
# SUMMARIES
# =============================================================================

def _seed_level_test_summary(episodes: pd.DataFrame) -> pd.DataFrame:
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

    def q25(x: pd.Series) -> float:
        return float(x.quantile(0.25))

    def q75(x: pd.Series) -> float:
        return float(x.quantile(0.75))

    summary = (
        episodes.groupby(keys, dropna=False, as_index=False)
        .agg(
            training_terminal_score=("training_terminal_score", "first"),
            last_training_return=("last_training_return", "first"),
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
    return summary


def _method_environment_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive aggregation over independent TRAINING seeds."""
    if seed_summary.empty:
        return pd.DataFrame()

    def q25(x: pd.Series) -> float:
        return float(x.quantile(0.25))

    def q75(x: pd.Series) -> float:
        return float(x.quantile(0.75))

    out = (
        seed_summary.groupby(["method", "run_name", "environment"], as_index=False)
        .agg(
            n_training_seeds=("training_seed", "nunique"),
            median_test_mean_return=("test_mean_return", "median"),
            p25_test_mean_return=("test_mean_return", q25),
            p75_test_mean_return=("test_mean_return", q75),
            mean_test_mean_return=("test_mean_return", "mean"),
        )
    )
    out["iqr_test_mean_return"] = (
        out["p75_test_mean_return"] - out["p25_test_mean_return"]
    )
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _ensure_project_on_path()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Sanity check: held-out seeds must not overlap training seeds.
    overlap = set(TRAINING_SEEDS) & set(TEST_EPISODE_SEEDS)
    if overlap:
        raise ValueError(f"Test seeds overlap training seeds: {sorted(overlap)}")

    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if SMOKE_TEST:
        methods = SMOKE_METHODS
        envs = [SMOKE_ENV]
        training_seeds = [SMOKE_TRAINING_SEED]
        test_seeds = TEST_EPISODE_SEEDS[:SMOKE_TEST_EPISODES]
    else:
        methods = list(RUN_NAMES.keys())
        envs = ENVIRONMENTS
        training_seeds = TRAINING_SEEDS
        test_seeds = TEST_EPISODE_SEEDS

    # Persist the exact held-out episode seeds used in this campaign.
    pd.DataFrame(
        {
            "test_episode": np.arange(1, len(test_seeds) + 1),
            "test_seed": test_seeds,
        }
    ).to_csv(TEST_OUTPUT_ROOT / "heldout_episode_seeds.csv", index=False)

    protocol = {
        "run_names": RUN_NAMES,
        "environments": envs,
        "training_seeds": training_seeds,
        "terminal_window_steps": TERMINAL_WINDOW_STEPS,
        "minimum_terminal_support_steps": MIN_TERMINAL_SUPPORT_STEPS,
        "minimum_terminal_points": MIN_TERMINAL_POINTS,
        "target_training_steps": TARGET_TRAINING_STEPS,
        "asha_require_full_budget": ASHA_REQUIRE_FULL_BUDGET,
        "n_test_episodes": len(test_seeds),
        "test_episode_seeds": list(map(int, test_seeds)),
        "explore": EXPLORE,
        "terminal_checkpoint_requirement": (
            "Training campaign saved checkpoint_at_end=True for every eligible trial."
        ),
        "champion_selection": (
            "Highest time-weighted mean training return over the final "
            "200k interactions of the final causal branch, requiring at least "
            "100k interactions of terminal support and at least 3 observations; "
            "test data are never used for selection."
        ),
        "statistical_unit": "independent training seed",
    }
    (TEST_OUTPUT_ROOT / "test_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    all_candidates: List[pd.DataFrame] = []
    all_episodes: List[pd.DataFrame] = []
    failures: List[dict] = []

    # Initialize Ray once. Individual Algorithm checkpoints are restored and
    # stopped sequentially, avoiding concurrent GPU/resource contention.
    ray.init(
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
        log_to_driver=False,
        include_dashboard=False,
    )

    started = time.time()

    try:
        total = len(methods) * len(envs) * len(training_seeds)
        done = 0

        for env_name in envs:
            for training_seed in training_seeds:
                for method in methods:
                    done += 1
                    print("\n" + "=" * 80)
                    print(
                        f"[{done}/{total}] {method} | {env_name} | "
                        f"training seed={training_seed}"
                    )
                    print("=" * 80)

                    try:
                        candidates, champion = select_training_champion(
                            method,
                            env_name,
                            training_seed,
                        )
                        all_candidates.append(candidates)

                        print(
                            "Training champion: "
                            f"{champion['agent_id']} | "
                            f"terminal score={champion['training_terminal_score']:.3f} | "
                            f"last return={champion['last_training_return']:.3f}"
                        )

                        episodes = evaluate_checkpoint(
                            method,
                            env_name,
                            training_seed,
                            champion,
                            test_seeds,
                        )
                        all_episodes.append(episodes)

                        print(
                            f"Held-out test ({len(episodes)} episodes, explore=False): "
                            f"mean={episodes['test_return'].mean():.3f} | "
                            f"median={episodes['test_return'].median():.3f} | "
                            f"IQR=[{episodes['test_return'].quantile(0.25):.3f}, "
                            f"{episodes['test_return'].quantile(0.75):.3f}]"
                        )

                    except Exception as exc:
                        logger.exception(
                            "Failed: %s | %s | seed=%s",
                            method,
                            env_name,
                            training_seed,
                        )
                        failures.append(
                            {
                                "method": method,
                                "environment": env_name,
                                "training_seed": training_seed,
                                "error": repr(exc),
                            }
                        )
                        if STRICT:
                            raise

    finally:
        if ray.is_initialized():
            ray.shutdown()

    candidates_df = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame()
    )
    episodes_df = (
        pd.concat(all_episodes, ignore_index=True)
        if all_episodes
        else pd.DataFrame()
    )

    seed_summary_df = _seed_level_test_summary(episodes_df)
    method_env_summary_df = _method_environment_summary(seed_summary_df)

    candidates_df.to_csv(
        TEST_OUTPUT_ROOT / "training_champion_candidates.csv",
        index=False,
    )
    episodes_df.to_csv(
        TEST_OUTPUT_ROOT / "heldout_test_episodes.csv",
        index=False,
    )
    seed_summary_df.to_csv(
        TEST_OUTPUT_ROOT / "heldout_test_seed_summary.csv",
        index=False,
    )
    method_env_summary_df.to_csv(
        TEST_OUTPUT_ROOT / "heldout_test_method_environment_summary.csv",
        index=False,
    )
    pd.DataFrame(failures).to_csv(
        TEST_OUTPUT_ROOT / "heldout_test_failures.csv",
        index=False,
    )

    elapsed = time.time() - started
    print("\n" + "=" * 80)
    print("TEST CAMPAIGN COMPLETE")
    print(f"Output: {TEST_OUTPUT_ROOT}")
    print(f"Elapsed: {elapsed / 60.0:.1f} min")
    print(f"Failures: {len(failures)}")
    print("=" * 80)

    if not method_env_summary_df.empty:
        print("\nDescriptive summary across independent training seeds:")
        print(method_env_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
