#!/usr/bin/env python
"""
Experiment E2: PianoMPC planning horizon sweep.

Evaluate PianoMPC with different planning horizons to highlight the benefit
of lookahead over greedy (horizon=1) control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover - SciPy may be unavailable
    stats = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import PianoGymEnv
from src.agents import get_agent
from src.utils import compute_metrics, save_results
from src.safety import ExternalSafetyGuard


def run_episode(
    env: PianoGymEnv,
    agent,
    profile: Dict | None = None,
    seed: int | None = None,
    *,
    guard_enabled: bool = True,
    guard_kwargs: Dict | None = None,
):
    """Single episode rollout (reused from compare experiment)."""
    obs = env.reset(profile=profile)
    agent.reset()
    guard = ExternalSafetyGuard(
        env,
        enabled=guard_enabled,
        **(guard_kwargs or {}),
    )
    guard.reset()

    trajectory = {
        "rewards": [],
        "raw_rewards": [],
        "obs_history": [obs],
        "info_history": [],
        "actions": [],
        "proposed_actions": [],
        "guard_history": [],
        "steps": 0,
        "done": False,
    }

    while not env.done:
        action = agent.select_action(obs)
        executed_action, guard_decision = guard.enforce(int(action), obs)
        next_obs, reward, done, info = env.step(executed_action)
        guard.annotate_info(info)
        next_obs_with_done = dict(next_obs)
        next_obs_with_done["done"] = done
        next_obs_with_done["_info"] = info
        agent.update(obs, executed_action, reward, next_obs_with_done)

        trajectory["rewards"].append(reward)
        trajectory["raw_rewards"].append(info.get("raw_reward", reward))
        trajectory["obs_history"].append(next_obs)
        trajectory["info_history"].append(info)
        trajectory["actions"].append(executed_action)
        trajectory["proposed_actions"].append(action)
        trajectory["guard_history"].append(guard_decision.as_dict())
        trajectory["steps"] += 1

        obs = next_obs

    trajectory["done"] = env.done
    return trajectory


def _t_critical(df: int) -> float:
    if df <= 0:
        return 0.0
    if stats is None:
        return 1.96
    try:
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.96


def macro_average(per_profile: Sequence[Dict], base_keys: Sequence[str]) -> Dict[str, float]:
    """Macro average metrics across profiles (mean of means)."""
    out: Dict[str, float] = {}
    P = max(len(per_profile), 1)
    for key in base_keys:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        col = [p.get(mean_key) for p in per_profile if mean_key in p]
        if not col:
            continue
        arr = np.asarray(col, dtype=float)
        out[mean_key] = float(np.nanmean(arr))

        if P > 1:
            t_critical = _t_critical(P - 1)
            sem = np.nanstd(arr, ddof=1) / np.sqrt(P)
            out[std_key] = float(sem * t_critical)
        else:
            out[std_key] = 0.0
    return out


def run_experiment(
    horizons: Sequence[int] = (1, 3, 5, 10),
    num_runs: int = 10,
    profiles: Sequence[Dict] | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if profiles is None:
        profiles = [
            {"name": "balanced", "left_weakness": 0.0},
            {"name": "mild_left_weak", "left_weakness": 0.1},
            {"name": "severe_left_weak", "left_weakness": 0.2},
        ]

    probe_env = PianoGymEnv(seed=123)
    probe_obs = probe_env.reset()
    num_actions = probe_env.cfg.num_actions

    results: Dict[str, Dict[str, Dict[str, float]]] = {
        profile["name"]: {} for profile in profiles
    }
    macro_results: Dict[str, Dict[str, float]] = {}

    base_keys = [
        "time_to_mastery",
        "feasible_rate",
        "avg_fatigue",
        "total_reward_raw",
    ]

    print("=" * 60)
    print("Running PianoMPC Horizon Sweep Experiment")
    print("=" * 60)
    guard_config = {"guard_delta": 0.08, "safety_margin": 0.05}

    for horizon in horizons:
        print(f"\n>> Horizon H = {horizon}")
        per_profile_metrics: List[Dict[str, float]] = []

        for profile in profiles:
            profile_name = profile["name"]
            trajectories = []

            print(f"  Profile: {profile_name}")
            for run in range(num_runs):
                seed = run * 100
                env = PianoGymEnv(seed=seed)
                agent = get_agent(
                    "pianoMPC",
                    num_actions=num_actions,
                    seed=seed,
                    horizon=int(horizon),
                    pool_size=6,
                    reward_weights=probe_env.cfg.w_reward,
                    reward_norm=probe_env.cfg.reward_norm,
                    fatigue_limit=probe_env.cfg.fatigue_threshold,
                    rest_recovery=probe_env.cfg.rest_recovery,
                    guard_delta=0.08,
                    guard_horizon=1,
                    guard_safety_margin=0.05,
                )

                traj = run_episode(
                    env,
                    agent,
                    profile=profile,
                    seed=seed,
                    guard_enabled=True,
                    guard_kwargs=guard_config,
                )
                trajectories.append(traj)

            metrics = compute_metrics(
                trajectories,
                mastery_window=env.cfg.mastery_window,
                optimal_return=0.0,
            )

            per_profile_metrics.append(metrics)

            t_critical = _t_critical(num_runs - 1)
            ttm_ci = (metrics["time_to_mastery_std"] / np.sqrt(num_runs)) * t_critical
            feas_ci = (metrics["feasible_rate_std"] / np.sqrt(num_runs)) * t_critical
            print(
                f"    TTM: {metrics['time_to_mastery_mean']:.1f} ± {ttm_ci:.1f}, "
                f"Feas.: {metrics['feasible_rate_mean']:.3f} ± {feas_ci:.3f}"
            )

            clean_metrics = {
                key: float(val)
                for key, val in metrics.items()
                if not key.endswith("_raw") and isinstance(val, (int, float, np.floating, np.integer))
            }
            results[profile_name][str(horizon)] = clean_metrics

        macro_results[str(horizon)] = macro_average(per_profile_metrics, base_keys)

    data_dir = Path("output/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "num_runs": num_runs,
            "horizons": [int(h) for h in horizons],
            "profiles": [p["name"] for p in profiles],
        },
        "metrics": results,
        "macro": macro_results,
    }

    with (data_dir / "pianoMPC_horizon.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    formatted_macro = {
        f"H={horizon}": {
            key: value
            for key, value in metrics.items()
            if not key.endswith("_raw")
        }
        for horizon, metrics in macro_results.items()
    }
    save_results(formatted_macro, save_path="output/data/pianoMPC_horizon_results.txt")

    print("\nSaved: output/data/pianoMPC_horizon.json")
    print("Saved: output/data/pianoMPC_horizon_results.txt")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="PianoMPC horizon sweep experiment")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per horizon/profile (default: 10)")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Planning horizons to evaluate (default: 1 3 5 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(horizons=args.horizons, num_runs=args.num_runs)
