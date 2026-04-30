#!/usr/bin/env python
"""
Experiment E3: Safety mechanism ablations.

Compare layered safety configurations (LinUCB variants, PianoMPC penalty toggle, Safe-AC
reference) to highlight safety-performance trade-offs.
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
except Exception:  # pragma: no cover
    stats = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import PianoGymEnv
from src.agents import get_agent
from src.utils import compute_metrics, save_results
from src.safety import ExternalSafetyGuard


def _t_critical(df: int) -> float:
    if df <= 0:
        return 0.0
    if stats is None:
        return 1.96
    try:
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.96


def run_episode(
    env: PianoGymEnv,
    agent,
    profile: Dict | None = None,
    seed: int | None = None,
    *,
    guard_enabled: bool = True,
    guard_kwargs: Dict | None = None,
):
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


def macro_average(per_profile: Sequence[Dict], base_keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    P = max(len(per_profile), 1)
    for key in base_keys:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        col = [item.get(mean_key) for item in per_profile if mean_key in item]
        if not col:
            continue
        arr = np.asarray(col, dtype=float)
        out[mean_key] = float(np.nanmean(arr))
        if P > 1:
            t_crit = _t_critical(P - 1)
            sem = np.nanstd(arr, ddof=1) / np.sqrt(P)
            out[std_key] = float(sem * t_crit)
        else:
            out[std_key] = 0.0
    return out


def run_experiment(
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
    context_dim = len(probe_obs["context"]) + len(probe_obs["action_features"][0])
    state_dim = len(probe_obs["context"])
    num_actions = probe_env.cfg.num_actions

    guard_kwargs = dict(
        guard_delta=0.08,
        guard_horizon=1,
        guard_safety_margin=0.05,
    )
    external_guard_config = {"guard_delta": 0.08, "safety_margin": 0.05}

    configurations = [
        {
            "name": "LinUCB-both",
            "agent_key": "linucb",
            "label": "LinUCB (guard+soft)",
            "params": {
                "context_dim": context_dim,
                "alpha": 2.0,
                "ridge": 1e-3,
                "lambda_fatigue": 2.0,
                "enable_guard": True,
            },
        },
        {
            "name": "LinUCB-guard",
            "agent_key": "linucb",
            "label": "LinUCB (guard only)",
            "params": {
                "context_dim": context_dim,
                "alpha": 2.0,
                "ridge": 1e-3,
                "lambda_fatigue": 0.0,
                "enable_guard": True,
            },
        },
        {
            "name": "LinUCB-soft",
            "agent_key": "linucb",
            "label": "LinUCB (soft only)",
            "params": {
                "context_dim": context_dim,
                "alpha": 2.0,
                "ridge": 1e-3,
                "lambda_fatigue": 2.0,
                "enable_guard": False,
            },
        },
        {
            "name": "LinUCB-none",
            "agent_key": "linucb",
            "label": "LinUCB (no safety)",
            "params": {
                "context_dim": context_dim,
                "alpha": 2.0,
                "ridge": 1e-3,
                "lambda_fatigue": 0.0,
                "enable_guard": False,
            },
        },
        {
            "name": "PianoMPC-default",
            "agent_key": "pianoMPC",
            "label": "PianoMPC (default)",
            "params": {
                "pool_size": 6,
                "reward_weights": probe_env.cfg.w_reward,
                "reward_norm": probe_env.cfg.reward_norm,
                "fatigue_limit": probe_env.cfg.fatigue_threshold,
                "rest_recovery": probe_env.cfg.rest_recovery,
            },
        },
        {
            "name": "PianoMPC-no-soft",
            "agent_key": "pianoMPC",
            "label": "PianoMPC (no soft)",
            "params": {
                "pool_size": 6,
                "reward_weights": probe_env.cfg.w_reward,
                "reward_norm": probe_env.cfg.reward_norm,
                "fatigue_limit": probe_env.cfg.fatigue_threshold,
                "rest_recovery": probe_env.cfg.rest_recovery,
                "fatigue_penalty": 0.0,
            },
        },
        {
            "name": "Safe-AC",
            "agent_key": "safe_ac",
            "label": "Safe-AC",
            "params": {
                "state_dim": state_dim,
                "hidden_dim": 32,
                "cost_limit": probe_env.cfg.fatigue_threshold,
            },
        },
    ]

    profile_results: Dict[str, Dict[str, Dict[str, float]]] = {
        profile["name"]: {} for profile in profiles
    }
    macro_results: Dict[str, Dict[str, float]] = {}

    base_keys = [
        "time_to_mastery",
        "feasible_rate",
        "fatigue_overload_rate",
        "avg_constraint_violation",
        "avg_fatigue",
        "total_reward_raw",
    ]

    print("=" * 60)
    print("Running Safety Ablation Experiment")
    print("=" * 60)

    for config in configurations:
        name = config["name"]
        agent_key = config["agent_key"]
        params = dict(config["params"])
        print(f"\n>> {config['label']}")

        per_profile_metrics: List[Dict[str, float]] = []

        for profile in profiles:
            profile_name = profile["name"]
            trajectories = []

            print(f"  Profile: {profile_name}")
            for run in range(num_runs):
                # Match the main comparison seed schedule so the default PianoMPC row is directly comparable.
                seed = run * 100
                env = PianoGymEnv(seed=seed)

                agent_params = dict(params)
                if agent_key == "linucb":
                    agent_params.setdefault("use_sherman_morrison", True)
                agent = get_agent(
                    agent_key,
                    num_actions=num_actions,
                    seed=seed,
                    **agent_params,
                    **guard_kwargs,
                )

                traj = run_episode(
                    env,
                    agent,
                    profile=profile,
                    seed=seed,
                    guard_enabled=True,
                    guard_kwargs=external_guard_config,
                )
                trajectories.append(traj)

            metrics = compute_metrics(
                trajectories,
                mastery_window=env.cfg.mastery_window,
                optimal_return=0.0,
            )
            per_profile_metrics.append(metrics)

            t_crit = _t_critical(num_runs - 1)
            ttm_ci = (metrics["time_to_mastery_std"] / np.sqrt(num_runs)) * t_crit
            feas_ci = (metrics["feasible_rate_std"] / np.sqrt(num_runs)) * t_crit
            print(
                f"    TTM: {metrics['time_to_mastery_mean']:.1f} ± {ttm_ci:.1f}, "
                f"Feas.: {metrics['feasible_rate_mean']:.3f} ± {feas_ci:.3f}"
            )

            clean_metrics = {
                key: float(val)
                for key, val in metrics.items()
                if not key.endswith("_raw") and isinstance(val, (int, float, np.floating, np.integer))
            }
            profile_results[profile_name][name] = clean_metrics

        macro_results[name] = macro_average(per_profile_metrics, base_keys)

    data_dir = Path("output/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "num_runs": num_runs,
            "profiles": [p["name"] for p in profiles],
            "configurations": configurations,
        },
        "metrics": profile_results,
        "macro": macro_results,
    }

    with (data_dir / "safety_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    formatted = {
        cfg["label"]: {
            key: value
            for key, value in macro_results[cfg["name"]].items()
            if not key.endswith("_raw")
        }
        for cfg in configurations
    }
    save_results(formatted, save_path="output/data/safety_ablation_results.txt")

    print("\nSaved: output/data/safety_ablation.json")
    print("Saved: output/data/safety_ablation_results.txt")
    return profile_results


def parse_args():
    parser = argparse.ArgumentParser(description="Safety ablation experiment")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per configuration/profile (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(num_runs=args.num_runs)
