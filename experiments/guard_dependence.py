#!/usr/bin/env python
"""
Experiment R1: Environment-side guard dependence.

This script tests whether the main PianoMPC advantage depends on the shared
environment-side execution guard. It runs the same agents under multiple wrapper
settings and reports both outcome metrics and guard-intervention diagnostics:
proposed-vs-executed action changes, proposed one-step cap violations, and
proposed relaxed-threshold violations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import get_agent  # noqa: E402
from src.env import PianoGymEnv  # noqa: E402
from src.metrics import compute_guard_intervention_stats  # noqa: E402
from src.safety import ExternalSafetyGuard  # noqa: E402
from src.utils import compute_metrics, save_results  # noqa: E402


DEFAULT_PROFILES: Sequence[Dict] = (
    {"name": "balanced", "left_weakness": 0.0},
    {"name": "mild_left_weak", "left_weakness": 0.1},
    {"name": "severe_left_weak", "left_weakness": 0.2},
)

DEFAULT_SCENARIOS: Dict[str, Dict[str, object]] = {
    "shared_guard": {
        "enabled": True,
        "guard_kwargs": {"guard_delta": 0.08, "safety_margin": 0.05},
        "description": "Current shared environment-side wrapper.",
    },
    "weaker_wrapper": {
        "enabled": True,
        "guard_kwargs": {"guard_delta": 0.15, "safety_margin": 0.00},
        "description": "Less conservative wrapper with no step margin and a larger relaxed band.",
    },
    "no_wrapper": {
        "enabled": False,
        "guard_kwargs": {"guard_delta": 0.08, "safety_margin": 0.05},
        "description": "No environment-side action projection; diagnostics are still logged.",
    },
}

OUTPUT_ROOT = Path("output/data/p1_guard")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _make_seed(scenario: str, agent_name: str, profile_name: str, run_idx: int) -> int:
    token = f"{scenario}|{agent_name}|{profile_name}|{run_idx}".encode("utf-8")
    return zlib.crc32(token) & 0xFFFFFFFF


def _t_critical(df: int) -> float:
    if df <= 0:
        return 0.0
    if stats is None:
        return 1.96
    try:
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.96


def _agent_configs(probe_env: PianoGymEnv, probe_obs: Mapping) -> Dict[str, Dict[str, object]]:
    context_dim = len(probe_obs["context"]) + len(probe_obs["action_features"][0])
    state_dim = len(probe_obs["context"])
    return {
        "LinUCB": {
            "name": "linucb",
            "context_dim": context_dim,
            "alpha": 2.0,
            "ridge": 1e-3,
            "use_sherman_morrison": True,
        },
        "Thompson": {"name": "thompson", "context_dim": context_dim, "v": 0.8},
        "PianoMPC": {
            "name": "pianoMPC",
            "reward_weights": probe_env.cfg.w_reward,
            "reward_norm": probe_env.cfg.reward_norm,
            "fatigue_limit": probe_env.cfg.fatigue_threshold,
            "pool_size": 6,
            "rest_recovery": probe_env.cfg.rest_recovery,
        },
        "DQN": {"name": "dqn", "state_dim": state_dim, "hidden_dim": 64, "epsilon": 0.2},
        "BayesianMAB": {
            "name": "bayesianmab",
            "discount": 0.995,
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "sigma_noise": 0.5,
        },
        "CCB-DF": {
            "name": "ccb_df",
            "context_dim": context_dim,
            "delay_window": 1,
            "weight_clip": 10.0,
        },
        "Safe-AC": {
            "name": "safe_ac",
            "state_dim": state_dim,
            "hidden_dim": 32,
            "cost_limit": probe_env.cfg.fatigue_threshold,
        },
        "AutoCurriculum": {
            "name": "autocurriculum",
            "advance_threshold": 0.7,
            "regress_threshold": 0.3,
        },
    }


def run_episode(
    env: PianoGymEnv,
    agent,
    profile: Dict,
    *,
    guard_enabled: bool,
    guard_kwargs: Dict[str, float],
) -> Dict:
    obs = env.reset(profile=profile)
    agent.reset()
    guard = ExternalSafetyGuard(env, enabled=guard_enabled, **guard_kwargs)
    guard.reset()

    trajectory = {
        "rewards": [],
        "raw_rewards": [],
        "obs_history": [obs],
        "info_history": [],
        "actions": [],
        "action_probs": [],
        "proposed_actions": [],
        "guard_history": [],
        "steps": 0,
        "done": False,
        "initial_info": {
            "true_skills": env.x.copy(),
            "fatigue": float(env.f),
            "retention": float(env.r),
        },
    }

    while not env.done:
        action = int(agent.select_action(obs))
        prob = agent.action_prob(obs, action) if hasattr(agent, "action_prob") else None
        executed_action, guard_decision = guard.enforce(action, obs)

        next_obs, reward, done, info = env.step(executed_action)
        guard.annotate_info(info)

        next_obs_with_done = dict(next_obs)
        next_obs_with_done["done"] = done
        next_obs_with_done["_info"] = info
        if hasattr(agent, "update"):
            agent.update(obs, executed_action, reward, next_obs_with_done)

        trajectory["rewards"].append(float(reward))
        trajectory["raw_rewards"].append(float(info.get("raw_reward", reward)))
        trajectory["obs_history"].append(next_obs)
        trajectory["info_history"].append(info)
        trajectory["actions"].append(int(executed_action))
        trajectory["proposed_actions"].append(int(action))
        trajectory["action_probs"].append(prob)
        trajectory["guard_history"].append(guard_decision.as_dict())
        trajectory["steps"] += 1

        obs = next_obs

    trajectory["done"] = bool(env.done)
    mastery_step = next(
        (
            idx + 1
            for idx, info in enumerate(trajectory["info_history"])
            if info.get("mastery_count", 0) >= env.cfg.mastery_window
        ),
        0,
    )
    trajectory["mastery_step"] = int(mastery_step)
    return trajectory


def _add_guard_stats(metrics: Dict[str, object], trajectories: Sequence[Mapping]) -> None:
    if not trajectories:
        return
    stats_per_traj = [
        compute_guard_intervention_stats(traj.get("guard_history", [])).__dict__
        for traj in trajectories
    ]
    keys = sorted(stats_per_traj[0].keys())
    for key in keys:
        vals = np.asarray([item[key] for item in stats_per_traj], dtype=float)
        metrics[f"{key}_mean"] = float(np.nanmean(vals))
        metrics[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0


def _macro_average(per_profile: Sequence[Mapping[str, object]], keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    profile_count = max(len(per_profile), 1)
    for key in keys:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        vals = [float(item[mean_key]) for item in per_profile if mean_key in item]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[mean_key] = float(np.nanmean(arr))
        if profile_count > 1:
            sem = np.nanstd(arr, ddof=1) / math.sqrt(profile_count)
            out[std_key] = float(sem * _t_critical(profile_count - 1))
        else:
            out[std_key] = 0.0
    return out


def _clean_metrics(metrics: Mapping[str, object]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in metrics.items():
        if key.endswith("_raw"):
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            value_float = float(value)
            if not math.isnan(value_float):
                clean[key] = value_float
    return clean


def run_experiment(
    *,
    num_runs: int = 10,
    agents: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    profiles: Sequence[Dict] = DEFAULT_PROFILES,
) -> Dict[str, object]:
    probe_env = PianoGymEnv(seed=123)
    probe_obs = probe_env.reset()
    configs = _agent_configs(probe_env, probe_obs)
    if agents is None:
        agents = tuple(configs.keys())
    if scenarios is None:
        scenarios = tuple(DEFAULT_SCENARIOS.keys())

    unknown_agents = [agent for agent in agents if agent not in configs]
    if unknown_agents:
        raise ValueError(f"Unknown agents: {unknown_agents}")
    unknown_scenarios = [scenario for scenario in scenarios if scenario not in DEFAULT_SCENARIOS]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {unknown_scenarios}")

    profile_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    macro_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    records: List[Dict[str, object]] = []

    base_keys = [
        "time_to_mastery",
        "feasible_rate",
        "fatigue_overload_rate",
        "avg_constraint_violation",
        "avg_fatigue",
        "total_reward_raw",
        "guard_replacement_rate",
        "action_changed_rate",
        "rest_fallback_rate",
        "relaxed_execution_rate",
        "proposed_step_cap_violation_rate",
        "proposed_relaxed_violation_rate",
        "executed_step_cap_violation_rate",
    ]

    print("=" * 70)
    print("Running guard-dependence experiment")
    print("=" * 70)

    for scenario_name in scenarios:
        scenario = DEFAULT_SCENARIOS[scenario_name]
        guard_enabled = bool(scenario["enabled"])
        guard_kwargs = dict(scenario["guard_kwargs"])  # type: ignore[arg-type]
        profile_results[scenario_name] = {}
        macro_results[scenario_name] = {}
        print(f"\nScenario: {scenario_name} | enabled={guard_enabled} | {guard_kwargs}")

        for agent_name in agents:
            per_profile_metrics: List[Dict[str, float]] = []
            profile_results[scenario_name][agent_name] = {}
            print(f"  Agent: {agent_name}")

            for profile in profiles:
                profile_name = profile["name"]
                trajectories: List[Dict] = []
                config = dict(configs[agent_name])
                factory_name = str(config.pop("name"))

                for run_idx in range(num_runs):
                    seed = _make_seed(scenario_name, agent_name, profile_name, run_idx)
                    env = PianoGymEnv(seed=seed)
                    agent = get_agent(
                        factory_name,
                        num_actions=env.cfg.num_actions,
                        seed=seed,
                        guard_delta=0.08,
                        guard_horizon=1,
                        guard_safety_margin=0.05,
                        **config,
                    )
                    traj = run_episode(
                        env,
                        agent,
                        profile,
                        guard_enabled=guard_enabled,
                        guard_kwargs=guard_kwargs,
                    )
                    trajectories.append(traj)

                metrics = compute_metrics(
                    trajectories,
                    mastery_window=env.cfg.mastery_window,
                    optimal_return=0.0,
                )
                _add_guard_stats(metrics, trajectories)
                clean = _clean_metrics(metrics)
                per_profile_metrics.append(clean)
                profile_results[scenario_name][agent_name][profile_name] = clean

                row = {
                    "scenario": scenario_name,
                    "agent": agent_name,
                    "profile": profile_name,
                    "num_runs": num_runs,
                    "guard_enabled": guard_enabled,
                    "guard_delta": guard_kwargs.get("guard_delta"),
                    "safety_margin": guard_kwargs.get("safety_margin"),
                }
                row.update(clean)
                records.append(row)
                print(
                    f"    {profile_name:17s} "
                    f"TTM={clean.get('time_to_mastery_mean', float('nan')):.1f} "
                    f"Feas={clean.get('feasible_rate_mean', float('nan')):.3f} "
                    f"Repl={clean.get('guard_replacement_rate_mean', float('nan')):.3f} "
                    f"PropStepViol={clean.get('proposed_step_cap_violation_rate_mean', float('nan')):.3f}"
                )

            macro_results[scenario_name][agent_name] = _macro_average(per_profile_metrics, base_keys)

    payload = {
        "meta": {
            "num_runs": num_runs,
            "agents": list(agents),
            "profiles": [profile["name"] for profile in profiles],
            "scenarios": {
                name: {
                    "enabled": bool(DEFAULT_SCENARIOS[name]["enabled"]),
                    "guard_kwargs": DEFAULT_SCENARIOS[name]["guard_kwargs"],
                    "description": DEFAULT_SCENARIOS[name]["description"],
                }
                for name in scenarios
            },
        },
        "metrics": profile_results,
        "macro": macro_results,
        "records": records,
    }

    json_path = OUTPUT_ROOT / "guard_dependence.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = OUTPUT_ROOT / "guard_dependence_summary.csv"
    if records:
        fieldnames = sorted({key for record in records for key in record.keys()})
        leading = ["scenario", "agent", "profile", "num_runs", "guard_enabled", "guard_delta", "safety_margin"]
        fieldnames = leading + [key for key in fieldnames if key not in leading]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    formatted = {
        f"{scenario}/{agent}": {
            key: value
            for key, value in macro_results[scenario][agent].items()
            if not key.endswith("_raw")
        }
        for scenario in scenarios
        for agent in agents
    }
    save_results(formatted, save_path=str(OUTPUT_ROOT / "guard_dependence_results.txt"))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guard dependence experiment")
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per scenario/agent/profile")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Agent names to run. Defaults to all comparison agents.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=tuple(DEFAULT_SCENARIOS.keys()),
        help="Guard scenarios to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(num_runs=args.num_runs, agents=args.agents, scenarios=args.scenarios)
