#!/usr/bin/env python
"""
Experiment E10: scoped model-mismatch battery.

This experiment complements the parameter-sweep mismatch analysis with a small
set of targeted perturbations. The environment is perturbed while agents
receive nominal action-feature tags, so planning is evaluated with an imperfect
model rather than an identical copy of the simulator.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import Config  # noqa: E402
from src.agents import get_agent  # noqa: E402
from src.env import PianoGymEnv  # noqa: E402
from src.safety import ExternalSafetyGuard  # noqa: E402
from src.utils import compute_metrics, save_results  # noqa: E402


OUTPUT_ROOT = Path("output/data/p1_mismatch")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

PROFILES: Sequence[Dict] = (
    {"name": "balanced", "left_weakness": 0.0},
    {"name": "mild_left_weak", "left_weakness": 0.1},
    {"name": "severe_left_weak", "left_weakness": 0.2},
)

AGENTS: Sequence[str] = ("PianoMPC", "BayesianMAB", "CCB-DF", "LinUCB", "Thompson")

SCENARIOS: Dict[str, str] = {
    "nominal": "Unperturbed PianoGym environment.",
    "observation_noise_x3": "Async, dominance, fatigue, retention, and latent-skill observation noise tripled.",
    "observation_noise_x2_transfer_x1p2": "Observation noise doubled and transfer/interference scale increased to 1.2x.",
    "observation_noise_x3_transfer_x1p2": "Observation noise tripled and transfer/interference scale increased to 1.2x.",
}

GUARD_CONFIG = {"guard_delta": 0.08, "safety_margin": 0.05}
AGENT_GUARD_PARAMS = {"guard_delta": 0.08, "guard_horizon": 1, "guard_safety_margin": 0.05}


def _make_seed(scenario: str, agent_name: str, profile_name: str, run_idx: int) -> int:
    token = f"scoped_mismatch|{scenario}|{agent_name}|{profile_name}|{run_idx}".encode("utf-8")
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


def _scale_observation_noise(config: Config, scale: float) -> None:
    config.sigma_async *= scale
    config.sigma_dom *= scale
    config.fatigue_noise *= scale
    config.retention_noise *= scale
    config.sigma_x *= scale


def _scenario_config(name: str) -> Config:
    config = Config()
    if name == "nominal":
        return config
    if name == "observation_noise_x3":
        _scale_observation_noise(config, 3.0)
    elif name == "observation_noise_x2_transfer_x1p2":
        _scale_observation_noise(config, 2.0)
        config.set_transfer_strength(1.2)
    elif name == "observation_noise_x3_transfer_x1p2":
        _scale_observation_noise(config, 3.0)
        config.set_transfer_strength(1.2)
    else:
        raise ValueError(f"Unknown mismatch scenario: {name}")
    return config


def _agent_configs(nominal_env: PianoGymEnv, nominal_obs: Mapping) -> Dict[str, Dict[str, object]]:
    context_dim = len(nominal_obs["context"]) + len(nominal_obs["action_features"][0])
    return {
        "PianoMPC": {
            "factory": "pianoMPC",
            "kwargs": {
                "reward_weights": nominal_env.cfg.w_reward,
                "reward_norm": nominal_env.cfg.reward_norm,
                "fatigue_limit": nominal_env.cfg.fatigue_threshold,
                "pool_size": 6,
                "rest_recovery": nominal_env.cfg.rest_recovery,
            },
        },
        "BayesianMAB": {
            "factory": "bayesianmab",
            "kwargs": {
                "discount": 0.995,
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "sigma_noise": 0.5,
            },
        },
        "CCB-DF": {
            "factory": "ccb_df",
            "kwargs": {"context_dim": context_dim, "delay_window": 1, "weight_clip": 10.0},
        },
        "LinUCB": {
            "factory": "linucb",
            "kwargs": {
                "context_dim": context_dim,
                "alpha": 2.0,
                "ridge": 1e-3,
                "use_sherman_morrison": True,
            },
        },
        "Thompson": {"factory": "thompson", "kwargs": {"context_dim": context_dim, "v": 0.8}},
    }


def _set_nominal_reported_features(env: PianoGymEnv, nominal_features: np.ndarray) -> None:
    """Expose nominal action tags to agents while true environment dynamics differ."""
    env.action_features = np.asarray(nominal_features, dtype=float).copy()


def run_episode(
    *,
    env: PianoGymEnv,
    agent,
    profile: Dict,
    nominal_features: np.ndarray,
) -> Dict:
    obs = env.reset(profile=profile)
    _set_nominal_reported_features(env, nominal_features)
    obs = dict(obs)
    obs["action_features"] = nominal_features.copy()
    agent.reset()
    guard = ExternalSafetyGuard(env, enabled=True, **GUARD_CONFIG)
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
        "initial_info": {
            "true_skills": env.x.copy(),
            "fatigue": float(env.f),
            "retention": float(env.r),
        },
    }

    while not env.done:
        action = int(agent.select_action(obs))
        executed_action, guard_decision = guard.enforce(action, obs)
        next_obs, reward, done, info = env.step(executed_action)
        guard.annotate_info(info)
        next_obs = dict(next_obs)
        next_obs["action_features"] = nominal_features.copy()

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


def _clean_metrics(metrics: Mapping[str, object]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in metrics.items():
        if key.endswith("_raw"):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            value_float = float(value)
            if not math.isnan(value_float):
                clean[key] = value_float
    return clean


def _macro_average(per_profile: Sequence[Mapping[str, float]], base_keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    profile_count = max(len(per_profile), 1)
    for key in base_keys:
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


def run_experiment(
    *,
    num_runs: int = 10,
    scenarios: Sequence[str] | None = None,
    agents: Sequence[str] = AGENTS,
) -> Dict[str, object]:
    if scenarios is None:
        scenarios = tuple(SCENARIOS.keys())

    nominal_env = PianoGymEnv(config=Config(), seed=123)
    nominal_obs = nominal_env.reset(profile=PROFILES[0])
    nominal_features = np.asarray(nominal_obs["action_features"], dtype=float)
    agent_configs = _agent_configs(nominal_env, nominal_obs)

    unknown_agents = [agent for agent in agents if agent not in agent_configs]
    if unknown_agents:
        raise ValueError(f"Unknown agents: {unknown_agents}")

    base_keys = [
        "time_to_mastery",
        "feasible_rate",
        "fatigue_overload_rate",
        "avg_constraint_violation",
        "avg_fatigue",
        "total_reward_raw",
    ]
    records: List[Dict[str, object]] = []
    profile_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    macro_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    print("=" * 70)
    print("Running scoped model-mismatch experiment")
    print("=" * 70)

    for scenario_name in scenarios:
        profile_results[scenario_name] = {}
        macro_results[scenario_name] = {}
        print(f"\nScenario: {scenario_name}")

        for agent_name in agents:
            profile_results[scenario_name][agent_name] = {}
            per_profile: List[Dict[str, float]] = []
            agent_desc = agent_configs[agent_name]
            factory = str(agent_desc["factory"])
            kwargs_base = copy.deepcopy(agent_desc["kwargs"])
            print(f"  Agent: {agent_name}")

            for profile in PROFILES:
                profile_name = profile["name"]
                trajectories: List[Dict] = []

                for run_idx in range(num_runs):
                    seed = _make_seed(scenario_name, agent_name, profile_name, run_idx)
                    env = PianoGymEnv(config=_scenario_config(scenario_name), seed=seed)
                    agent = get_agent(
                        factory,
                        num_actions=env.cfg.num_actions,
                        seed=seed,
                        **AGENT_GUARD_PARAMS,
                        **copy.deepcopy(kwargs_base),
                    )
                    trajectory = run_episode(
                        env=env,
                        agent=agent,
                        profile=profile,
                        nominal_features=nominal_features,
                    )
                    trajectories.append(trajectory)

                metrics = compute_metrics(
                    trajectories,
                    mastery_window=env.cfg.mastery_window,
                    optimal_return=0.0,
                )
                clean = _clean_metrics(metrics)
                per_profile.append(clean)
                profile_results[scenario_name][agent_name][profile_name] = clean
                record = {
                    "scenario": scenario_name,
                    "scenario_description": SCENARIOS[scenario_name],
                    "agent": agent_name,
                    "profile": profile_name,
                    "num_runs": num_runs,
                }
                record.update(clean)
                records.append(record)
                print(
                    f"    {profile_name:17s} "
                    f"TTM={clean.get('time_to_mastery_mean', float('nan')):.1f} "
                    f"Feas={clean.get('feasible_rate_mean', float('nan')):.3f}"
                )

            macro_results[scenario_name][agent_name] = _macro_average(per_profile, base_keys)

    payload = {
        "meta": {
            "num_runs": num_runs,
            "profiles": [profile["name"] for profile in PROFILES],
            "agents": list(agents),
            "scenarios": {name: SCENARIOS[name] for name in scenarios},
            "agent_features": "Agents receive nominal action_features while environment dynamics use scenario-specific configs.",
        },
        "metrics": profile_results,
        "macro": macro_results,
        "records": records,
    }

    json_path = OUTPUT_ROOT / "scoped_mismatch.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = OUTPUT_ROOT / "scoped_mismatch_summary.csv"
    if records:
        fieldnames = sorted({key for row in records for key in row.keys()})
        leading = ["scenario", "scenario_description", "agent", "profile", "num_runs"]
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
    save_results(formatted, save_path=str(OUTPUT_ROOT / "scoped_mismatch_results.txt"))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scoped model-mismatch experiment")
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per scenario/agent/profile")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS.keys()),
        default=None,
        help="Model-mismatch scenarios to run.",
    )
    parser.add_argument("--agents", nargs="+", default=list(AGENTS), help="Agents to evaluate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(num_runs=args.num_runs, scenarios=args.scenarios, agents=args.agents)
