#!/usr/bin/env python
"""
Experiment E5: Suite robustness

Evaluate top-performing agents from E1 across the 3×3 PianoGym task suite
(category × transfer strength) and report time-to-mastery / feasible rate
statistics for each task.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import PianoGymEnv  # noqa: E402
from src.agents import get_agent  # noqa: E402
from src.utils import compute_metrics  # noqa: E402
from src.suite import suite_specs, TaskSpec, build_task_config  # noqa: E402
from src.safety import ExternalSafetyGuard  # noqa: E402


TOP_AGENTS: Tuple[str, ...] = ("PianoMPC", "CCB-DF", "BayesianMAB", "Thompson")
DEFAULT_RUNS = 10

OUTPUT_ROOT = Path("output/data/p1_suite")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUTPUT_ROOT / "e5_suite_summary.csv"
JSON_PATH = OUTPUT_ROOT / "e5_suite_results.json"


def _make_seed(agent_name: str, spec_name: str, run_idx: int) -> int:
    token = f"{agent_name}|{spec_name}|{run_idx}".encode("utf-8")
    return zlib.crc32(token) & 0xFFFFFFFF


def _initial_agent_kwargs(agent_name: str, env: PianoGymEnv, context_dim: int, state_dim: int) -> Dict:
    if agent_name == "PianoMPC":
        return {
            "factory": "pianoMPC",
            "kwargs": {
                "reward_weights": env.cfg.w_reward,
                "reward_norm": env.cfg.reward_norm,
                "fatigue_limit": env.fatigue_threshold,
                "pool_size": 6,
                "rest_recovery": env.cfg.rest_recovery,
            },
        }
    if agent_name == "CCB-DF":
        return {
            "factory": "ccb_df",
            "kwargs": {
                "context_dim": context_dim,
                "delay_window": 1,
                "weight_clip": 10.0,
            },
        }
    if agent_name == "BayesianMAB":
        return {
            "factory": "bayesianmab",
            "kwargs": {
                "discount": 0.995,
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "sigma_noise": 0.5,
            },
        }
    if agent_name == "Thompson":
        return {
            "factory": "thompson",
            "kwargs": {
                "context_dim": context_dim,
                "v": 0.8,
            },
        }
    raise ValueError(f"Unsupported agent: {agent_name}")


def _simulate_episode(
    env: PianoGymEnv,
    agent,
    initial_obs,
    *,
    guard_enabled: bool = True,
    guard_kwargs: Dict | None = None,
) -> Dict:
    agent.reset()
    obs = initial_obs
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

    def _action_prob(current_obs, action_idx):
        if hasattr(agent, "action_prob"):
            try:
                return float(agent.action_prob(current_obs, action_idx))
            except Exception:
                return None
        if hasattr(agent, "get_action_probabilities"):
            try:
                probs = agent.get_action_probabilities(current_obs)
                if probs is None:
                    return None
                if isinstance(probs, Iterable):
                    arr = list(probs)
                    if 0 <= action_idx < len(arr):
                        return float(arr[action_idx])
            except Exception:
                return None
        return None

    while not env.done:
        action = int(agent.select_action(obs))
        prob = _action_prob(obs, action)
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
        trajectory["actions"].append(executed_action)
        trajectory["proposed_actions"].append(action)
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
    trajectory["mastery_step"] = mastery_step
    return trajectory


def run_suite(num_runs: int, agent_list: Iterable[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary_records: List[Dict[str, object]] = []
    metrics_per_agent: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    all_specs = suite_specs()
    agent_list = list(agent_list)
    print(f"E5 Suite robustness: evaluating agents {agent_list} over {len(all_specs)} tasks, {num_runs} runs each.")
    guard_config = {"guard_delta": 0.08, "safety_margin": 0.05}

    for spec in all_specs:
        print(f"\n=== Task: {spec.name} | category={spec.category}, transfer={spec.transfer_strength} ===")
        for agent_name in agent_list:
            trajectories = []
            for run_idx in range(num_runs):
                seed = _make_seed(agent_name, spec.name, run_idx)
                config = build_task_config(spec)
                env = PianoGymEnv(config=config, seed=seed)
                initial_obs = env.reset(
                    profile=spec.profile,
                    transfer_strength=spec.transfer_strength,
                    enforce_constraint=spec.enforce_constraint,
                    enable_nonstationary=spec.enable_nonstationary,
                )

                context_dim = len(initial_obs["context"]) + len(initial_obs["action_features"][0])
                state_dim = len(initial_obs["context"])

                agent_desc = _initial_agent_kwargs(agent_name, env, context_dim, state_dim)
                factory_name = agent_desc["factory"]
                agent_kwargs = dict(agent_desc["kwargs"])
                if factory_name in {"linucb", "thompson", "ccb_df"}:
                    agent_kwargs["context_dim"] = context_dim
                if factory_name in {"dqn", "safe_ac"}:
                    agent_kwargs["state_dim"] = state_dim

                agent = get_agent(
                    factory_name,
                    num_actions=env.cfg.num_actions,
                    seed=seed,
                    guard_delta=0.08,
                    guard_horizon=1,
                    guard_safety_margin=0.05,
                    **agent_kwargs,
                )
                trajectory = _simulate_episode(
                    env,
                    agent,
                    initial_obs,
                    guard_enabled=True,
                    guard_kwargs=guard_config,
                )
                trajectories.append(trajectory)

            metrics = compute_metrics(trajectories, mastery_window=config.mastery_window, optimal_return=0.0)
            metrics_filtered = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, bool)) and not math.isnan(float(value))
            }

            metrics_per_agent[agent_name][spec.name] = metrics_filtered
            record = {
                "agent": agent_name,
                "task_name": spec.name,
                "category": spec.category,
                "transfer_strength": spec.transfer_strength,
                "enforce_constraint": bool(spec.enforce_constraint),
                "enable_nonstationary": bool(spec.enable_nonstationary),
                "num_runs": num_runs,
            }
            record.update(metrics_filtered)
            summary_records.append(record)

            ttm = metrics_filtered.get("time_to_mastery_mean")
            feas = metrics_filtered.get("feasible_rate_mean")
            print(
                f"  {agent_name:10s} TTM={ttm:.1f} | Feas={feas:.3f}"
                if ttm is not None and feas is not None
                else f"  {agent_name:10s} (metrics missing)"
            )

    _write_summary(summary_records)
    _write_json(agent_list, all_specs, summary_records, metrics_per_agent, num_runs)
    return metrics_per_agent


def _write_summary(records: List[Dict[str, object]]) -> None:
    if not records:
        return
    numeric_keys = sorted(
        {key for rec in records for key in rec.keys() if key.endswith("_mean") or key.endswith("_std")}
    )
    fieldnames = [
        "agent",
        "task_name",
        "category",
        "transfer_strength",
        "enforce_constraint",
        "enable_nonstationary",
        "num_runs",
    ] + numeric_keys

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {key: rec.get(key, "") for key in fieldnames}
            writer.writerow(row)


def _write_json(
    agents: List[str],
    specs: List[TaskSpec],
    records: List[Dict[str, object]],
    metrics_per_agent: Dict[str, Dict[str, Dict[str, float]]],
    num_runs: int,
) -> None:
    payload = {
        "meta": {
            "num_runs": num_runs,
            "agents": agents,
            "tasks": [asdict(spec) for spec in specs],
        },
        "records": records,
        "metrics": metrics_per_agent,
    }
    with JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run E5 suite robustness experiment.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of runs per agent/task (default: 10).")
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        choices=TOP_AGENTS,
        help="Optional subset of agents to evaluate (default: top agents).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_agents = args.agents if args.agents else TOP_AGENTS
    run_suite(num_runs=args.runs, agent_list=selected_agents)


if __name__ == "__main__":
    main()
