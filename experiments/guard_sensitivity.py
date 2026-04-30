#!/usr/bin/env python
"""
Guard sensitivity ablation for the environment-side safety interface.

This experiment keeps the existing relaxed guard band fixed at delta_guard=0.08
and varies only the conservative one-step margin around the fatigue threshold.
It is intended as a sensitivity check, not as a replacement for the default
shared wrapper configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.guard_dependence import (  # noqa: E402
    DEFAULT_PROFILES,
    _add_guard_stats,
    _agent_configs,
    _clean_metrics,
    _macro_average,
    run_episode,
)
from src.agents import get_agent  # noqa: E402
from src.env import PianoGymEnv  # noqa: E402
from src.utils import compute_metrics, save_results  # noqa: E402


OUTPUT_ROOT = Path("output/data/p1_guard")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_MARGINS: Sequence[float] = (0.00, 0.01, 0.03, 0.05, 0.07, 0.10)
DEFAULT_GUARD_DELTA = 0.08


def _make_seed(scenario: str, agent_name: str, profile_name: str, run_idx: int) -> int:
    # Keep the random stream paired across safety-margin settings so the
    # sensitivity comparison is not confounded by different initial conditions.
    token = f"guard_sensitivity|{agent_name}|{profile_name}|{run_idx}".encode("utf-8")
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


def _scenario_name(margin: float) -> str:
    return f"margin_{margin:.2f}".replace(".", "p")


def _margin_scenarios(margins: Sequence[float]) -> Dict[str, Dict[str, float | str]]:
    scenarios: Dict[str, Dict[str, float | str]] = {}
    for margin in margins:
        name = _scenario_name(float(margin))
        scenarios[name] = {
            "guard_delta": DEFAULT_GUARD_DELTA,
            "safety_margin": float(margin),
            "description": (
                f"Shared environment-side guard with delta_guard={DEFAULT_GUARD_DELTA:.2f} "
                f"and one-step safety_margin={float(margin):.2f}."
            ),
        }
    return scenarios


def _macro_average_with_ci(per_profile: Sequence[Mapping[str, object]], keys: Sequence[str]) -> Dict[str, float]:
    out = _macro_average(per_profile, keys)
    profile_count = max(len(per_profile), 1)
    for key in keys:
        mean_key = f"{key}_mean"
        ci_key = f"{key}_ci95"
        vals = [float(item[mean_key]) for item in per_profile if mean_key in item]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        if profile_count > 1:
            sem = np.nanstd(arr, ddof=1) / math.sqrt(profile_count)
            out[ci_key] = float(sem * _t_critical(profile_count - 1))
        else:
            out[ci_key] = 0.0
    return out


def run_experiment(
    *,
    num_runs: int = 10,
    margins: Sequence[float] = DEFAULT_MARGINS,
    agents: Sequence[str] | None = None,
) -> Dict[str, object]:
    probe_env = PianoGymEnv(seed=123)
    probe_obs = probe_env.reset()
    configs = _agent_configs(probe_env, probe_obs)
    if agents is None:
        agents = tuple(configs.keys())

    unknown_agents = [agent for agent in agents if agent not in configs]
    if unknown_agents:
        raise ValueError(f"Unknown agents: {unknown_agents}")

    scenarios = _margin_scenarios(margins)
    base_keys = [
        "time_to_mastery",
        "feasible_rate",
        "fatigue_overload_rate",
        "avg_constraint_violation",
        "avg_fatigue",
        "guard_replacement_rate",
        "proposed_step_cap_violation_rate",
    ]

    profile_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    macro_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    records = []

    print("=" * 70)
    print("Running guard-sensitivity experiment")
    print("=" * 70)

    for scenario_name, scenario in scenarios.items():
        guard_kwargs = {
            "guard_delta": float(scenario["guard_delta"]),
            "safety_margin": float(scenario["safety_margin"]),
        }
        profile_results[scenario_name] = {}
        macro_results[scenario_name] = {}
        print(f"\nScenario: {scenario_name} | {guard_kwargs}")

        for agent_name in agents:
            profile_results[scenario_name][agent_name] = {}
            per_profile = []
            cfg = dict(configs[agent_name])
            factory_name = cfg.pop("name")
            print(f"  Agent: {agent_name}")

            for profile in DEFAULT_PROFILES:
                profile_name = profile["name"]
                trajectories = []
                for run_idx in range(num_runs):
                    seed = _make_seed(scenario_name, agent_name, profile_name, run_idx)
                    env = PianoGymEnv(seed=seed)
                    agent = get_agent(
                        factory_name,
                        num_actions=env.cfg.num_actions,
                        seed=seed,
                        guard_delta=DEFAULT_GUARD_DELTA,
                        guard_horizon=1,
                        guard_safety_margin=0.05,
                        **cfg,
                    )
                    trajectories.append(
                        run_episode(
                            env,
                            agent,
                            profile,
                            guard_enabled=True,
                            guard_kwargs=guard_kwargs,
                        )
                    )

                metrics = compute_metrics(
                    trajectories,
                    mastery_window=env.cfg.mastery_window,
                    optimal_return=0.0,
                )
                _add_guard_stats(metrics, trajectories)
                clean = _clean_metrics(metrics)
                profile_results[scenario_name][agent_name][profile_name] = clean
                per_profile.append(clean)
                row = {
                    "scenario": scenario_name,
                    "agent": agent_name,
                    "profile": profile_name,
                    "num_runs": num_runs,
                    **guard_kwargs,
                }
                row.update(clean)
                records.append(row)
                print(
                    f"    {profile_name:17s} "
                    f"TTM={clean.get('time_to_mastery_mean', float('nan')):.1f} "
                    f"Feas={clean.get('feasible_rate_mean', float('nan')):.3f} "
                    f"Repl={clean.get('guard_replacement_rate_mean', float('nan')):.3f}"
                )

            macro_results[scenario_name][agent_name] = _macro_average_with_ci(per_profile, base_keys)

    payload = {
        "meta": {
            "num_runs": num_runs,
            "profiles": [profile["name"] for profile in DEFAULT_PROFILES],
            "agents": list(agents),
            "guard_delta": DEFAULT_GUARD_DELTA,
            "safety_margins": [float(margin) for margin in margins],
            "scenarios": scenarios,
            "interpretation": "Sensitivity check over one-step safety_margin; margin_0p05 is the original default.",
        },
        "metrics": profile_results,
        "macro": macro_results,
        "records": records,
    }

    json_path = OUTPUT_ROOT / "guard_sensitivity.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = OUTPUT_ROOT / "guard_sensitivity_summary.csv"
    if records:
        fieldnames = sorted({key for row in records for key in row.keys()})
        leading = ["scenario", "agent", "profile", "num_runs", "guard_delta", "safety_margin"]
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
    save_results(formatted, save_path=str(OUTPUT_ROOT / "guard_sensitivity_results.txt"))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guard-sensitivity ablation")
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per margin/agent/profile")
    parser.add_argument(
        "--margins",
        nargs="+",
        type=float,
        default=list(DEFAULT_MARGINS),
        help="One-step safety margins to evaluate.",
    )
    parser.add_argument("--agents", nargs="+", default=None, help="Optional subset of agents")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(num_runs=args.num_runs, margins=args.margins, agents=args.agents)
