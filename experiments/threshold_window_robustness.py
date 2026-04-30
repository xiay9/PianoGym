#!/usr/bin/env python
"""
Experiment E7: Threshold and mastery-window robustness.

Evaluate how sensitive top-performing agents are to fatigue-threshold scaling
and mastery-window choices, and report Kendall tau rank correlation against
the baseline configuration.
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import Config
from src.env import PianoGymEnv  # noqa: E402
from src.agents import get_agent  # noqa: E402
from src.utils import compute_metrics  # noqa: E402
from src.data_logging import ensure_data_dirs, resolve_data_path  # noqa: E402
from src.safety import ExternalSafetyGuard  # noqa: E402


THRESHOLD_SCALES: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
MASTERY_WINDOWS: Tuple[int, ...] = (1, 2, 3, 4, 5)
NUM_RUNS = 10
SEED_OFFSET = 211

PROFILE_NAME = "balanced"
TOP_AGENTS: Tuple[str, ...] = ("PianoMPC", "CCB-DF", "BayesianMAB")

OUTPUT_JSON = "e7_robustness_results.json"
OUTPUT_CSV = "e7_robustness_summary.csv"
DATA_SECTION = "p1_stability"
EXTERNAL_GUARD_CONFIG = {"guard_delta": 0.08, "safety_margin": 0.05}


def _make_seed(agent_name: str, setting_key: str, run_idx: int) -> int:
    token = f"{agent_name}|{setting_key}|{run_idx}".encode("utf-8")
    return zlib.crc32(token) & 0xFFFFFFFF


def _agent_factory_descriptor(agent_name: str, env: PianoGymEnv, context_dim: int) -> Tuple[str, Dict]:
    if agent_name == "PianoMPC":
        return "pianoMPC", {
            "reward_weights": env.cfg.w_reward,
            "reward_norm": env.cfg.reward_norm,
            "fatigue_limit": env.fatigue_threshold,
            "rest_recovery": env.cfg.rest_recovery,
            "pool_size": 6,
        }
    if agent_name == "CCB-DF":
        return "ccb_df", {
            "context_dim": context_dim,
            "delay_window": 1,
            "weight_clip": 10.0,
        }
    if agent_name == "BayesianMAB":
        return "bayesianmab", {
            "discount": 0.995,
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "sigma_noise": 0.5,
        }
    raise ValueError(f"Unsupported agent: {agent_name}")


def _simulate_runs(
    agents: Sequence[str],
    threshold_scale: float | None,
    mastery_window: int | None,
) -> Dict[str, Dict[str, float]]:
    """
    Run NUM_RUNS episodes per agent under the specified settings and return compute_metrics outputs.
    threshold_scale: multiply baseline fatigue threshold by this factor (None -> baseline).
    mastery_window: override mastery window (None -> baseline).
    """
    metrics_out: Dict[str, Dict[str, float]] = {}

    config_base = Config()
    baseline_threshold = config_base.fatigue_threshold
    threshold_value = baseline_threshold * threshold_scale if threshold_scale is not None else baseline_threshold
    window_value = mastery_window if mastery_window is not None else config_base.mastery_window

    profile_override = {
        "name": PROFILE_NAME,
        "fatigue_threshold": threshold_value,
    }

    for agent_name in agents:
        trajectories = []
        for run_idx in range(NUM_RUNS):
            setting_key = f"thr:{threshold_scale if threshold_scale is not None else 'baseline'}|" \
                          f"W:{window_value}"
            seed = _make_seed(agent_name, setting_key, run_idx)

            cfg_variant = Config()
            cfg_variant.fatigue_threshold = threshold_value
            cfg_variant.mastery_window = window_value

            env = PianoGymEnv(config=cfg_variant, seed=seed)
            obs = env.reset(profile=profile_override)

            context_dim = len(obs["context"]) + len(obs["action_features"][0])
            factory_name, kwargs = _agent_factory_descriptor(agent_name, env, context_dim)
            agent = get_agent(
                factory_name,
                num_actions=env.cfg.num_actions,
                seed=seed,
                guard_delta=0.08,
                guard_horizon=1,
                guard_safety_margin=0.05,
                **kwargs,
            )
            agent.reset()
            guard = ExternalSafetyGuard(
                env,
                enabled=True,
                **EXTERNAL_GUARD_CONFIG,
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
            }

            while not env.done:
                action = int(agent.select_action(obs))
                executed_action, guard_decision = guard.enforce(action, obs)
                next_obs, reward, done, info = env.step(executed_action)
                guard.annotate_info(info)
                next_obs_with_flag = dict(next_obs)
                next_obs_with_flag["done"] = done
                next_obs_with_flag["_info"] = info
                if hasattr(agent, "update") and callable(agent.update):
                    agent.update(obs, executed_action, reward, next_obs_with_flag)

                trajectory["rewards"].append(float(reward))
                trajectory["raw_rewards"].append(float(info.get("raw_reward", reward)))
                trajectory["obs_history"].append(next_obs)
                trajectory["info_history"].append(info)
                trajectory["actions"].append(executed_action)
                trajectory["proposed_actions"].append(action)
                trajectory["guard_history"].append(guard_decision.as_dict())
                trajectory["steps"] += 1

                obs = next_obs

            metrics = compute_metrics(
                [trajectory],
                mastery_window=env.cfg.mastery_window,
                optimal_return=0.0,
            )
            trajectories.append(metrics)

        # aggregate macro metrics (mean over runs using compute_metrics again)
        # We can average using compute_metrics on aggregated? Instead compute simple average for means, std.
        ttm_values = [m["time_to_mastery_mean"] for m in trajectories]
        feas_values = [m["feasible_rate_mean"] for m in trajectories]

        avg_fatigue_values = [m.get("avg_fatigue_mean", 0.0) for m in trajectories]

        def _mean(values: List[float]) -> float:
            if not values:
                return float("nan")
            return float(sum(values) / len(values))

        def _std(values: List[float]) -> float:
            n = len(values)
            if n <= 1:
                return 0.0
            mean_val = _mean(values)
            var = sum((v - mean_val) ** 2 for v in values) / (n - 1)
            return float(math.sqrt(var))

        metrics_out[agent_name] = {
            "time_to_mastery_mean": _mean(ttm_values),
            "time_to_mastery_std": _std(ttm_values),
            "feasible_rate_mean": _mean(feas_values),
            "feasible_rate_std": _std(feas_values),
            "avg_fatigue_mean": _mean(avg_fatigue_values),
            "avg_fatigue_std": _std(avg_fatigue_values),
            "threshold_scale": threshold_scale if threshold_scale is not None else 1.0,
            "threshold_value": threshold_value,
            "mastery_window": window_value,
        }

    return metrics_out


def _rank_from_metrics(metrics: Mapping[str, Mapping[str, float]]) -> Dict[str, int]:
    order = sorted(
        metrics.keys(),
        key=lambda agent: (metrics[agent].get("time_to_mastery_mean", float("inf")), agent),
    )
    return {agent: idx for idx, agent in enumerate(order)}


def _kendall_tau(rank_a: Mapping[str, int], rank_b: Mapping[str, int]) -> float:
    agents = [agent for agent in rank_a.keys() if agent in rank_b]
    n = len(agents)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = agents[i], agents[j]
            diff_a = rank_a[a_i] - rank_a[a_j]
            diff_b = rank_b[a_i] - rank_b[a_j]
            if diff_a == 0 or diff_b == 0:
                # tie; skip contribution
                continue
            if diff_a * diff_b > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


def run_threshold_window_robustness() -> Dict[str, object]:
    ensure_data_dirs()
    threshold_results: Dict[float, Dict[str, Dict[str, float]]] = {}
    window_results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for scale in THRESHOLD_SCALES:
        threshold_results[scale] = _simulate_runs(TOP_AGENTS, threshold_scale=scale, mastery_window=None)

    for window in MASTERY_WINDOWS:
        window_results[window] = _simulate_runs(TOP_AGENTS, threshold_scale=None, mastery_window=window)

    baseline_threshold_rank = _rank_from_metrics(threshold_results[1.0])
    baseline_window_rank = _rank_from_metrics(window_results[3])

    threshold_tau: Dict[float, float] = {}
    for scale, metrics in threshold_results.items():
        rank_map = _rank_from_metrics(metrics)
        tau = _kendall_tau(baseline_threshold_rank, rank_map)
        threshold_tau[scale] = tau

    window_tau: Dict[int, float] = {}
    for window, metrics in window_results.items():
        rank_map = _rank_from_metrics(metrics)
        tau = _kendall_tau(baseline_window_rank, rank_map)
        window_tau[window] = tau

    # write summary CSV
    summary_rows: List[Dict[str, object]] = []
    for scale, metrics in threshold_results.items():
        for agent, values in metrics.items():
            summary_rows.append({
                "scenario": "threshold",
                "threshold_scale": scale,
                "threshold_value": values.get("threshold_value"),
                "mastery_window": values.get("mastery_window"),
                "agent": agent,
                "time_to_mastery_mean": values.get("time_to_mastery_mean"),
                "time_to_mastery_std": values.get("time_to_mastery_std"),
                "feasible_rate_mean": values.get("feasible_rate_mean"),
                "feasible_rate_std": values.get("feasible_rate_std"),
                "avg_fatigue_mean": values.get("avg_fatigue_mean"),
                "avg_fatigue_std": values.get("avg_fatigue_std"),
                "kendall_tau": threshold_tau[scale],
            })

    for window, metrics in window_results.items():
        for agent, values in metrics.items():
            summary_rows.append({
                "scenario": "window",
                "threshold_scale": values.get("threshold_scale"),
                "threshold_value": values.get("threshold_value"),
                "mastery_window": window,
                "agent": agent,
                "time_to_mastery_mean": values.get("time_to_mastery_mean"),
                "time_to_mastery_std": values.get("time_to_mastery_std"),
                "feasible_rate_mean": values.get("feasible_rate_mean"),
                "feasible_rate_std": values.get("feasible_rate_std"),
                "avg_fatigue_mean": values.get("avg_fatigue_mean"),
                "avg_fatigue_std": values.get("avg_fatigue_std"),
                "kendall_tau": window_tau[window],
            })

    summary_path = resolve_data_path(DATA_SECTION, OUTPUT_CSV)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        header = [
            "scenario",
            "threshold_scale",
            "threshold_value",
            "mastery_window",
            "agent",
            "time_to_mastery_mean",
            "time_to_mastery_std",
            "feasible_rate_mean",
            "feasible_rate_std",
            "avg_fatigue_mean",
            "avg_fatigue_std",
            "kendall_tau",
        ]
        fh.write(",".join(header) + "\n")
        for row in summary_rows:
            values = [row.get(col, "") for col in header]
            fh.write(",".join(str(value) for value in values) + "\n")

    json_payload = {
        "meta": {
            "profile": PROFILE_NAME,
            "baseline_threshold": threshold_results[1.0][TOP_AGENTS[0]]["threshold_value"],
            "baseline_mastery_window": 3,
            "num_runs": NUM_RUNS,
            "agents": list(TOP_AGENTS),
        },
        "threshold": {
            "metrics": threshold_results,
            "kendall_tau": threshold_tau,
        },
        "window": {
            "metrics": window_results,
            "kendall_tau": window_tau,
        },
    }
    json_path = resolve_data_path(DATA_SECTION, OUTPUT_JSON)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2)

    print(f"✓ Summary CSV: {summary_path}")
    print(f"✓ Detail JSON: {json_path}")
    return json_payload


if __name__ == "__main__":
    run_threshold_window_robustness()
