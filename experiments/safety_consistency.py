"""
Experiment A1: Online safety consistency diagnosis

Scan rolling-window and peak metrics to verify consistency between guards
(hard constraints) and soft-multiplier behavior.
"""
from __future__ import annotations

import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import PianoGymEnv  # noqa: E402
from src.agents import get_agent  # noqa: E402
from src.data_logging import ensure_data_dirs, resolve_data_path, write_csv_records  # noqa: E402
from src.safety import ExternalSafetyGuard  # noqa: E402


WINDOW_SIZES = (5, 10, 20)
GUARD_HORIZONS = (3, 5, 7)
DELTA_GUARDS = (0.0, 0.05, 0.1)

NUM_SEEDS = 5
SEED_OFFSET = 100  # Avoid using overly similar random sequences across runs.

# Focus on PianoMPC for the main paper; this can be extended to all eight algorithms.
AGENTS = ("pianoMPC",)

# Lower the fatigue threshold so violations become observable.
PROFILES: Tuple[Mapping, ...] = (
    {'name': 'balanced', 'left_weakness': 0.0, 'fatigue_threshold': 0.25},
    {'name': 'mild_left_weak', 'left_weakness': 0.1, 'fatigue_threshold': 0.22},
    {'name': 'severe_left_weak', 'left_weakness': 0.2, 'fatigue_threshold': 0.20},
)

EXTERNAL_GUARD_MARGIN = 0.05


@dataclass
class StepRecord:
    step: int
    fatigue: float
    reward: float
    action: int
    proposed_action: int | None
    guard_replaced: bool
    guard_reason: str | None
    env_guard: Dict[str, object] | None
    mastery_flag: int
    guard_pass: bool
    lambda_t: float | None
    fatigue_threshold: float
    predicted_peak_guard: float | None
    guard_threshold: float | None
    will_block: bool


def _make_seed(agent_name: str, profile_name: str, guard_horizon: int, delta: float, run_idx: int) -> int:
    """Generate stable, non-overlapping random seeds."""
    token = (
        f"{agent_name}|{profile_name}|{guard_horizon}|"
        f"{float(delta):.8f}|{run_idx}|{SEED_OFFSET}"
    ).encode("utf-8")
    return zlib.crc32(token) & 0xFFFFFFFF


def _extract_metadata_value(metadata: Mapping, key: str):
    if key in metadata:
        return metadata[key]
    base_meta = metadata.get('base_metadata')
    if isinstance(base_meta, Mapping) and key in base_meta:
        return base_meta[key]
    return None


def _extract_guard_pass(metadata: Mapping) -> bool:
    value = _extract_metadata_value(metadata, 'guard_pass')
    if value is None:
        return True
    return bool(value)


def _extract_bool(metadata: Mapping, key: str, default: bool = False) -> bool:
    value = _extract_metadata_value(metadata, key)
    if value is None:
        return default
    return bool(value)


def _make_agent(
    agent_name: str,
    env: PianoGymEnv,
    context_dim: int,
    seed: int,
    guard_horizon: int,
    guard_delta: float,
):
    num_actions = env.cfg.num_actions
    if agent_name == 'linucb':
        return get_agent(
            'linucb',
            num_actions=num_actions,
            context_dim=context_dim,
            alpha=2.0,
            ridge=1e-3,
            seed=seed,
            guard_horizon=guard_horizon,
            guard_delta=guard_delta,
        )
    if agent_name == 'pianoMPC':
        return get_agent(
            'pianoMPC',
            num_actions=num_actions,
            reward_weights=env.cfg.w_reward,
            reward_norm=env.cfg.reward_norm,
            fatigue_limit=env.fatigue_threshold,  # Use the dynamic threshold from the environment.
            rest_recovery=env.cfg.rest_recovery,
            seed=seed,
            guard_horizon=guard_horizon,
            guard_delta=guard_delta,
        )
    raise ValueError(f"Unsupported agent: {agent_name}")


def _collect_episode(
    agent_name: str,
    profile: Mapping,
    seed: int,
    guard_horizon: int,
    guard_delta: float,
) -> Tuple[List[StepRecord], Dict]:
    env = PianoGymEnv(seed=seed)
    obs = env.reset(profile=profile)
    context_dim = len(obs['context']) + len(obs['action_features'][0])
    agent = _make_agent(
        agent_name,
        env,
        context_dim,
        seed=seed,
        guard_horizon=guard_horizon,
        guard_delta=guard_delta,
    )
    agent.reset()
    guard = ExternalSafetyGuard(
        env,
        enabled=True,
        guard_delta=guard_delta,
        safety_margin=EXTERNAL_GUARD_MARGIN,
    )
    guard.reset()

    records: List[StepRecord] = []
    step = 0
    info_snapshot = {}

    while not env.done:
        action = agent.select_action(obs)
        metadata = getattr(agent, 'get_last_metadata', lambda: {})() or {}

        executed_action, guard_decision = guard.enforce(int(action), obs)
        next_obs, reward, done, info = env.step(executed_action)
        guard.annotate_info(info)
        mastery_flag = 1 if info.get('mastery_count', 0) >= env.cfg.mastery_window else 0
        # Use the actual environment threshold, which may be overridden by the profile.
        fatigue_threshold = env.fatigue_threshold
        lambda_t = _extract_metadata_value(metadata, 'lambda_t')
        predicted_peak_guard = _extract_metadata_value(metadata, 'predicted_peak_guard')
        guard_threshold = _extract_metadata_value(metadata, 'guard_threshold')
        will_block = _extract_bool(metadata, 'will_block', default=False)

        records.append(StepRecord(
            step=step,
            fatigue=float(info.get('fatigue', 0.0)),
            reward=float(reward),
            action=int(executed_action),
            proposed_action=int(action),
            guard_replaced=bool(guard_decision.replaced),
            guard_reason=guard_decision.reason,
            env_guard=guard_decision.as_dict(),
            mastery_flag=mastery_flag,
            guard_pass=_extract_guard_pass(metadata),
            lambda_t=float(lambda_t) if lambda_t is not None else None,
            fatigue_threshold=float(fatigue_threshold),
            predicted_peak_guard=float(predicted_peak_guard) if predicted_peak_guard is not None else None,
            guard_threshold=float(guard_threshold) if guard_threshold is not None else None,
            will_block=bool(will_block),
        ))

        info_snapshot = info  # Keep the final info record for reference.
        obs = next_obs
        step += 1

    return records, {
        'total_steps': step,
        'term_info': info_snapshot,
        'profile': profile.get('name', 'unknown'),
    }


def _precompute_windows_and_peaks(
    fatigue: np.ndarray,
    window_sizes: Sequence[int],
    guard_horizon: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Precompute rolling-window maxima and future peaks."""
    n = fatigue.size
    window_map: Dict[int, np.ndarray] = {}

    # Compute rolling-window maxima.
    for W in window_sizes:
        window_vals = np.zeros(n, dtype=float)
        for idx in range(n):
            start = max(0, idx - W + 1)
            window_vals[idx] = float(np.max(fatigue[start:idx + 1]))
        window_map[W] = window_vals

    # Compute future-peak values.
    peak_vals = np.zeros(n, dtype=float)
    for idx in range(n):
        end = min(n, idx + guard_horizon + 1)
        peak_vals[idx] = float(np.max(fatigue[idx:end]))

    return window_map, peak_vals


def _format_delta(delta: float) -> str:
    return f"{delta:.2f}".replace('.', 'p')


def _write_summary_csv(summary_rows: Sequence[Dict[str, object]]) -> None:
    """Write the summary CSV."""
    path = resolve_data_path("p1_safety", "safety_consistency_summary.csv")
    fieldnames = [
        "agent",
        "profile",
        "W",
        "Hg",
        "delta_guard",
        "num_steps",
        "num_valid_steps",  # Added: number of valid steps with a full window.
        "num_seeds",
        "window_violation_rate",
        "peak_violation_rate",
        "rate_diff",
        "agent_guard_pass_rate",
        "std_guard_pass_rate",  # Added: standard guard pass rate.
        "guard_false_negative_rate",  # Renamed: guard false-negative rate.
        "window_violation_with_guard_rate",  # Renamed: window violation rate when the guard passes.
    ]
    write_csv_records(path, summary_rows, fieldnames, append=False)


def _write_detail_csv(
    detail_rows: List[Dict[str, object]],
    agent_name: str,
    profile_name: str,
    W: int,
    Hg: int,
    delta: float,
) -> None:
    """Write the detailed CSV into a hierarchical folder structure for visualization."""
    # Folder layout: p1_safety/detail/agent_profile_W_Hg_delta/
    subdir = f"{agent_name}_{profile_name}_W{W}_Hg{Hg}_d{_format_delta(delta)}"
    filename = "detail.csv"
    # Build the nested path directly.
    base_path = resolve_data_path("p1_safety", "")
    detail_dir = base_path / "detail" / subdir
    detail_dir.mkdir(parents=True, exist_ok=True)
    path = detail_dir / filename

    if detail_rows:
        fieldnames = list(detail_rows[0].keys())
        write_csv_records(path, detail_rows, fieldnames, append=False)


def _aggregate_rows(
    rows: Iterable[Dict[str, object]],
    min_valid_steps: int = 30,
) -> Dict[str, float]:
    """Aggregate statistics using only full-window steps; return NaN when the sample is too small."""
    rows_list = list(rows)
    total = len(rows_list)
    if total == 0:
        return {
            "num_steps": 0,
            "num_valid_steps": 0,
            "window_violation_rate": float("nan"),
            "peak_violation_rate": float("nan"),
            "rate_diff": float("nan"),
            "agent_guard_pass_rate": float("nan"),
            "std_guard_pass_rate": float("nan"),
            "guard_false_negative_rate": float("nan"),
            "window_violation_with_guard_rate": float("nan"),
        }

    def _as_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in {"true", "1", "yes"}
        return bool(v)

    # Keep only rows with ``is_valid_step=True``.
    valid_rows = [r for r in rows_list if r.get("is_valid_step", True)]
    num_valid = len(valid_rows)

    # Too few valid steps: mark as NaN and filter later during plotting.
    if num_valid < min_valid_steps:
        nan = float("nan")
        return {
            "num_steps": total,
            "num_valid_steps": num_valid,
            "window_violation_rate": nan,
            "peak_violation_rate": nan,
            "rate_diff": nan,
            "agent_guard_pass_rate": nan,
            "std_guard_pass_rate": nan,
            "guard_false_negative_rate": nan,
            "window_violation_with_guard_rate": nan,
        }

    window_sum = sum(int(r["violation_window_flag"]) for r in valid_rows)
    peak_sum = sum(int(r["violation_peak_flag"]) for r in valid_rows)
    agent_guard_pass_sum = sum(1 for r in valid_rows if _as_bool(r["agent_guard_pass"]))
    std_guard_pass_sum = sum(1 for r in valid_rows if _as_bool(r["std_guard_pass"]))

    # Guard passes but the peak still violates the threshold -> false negative.
    agent_fn = sum(
        1 for r in valid_rows
        if bool(r["agent_guard_pass"]) and not bool(r["std_guard_pass"])
    )

    # Guard passes but the rolling window still violates the threshold.
    window_with_guard = sum(
        1 for r in valid_rows
        if _as_bool(r["std_guard_pass"]) and int(r["violation_window_flag"]) == 1
    )

    window_rate = window_sum / num_valid
    peak_rate = peak_sum / num_valid

    return {
        "num_steps": total,
        "num_valid_steps": num_valid,
        "window_violation_rate": window_rate,
        "peak_violation_rate": peak_rate,
        "rate_diff": window_rate - peak_rate,
        "agent_guard_pass_rate": agent_guard_pass_sum / num_valid,
        "std_guard_pass_rate": std_guard_pass_sum / num_valid,
        "guard_false_negative_rate": agent_fn / num_valid,
        "window_violation_with_guard_rate": window_with_guard / num_valid,
    }


def run_safety_consistency(
    agents: Sequence[str] = AGENTS,
    profiles: Sequence[Mapping] = PROFILES,
    window_sizes: Sequence[int] = WINDOW_SIZES,
    guard_horizons: Sequence[int] = GUARD_HORIZONS,
    delta_guards: Sequence[float] = DELTA_GUARDS,
    num_seeds: int = NUM_SEEDS,
    min_steps_per_seed: int = 200,   # Can be increased if needed.
) -> Dict[str, object]:
    """Main entry point: concatenate multiple episodes for each (agent, profile, Hg, delta, seed) tuple and then compute statistics."""
    ensure_data_dirs()

    combo_rows: Dict[Tuple[str, str, int, int, float], List[Dict[str, object]]] = defaultdict(list)
    combo_seeds: Dict[Tuple[str, str, int, int, float], set] = defaultdict(set)
    max_W = max(window_sizes)

    for agent_name in agents:
        for profile in profiles:
            profile_name = profile.get("name", "unknown")
            print(f"\n=== Agent: {agent_name} | Profile: {profile_name} ===")

            for guard_horizon in guard_horizons:
                for delta in delta_guards:
                    for run_idx in range(num_seeds):
                        base_seed = _make_seed(agent_name, profile_name, guard_horizon, delta, run_idx)

                        # These lists form one concatenated trajectory.
                        all_fatigue: List[float] = []
                        all_thresholds: List[float] = []
                        all_actions: List[int] = []
                        all_proposed_actions: List[int] = []
                        all_guard_replaced: List[bool] = []
                        all_guard_reason: List[str | None] = []
                        all_rewards: List[float] = []
                        all_mastery: List[int] = []
                        all_agent_pass: List[bool] = []
                        all_pred_peaks: List[float | None] = []
                        all_guard_thres: List[float | None] = []
                        all_will_block: List[bool] = []

                        total_steps = 0
                        episode_offset = 0

                        # Keep running until enough steps are collected.
                        while total_steps < min_steps_per_seed:
                            # Slightly perturb the seed for each episode.
                            seed = (base_seed + episode_offset) & 0xFFFFFFFF
                            records, _ = _collect_episode(
                                agent_name,
                                profile,
                                seed=seed,
                                guard_horizon=guard_horizon,
                                guard_delta=delta,
                            )
                            if not records:
                                break

                            for rec in records:
                                all_fatigue.append(rec.fatigue)
                                all_thresholds.append(rec.fatigue_threshold)
                                all_actions.append(rec.action)
                                all_proposed_actions.append(
                                    rec.proposed_action if rec.proposed_action is not None else rec.action
                                )
                                all_guard_replaced.append(rec.guard_replaced)
                                all_guard_reason.append(rec.guard_reason)
                                all_rewards.append(rec.reward)
                                all_mastery.append(rec.mastery_flag)
                                all_agent_pass.append(rec.guard_pass)
                                all_pred_peaks.append(rec.predicted_peak_guard)
                                all_guard_thres.append(rec.guard_threshold)
                                all_will_block.append(rec.will_block)

                            total_steps += len(records)
                            # Use a large offset to avoid nearby seeds.
                            episode_offset += 7919

                        if total_steps < max_W:
                            # If it is still too short, skip aggregation.
                            continue

                        # Compute rolling windows and peaks over the full concatenated trajectory.
                        fatigue_arr = np.asarray(all_fatigue, dtype=float)
                        th_arr = np.asarray(all_thresholds, dtype=float)

                        window_map, actual_peak_arr = _precompute_windows_and_peaks(
                            fatigue_arr, window_sizes, guard_horizon
                        )

                        for W, window_vals in window_map.items():
                            valid_idx = np.arange(total_steps) >= (W - 1)

                            # Window violation: compare against tau.
                            window_flags = window_vals > th_arr

                            # Online guard: compare against tau + delta using the future max.
                            std_peak_thresholds = th_arr + delta
                            peak_flags = actual_peak_arr > std_peak_thresholds
                            std_guard_pass_flags = actual_peak_arr <= std_peak_thresholds

                            rows: List[Dict[str, object]] = []
                            for idx in range(total_steps):
                                pred_peak = all_pred_peaks[idx]
                                guard_thr = all_guard_thres[idx]

                                rows.append({
                                    "agent": agent_name,
                                    "profile": profile_name,
                                    "seed": base_seed,   # Use ``base_seed`` to identify the concatenated trajectory.
                                    "W": W,
                                    "Hg": guard_horizon,
                                    "delta_guard": delta,
                                    "step": idx,
                                    "is_valid_step": bool(valid_idx[idx]),
                                    "action_id": all_actions[idx],
                                    "proposed_action_id": all_proposed_actions[idx],
                                    "env_guard_pass": bool(not all_guard_replaced[idx]),
                                    "env_guard_replaced": bool(all_guard_replaced[idx]),
                                    "env_guard_reason": all_guard_reason[idx],
                                    "reward": all_rewards[idx],
                                    "mastery_flag": all_mastery[idx],
                                    "f_t": fatigue_arr[idx],
                                    "window_max": float(window_vals[idx]),
                                    "predicted_peak": float(pred_peak) if pred_peak is not None else None,
                                    "actual_peak": float(actual_peak_arr[idx]),
                                    "window_threshold": float(th_arr[idx]),
                                    "peak_threshold": float(std_peak_thresholds[idx]),
                                    "agent_guard_pass": bool(all_agent_pass[idx]),
                                    "std_guard_pass": bool(std_guard_pass_flags[idx]),
                                    "lambda_t": None,     # Leave blank if it was not collected earlier.
                                    "will_block": bool(all_will_block[idx]),
                                    "violation_window_flag": int(window_flags[idx]),
                                    "violation_peak_flag": int(peak_flags[idx]),
                                })

                            combo_key = (agent_name, profile_name, W, guard_horizon, delta)
                            combo_rows[combo_key].extend(rows)
                            combo_seeds[combo_key].add(base_seed)

                            # Write one detail file for each (seed, W, Hg, delta).
                            _write_detail_csv(rows, agent_name, profile_name, W, guard_horizon, delta)

    # Aggregate summaries.
    summary_rows: List[Dict[str, object]] = []
    for combo_key, rows in combo_rows.items():
        agent_name, profile_name, W, Hg, delta = combo_key
        stats = _aggregate_rows(rows, min_valid_steps=30)
        stats.update({
            "agent": agent_name,
            "profile": profile_name,
            "W": W,
            "Hg": Hg,
            "delta_guard": delta,
            "num_seeds": len(combo_seeds.get(combo_key, set())),
        })
        summary_rows.append(stats)

    _write_summary_csv(summary_rows)

    print("\n" + "=" * 70)
    print("Safety consistency data saved to output/data/p1_safety/")
    print("  - Summary: safety_consistency_summary.csv")
    print(f"  - Details: {len(combo_rows)} subfolders in p1_safety/detail/")
    print("=" * 70)

    return {
        "summary": summary_rows,
        "detail_files": list(combo_rows.keys()),
    }


if __name__ == "__main__":
    run_safety_consistency()
