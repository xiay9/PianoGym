"""
PianoGym Metrics
Trajectory-level and aggregate metric computation (see PianoGym.md §5)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class TrajectoryStats:
    total_reward: float
    total_reward_raw: float
    avg_reward_raw: float
    steps: int
    mastered: bool
    time_to_mastery: int
    sample_efficiency: float
    avg_async: float
    avg_dom_gap: float
    avg_fid: float
    skill_improvement: float
    retention_delta: float
    independence_gain: float
    regret: float
    avg_fatigue: float
    avg_constraint_violation: float
    fatigue_overload_rate: float
    feasible_rate: float


@dataclass
class SafetyStats:
    avg_fatigue: float
    avg_constraint_violation: float
    fatigue_overload_rate: float
    feasible_rate: float


@dataclass
class GuardInterventionStats:
    guard_replacement_rate: float
    action_changed_rate: float
    rest_fallback_rate: float
    relaxed_execution_rate: float
    proposed_step_cap_violation_rate: float
    proposed_relaxed_violation_rate: float
    executed_step_cap_violation_rate: float


def _first_mastery_step(info_history: Sequence[Dict], mastery_window: int) -> int | None:
    for idx, info in enumerate(info_history, start=1):
        if info.get('mastery_count', 0) >= mastery_window:
            return idx
    return None


def _independence_gain(initial_obs: Dict, final_obs: Dict) -> float:
    async_gain = initial_obs['async'] - final_obs['async']
    dom_gain = abs(initial_obs['dom_gap']) - abs(final_obs['dom_gap'])
    fid_gain = final_obs['fid'] - initial_obs['fid']
    return np.mean([async_gain, dom_gain, fid_gain])


def _retention_delta(initial_info: Dict | None, final_info: Dict | None) -> float:
    if not initial_info or not final_info:
        return 0.0
    return final_info.get('retention', 0.0) - initial_info.get('retention', 0.0)


def compute_safety_stats(info_history: Sequence[Dict]) -> SafetyStats:
    """Compute safety diagnostics from fatigue samples and their thresholds.

    Definitions are intentionally independent of whether the environment used a
    constraint penalty. ``AvgViolation`` is the mean magnitude
    ``max(0, f_t - tau_t)``; ``OverloadRate`` is the fraction of samples with a
    positive violation; ``FeasibleRate`` is exactly ``1 - OverloadRate``.
    """
    if not info_history:
        return SafetyStats(
            avg_fatigue=0.0,
            avg_constraint_violation=0.0,
            fatigue_overload_rate=0.0,
            feasible_rate=1.0,
        )

    fatigue_vals = np.asarray([info.get('fatigue', 0.0) for info in info_history], dtype=float)
    thresholds = np.asarray(
        [
            info.get('fatigue_threshold', info.get('threshold', 1.0))
            for info in info_history
        ],
        dtype=float,
    )
    violation_vals = np.maximum(0.0, fatigue_vals - thresholds)
    overload_vals = violation_vals > 0.0
    overload_rate = float(np.mean(overload_vals)) if overload_vals.size else 0.0
    feasible_rate = float(np.clip(1.0 - overload_rate, 0.0, 1.0))
    return SafetyStats(
        avg_fatigue=float(np.mean(fatigue_vals)) if fatigue_vals.size else 0.0,
        avg_constraint_violation=float(np.mean(violation_vals)) if violation_vals.size else 0.0,
        fatigue_overload_rate=overload_rate,
        feasible_rate=feasible_rate,
    )


def _lookup_predicted_fatigue(predicted: Dict, action_idx: int) -> float:
    if action_idx in predicted:
        return float(predicted[action_idx])
    action_key = str(action_idx)
    if action_key in predicted:
        return float(predicted[action_key])
    return 0.0


def compute_guard_intervention_stats(guard_history: Sequence[Dict]) -> GuardInterventionStats:
    """Compute guard diagnostics from proposed and executed action records.

    These are intervention diagnostics, not safety constraints. They measure how
    often the wrapper altered a proposal and how often the proposal/execution
    exceeded the guard's one-step or relaxed thresholds.
    """
    if not guard_history:
        return GuardInterventionStats(
            guard_replacement_rate=0.0,
            action_changed_rate=0.0,
            rest_fallback_rate=0.0,
            relaxed_execution_rate=0.0,
            proposed_step_cap_violation_rate=0.0,
            proposed_relaxed_violation_rate=0.0,
            executed_step_cap_violation_rate=0.0,
        )

    replaced = []
    changed = []
    rest_fallback = []
    relaxed = []
    proposed_step = []
    proposed_relaxed = []
    executed_step = []

    for item in guard_history:
        if not isinstance(item, dict):
            continue
        original = int(item.get('original_action', -1))
        final = int(item.get('final_action', original))
        reason = str(item.get('reason', ''))
        predicted = item.get('predicted_next_fatigue', {})
        if not isinstance(predicted, dict):
            predicted = {}
        step_cap = float(item.get('step_cap', 0.0))
        guard_threshold = float(item.get('guard_threshold', step_cap))
        original_pred = _lookup_predicted_fatigue(predicted, original)
        final_pred = _lookup_predicted_fatigue(predicted, final)

        replaced.append(1.0 if item.get('replaced', False) else 0.0)
        changed.append(1.0 if original != final else 0.0)
        rest_fallback.append(1.0 if reason == 'rest_fallback' else 0.0)
        relaxed.append(1.0 if reason == 'relaxed' else 0.0)
        proposed_step.append(1.0 if original_pred > step_cap else 0.0)
        proposed_relaxed.append(1.0 if original_pred > guard_threshold else 0.0)
        executed_step.append(1.0 if final_pred > step_cap else 0.0)

    def mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    return GuardInterventionStats(
        guard_replacement_rate=mean(replaced),
        action_changed_rate=mean(changed),
        rest_fallback_rate=mean(rest_fallback),
        relaxed_execution_rate=mean(relaxed),
        proposed_step_cap_violation_rate=mean(proposed_step),
        proposed_relaxed_violation_rate=mean(proposed_relaxed),
        executed_step_cap_violation_rate=mean(executed_step),
    )


def compute_single_trajectory(
    trajectory: Dict,
    mastery_window: int = 3,
    optimal_return: float = 0.0,
) -> TrajectoryStats:
    rewards: Sequence[float] = trajectory.get('rewards', [])
    raw_rewards: Sequence[float] = trajectory.get('raw_rewards', [])
    obs_history: Sequence[Dict] = trajectory.get('obs_history', [])
    info_history: Sequence[Dict] = trajectory.get('info_history', [])
    initial_info = trajectory.get('initial_info')
    final_info = info_history[-1] if info_history else initial_info

    total_reward = float(np.sum(rewards))
    total_reward_raw = float(np.sum(raw_rewards)) if len(raw_rewards) == len(rewards) and raw_rewards else total_reward
    steps = trajectory.get('steps', len(rewards))
    avg_reward_raw = (total_reward_raw / steps) if steps > 0 else 0.0

    mastery_step = _first_mastery_step(info_history, mastery_window)
    mastered = mastery_step is not None
    time_to_mastery = mastery_step if mastered else steps
    sample_efficiency = float(time_to_mastery) if mastered else np.nan

    if obs_history:
        initial_obs = obs_history[0]
        final_obs = obs_history[-1]
        avg_async = float(np.mean([o['async'] for o in obs_history]))
        avg_dom_gap = float(np.mean([abs(o['dom_gap']) for o in obs_history]))
        avg_fid = float(np.mean([o['fid'] for o in obs_history]))
        independence_gain = float(_independence_gain(initial_obs, final_obs))
    else:
        avg_async = avg_dom_gap = avg_fid = 0.0
        independence_gain = 0.0

    if info_history:
        init_skills = info_history[0]['true_skills']
        final_skills = info_history[-1]['true_skills']
        skill_improvement = float(np.mean(final_skills - init_skills))

        safety_stats = compute_safety_stats(info_history)
    else:
        skill_improvement = 0.0
        safety_stats = compute_safety_stats(info_history)

    retention_delta = float(_retention_delta(initial_info, final_info))
    regret = float(optimal_return - total_reward)

    return TrajectoryStats(
        total_reward=total_reward,
        total_reward_raw=total_reward_raw,
        avg_reward_raw=avg_reward_raw,
        steps=steps,
        mastered=mastered,
        time_to_mastery=time_to_mastery,
        sample_efficiency=sample_efficiency,
        avg_async=avg_async,
        avg_dom_gap=avg_dom_gap,
        avg_fid=avg_fid,
        skill_improvement=skill_improvement,
        retention_delta=retention_delta,
        independence_gain=independence_gain,
        regret=regret,
        avg_fatigue=safety_stats.avg_fatigue,
        avg_constraint_violation=safety_stats.avg_constraint_violation,
        fatigue_overload_rate=safety_stats.fatigue_overload_rate,
        feasible_rate=safety_stats.feasible_rate,
    )


def aggregate_metrics(
    trajectories: Sequence[TrajectoryStats],
) -> Dict[str, float | List[float]]:
    summary: Dict[str, float | List[float]] = {}

    def _collect(attr: str) -> np.ndarray:
        values = np.array([getattr(t, attr) for t in trajectories], dtype=float)
        summary[f'{attr}_raw'] = values.tolist()
        finite = values[np.isfinite(values)]
        if finite.size:
            summary[f'{attr}_mean'] = float(np.mean(finite))
            summary[f'{attr}_std'] = float(np.std(finite))
        else:
            summary[f'{attr}_mean'] = np.nan
            summary[f'{attr}_std'] = np.nan
        return values

    for field in TrajectoryStats.__dataclass_fields__:
        _collect(field)

    mastery_rate = float(np.mean([1.0 if t.mastered else 0.0 for t in trajectories])) if trajectories else 0.0
    summary['mastery_rate'] = mastery_rate
    return summary


def compute_metrics(
    trajectories: Sequence[Dict],
    mastery_window: int = 3,
    optimal_return: float = 0.0,
) -> Dict[str, float | List[float]]:
    per_traj = [
        compute_single_trajectory(traj, mastery_window=mastery_window, optimal_return=optimal_return)
        for traj in trajectories
    ]
    return aggregate_metrics(per_traj)


def compute_regret_table(results: Dict[str, Dict[str, float]], objective: str = 'ttm',
                         feasibility_threshold: float = 0.90) -> Dict[str, float]:
    """Compute regret table under a given objective.

    Parameters
    ----------
    results : mapping from agent name to aggregated metrics.
    objective : 'ttm' (default) uses Time-to-Mastery; 'return' uses raw total reward.
    feasibility_threshold : minimum feasible_rate to be considered feasible (default 0.90).

    Notes
    -----
    Regret is computed only among feasible agents (feasible_rate >= threshold).
    Infeasible agents are assigned NaN regret.
    """
    if not results:
        return {}

    # Separate feasible and infeasible agents.
    feasible_agents = {
        agent: metrics for agent, metrics in results.items()
        if metrics.get('feasible_rate_mean', 0.0) >= feasibility_threshold
    }

    if not feasible_agents:
        # If no agent is feasible, every regret value is NaN.
        return {agent: np.nan for agent in results.keys()}

    if objective == 'ttm':
        metric_key = 'time_to_mastery_mean'
        # Compute the best score only among feasible agents.
        feasible_scores = {
            agent: float(metrics.get(metric_key, np.inf))
            for agent, metrics in feasible_agents.items()
        }
        best_score = min(feasible_scores.values())

        # Compute regret among feasible agents; assign NaN to infeasible ones.
        regret_dict = {}
        for agent, metrics in results.items():
            if agent in feasible_agents:
                score = float(metrics.get(metric_key, np.inf))
                regret_dict[agent] = float(score - best_score) if np.isfinite(score) else np.inf
            else:
                regret_dict[agent] = np.nan
        return regret_dict

    if objective == 'return':
        metric_key = 'total_reward_raw_mean'
        # Compute the best score only among feasible agents.
        feasible_scores = {
            agent: float(metrics.get(metric_key, -np.inf))
            for agent, metrics in feasible_agents.items()
        }
        best_score = max(feasible_scores.values())

        # Compute regret among feasible agents; assign NaN to infeasible ones.
        regret_dict = {}
        for agent, metrics in results.items():
            if agent in feasible_agents:
                score = float(metrics.get(metric_key, -np.inf))
                regret_dict[agent] = float(best_score - score) if np.isfinite(score) else np.inf
            else:
                regret_dict[agent] = np.nan
        return regret_dict

    raise ValueError(f"Unsupported objective for regret: {objective}")
