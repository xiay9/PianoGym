"""
Experiment 1: Strategy comparison
Compare Fixed/Random/UCB/LinUCB/Thompson on Time-to-Mastery and cumulative reward.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from src.env import PianoGymEnv
from src.agents import get_agent
from src.utils import compute_metrics, save_results, compute_regret_table
from src.safety import ExternalSafetyGuard
from configs import cfg


def run_episode(
    env,
    agent,
    profile=None,
    seed=None,
    *,
    guard_enabled: bool = True,
    guard_kwargs: dict | None = None,
):
    """Run a single episode."""
    obs = env.reset(profile=profile)
    agent.reset()
    guard = ExternalSafetyGuard(
        env,
        enabled=guard_enabled,
        **(guard_kwargs or {}),
    )
    guard.reset()

    trajectory = {
        'rewards': [],
        'raw_rewards': [],
        'obs_history': [obs],
        'info_history': [],
        'actions': [],
        'action_probs': [],
        'proposed_actions': [],
        'guard_history': [],
        'steps': 0,
        'done': False,
        'initial_info': {
            'true_skills': env.x.copy(),
            'fatigue': env.f,
            'retention': env.r,
        },
    }

    while not env.done:
        action = agent.select_action(obs)
        prob = agent.action_prob(obs, action)

        executed_action, guard_decision = guard.enforce(int(action), obs)

        next_obs, reward, done, info = env.step(executed_action)
        guard.annotate_info(info)

        next_obs_with_done = dict(next_obs)
        next_obs_with_done['done'] = done
        next_obs_with_done['_info'] = info
        agent.update(obs, executed_action, reward, next_obs_with_done)

        trajectory['rewards'].append(reward)
        trajectory['raw_rewards'].append(info.get('raw_reward', reward))
        trajectory['obs_history'].append(next_obs)
        trajectory['info_history'].append(info)
        trajectory['actions'].append(executed_action)
        trajectory['proposed_actions'].append(action)
        trajectory['action_probs'].append(prob)
        trajectory['guard_history'].append(guard_decision.as_dict())
        trajectory['steps'] += 1

        obs = next_obs

    trajectory['done'] = env.done
    mastery_step = next(
        (
            idx + 1
            for idx, info in enumerate(trajectory['info_history'])
            if info.get('mastery_count', 0) >= env.cfg.mastery_window
        ),
        0,
    )
    trajectory['mastery_step'] = mastery_step
    return trajectory


def run_experiment(num_runs=10, profiles=None, regret_thresholds=(0.90, 0.0)):
    """Run the comparison experiment.
    num_runs: number of runs per policy
    profiles: learner profile list
    """
    if profiles is None:
        profiles = [
            {'name': 'balanced', 'left_weakness': 0.0},
            {'name': 'mild_left_weak', 'left_weakness': 0.1},
            {'name': 'severe_left_weak', 'left_weakness': 0.2},
        ]

    probe_env = PianoGymEnv(seed=123)
    probe_obs = probe_env.reset()
    context_dim = len(probe_obs['context']) + len(probe_obs['action_features'][0])
    state_dim = len(probe_obs['context'])
    num_actions = probe_env.cfg.num_actions
    num_types = probe_env.cfg.num_exercise_types
    skill_dim = probe_env.cfg.K

    # Define the retained set of eight algorithms.
    agent_configs = {
        'LinUCB': {'name': 'linucb', 'context_dim': context_dim, 'alpha': 2.0, 'ridge': 1e-3, 'use_sherman_morrison': True},
        'Thompson': {'name': 'thompson', 'context_dim': context_dim, 'v': 0.8},
        'PianoMPC': {
            'name': 'pianoMPC',
            'horizon': 3,
            'reward_weights': probe_env.cfg.w_reward,
            'reward_norm': probe_env.cfg.reward_norm,
            'fatigue_limit': probe_env.cfg.fatigue_threshold,
            'pool_size': 6,
            'rest_recovery': probe_env.cfg.rest_recovery,
        },
        'DQN': {'name': 'dqn', 'state_dim': state_dim, 'hidden_dim': 64, 'epsilon': 0.2},
        'BayesianMAB': {'name': 'bayesianmab', 'discount': 0.995, 'prior_mu': 0.0, 'prior_sigma': 1.0, 'sigma_noise': 0.5},
        'CCB-DF': {'name': 'ccb_df', 'context_dim': context_dim, 'delay_window': 1, 'weight_clip': 10.0},
        'Safe-AC': {'name': 'safe_ac', 'state_dim': state_dim, 'hidden_dim': 32, 'cost_limit': probe_env.cfg.fatigue_threshold},
        'AutoCurriculum': {'name': 'autocurriculum', 'advance_threshold': 0.7, 'regress_threshold': 0.3},
    }

    results = {agent_name: [] for agent_name in agent_configs.keys()}
    all_trajectories = {agent_name: [] for agent_name in agent_configs.keys()}
    per_profile_results = {profile['name']: {agent: {} for agent in agent_configs.keys()} for profile in profiles}

    print("=" * 60)
    print("Running Strategy Comparison Experiment")
    print("=" * 60)
    guard_config = {'guard_delta': 0.08, 'safety_margin': 0.05}

    for profile in profiles:
        print(f"\nProfile: {profile['name']}")
        print("-" * 60)

        for agent_name, config in agent_configs.items():
            print(f"  Agent: {agent_name:12s} ", end='', flush=True)
            trajectories = []

            for run in range(num_runs):
                seed = run * 100
                env = PianoGymEnv(seed=seed)
                agent = get_agent(
                    num_actions=num_actions,
                    seed=seed,
                    guard_delta=0.08,           # Relax the guard slightly to allow actions near the threshold.
                    guard_horizon=1,            # One-step guard horizon, matching the paper setup.
                    guard_safety_margin=0.05,   # One-step fatigue cap: tau - 0.05, aligned with LinUCB.
                    **config
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

            all_trajectories[agent_name].extend(trajectories)

            # Compute metrics.
            metrics = compute_metrics(
                trajectories,
                mastery_window=env.cfg.mastery_window,
                optimal_return=0.0
            )
            results[agent_name].append(metrics)
            clean_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if not key.endswith('_raw') and isinstance(value, (int, float, np.floating, np.integer))
            }
            per_profile_results[profile['name']][agent_name] = clean_metrics

            # Report the 95% CI for each profile over 10 runs.
            # 95% CI = mean ± 1.96 * SEM for large samples, or use the t distribution.
            # For n=10, use t(9, 0.975) ≈ 2.262.
            from scipy import stats
            t_critical = stats.t.ppf(0.975, num_runs - 1) if num_runs > 1 else 1.96
            ttm_ci_half = (metrics['time_to_mastery_std'] / np.sqrt(num_runs)) * t_critical
            print(f"TTM: {metrics['time_to_mastery_mean']:.1f}±{ttm_ci_half:.1f} (95% CI)")

    # Aggregate results across profiles using macro-averaging:
    # first average within each profile, then average across profiles.
    def macro_average(per_profile: list, base_keys: list) -> dict:
        """Macro-average existing ``*_mean`` values across profiles; ``*_std`` stores the 95% CI half-width."""
        out = {}
        P = max(len(per_profile), 1)
        for k in base_keys:
            col = [d.get(f"{k}_mean") for d in per_profile if f"{k}_mean" in d]
            if not col:
                continue
            arr = np.asarray(col, dtype=float)
            out[f"{k}_mean"] = float(np.nanmean(arr))
            # Profile-level 95% CI half-width = t_critical * SEM.
            from scipy import stats
            t_critical = stats.t.ppf(0.975, P - 1) if P > 1 else 1.96
            sem = np.nanstd(arr, ddof=1) / np.sqrt(P) if P > 1 else 0.0
            out[f"{k}_std"] = float(sem * t_critical)
        return out

    # Base metrics that participate in macro-averaging.
    base_keys = [
        "time_to_mastery", "total_reward", "total_reward_raw", "avg_reward_raw",
        "sample_efficiency", "avg_async", "avg_dom_gap", "avg_fid",
        "skill_improvement", "retention_delta", "independence_gain", "regret",
        "avg_fatigue", "avg_constraint_violation", "fatigue_overload_rate",
        "feasible_rate", "steps"
    ]

    final_results = {}
    for agent_name in agent_configs.keys():
        final_results[agent_name] = macro_average(results[agent_name], base_keys)

    # Save data.
    data_dir = Path('output/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison data as JSON.
    with open(data_dir / 'comparison_data.json', 'w') as f:
        json_data = {}
        for agent_name, metrics in final_results.items():
            json_data[agent_name] = {k: float(v) if isinstance(v, (int, float, np.number)) else None
                                    for k, v in metrics.items() if v is not None and not k.endswith('_raw')}
        json.dump(json_data, f, indent=2)

    # Save per-profile metrics for Table 1.
    with open(data_dir / 'comparison_profiles.json', 'w') as f:
        profile_payload = {
            'meta': {
                'num_runs': num_runs,
                'profiles': [profile['name'] for profile in profiles],
            },
            'metrics': per_profile_results,
        }
        json.dump(profile_payload, f, indent=2)

    # Save learning-curve data as JSON.
    with open(data_dir / 'learning_curves_data.json', 'w') as f:
        json_data = {}
        for agent_name, trajs in all_trajectories.items():
            json_data[agent_name] = [
                {
                    'rewards': [float(r) for r in t['rewards']],
                    'raw_rewards': [float(r) for r in t.get('raw_rewards', [])],
                    'mastery_step': int(t.get('mastery_step', 0)),
                }
                for t in trajs
            ]
        json.dump(json_data, f, indent=2)

    # Save text results.
    save_results(final_results, save_path='output/data/comparison_results.txt')

    threshold_sequence = tuple(regret_thresholds) if regret_thresholds else (0.90,)
    for threshold in threshold_sequence:
        clamp_threshold = max(float(threshold), 0.0)
        filter_label = "No feasibility filter" if threshold <= 0 else f"feasible_rate ≥ {threshold:.2f}"

        regret_ttm = compute_regret_table(final_results, objective='ttm', feasibility_threshold=clamp_threshold)
        print(f"\nRegret summary (Time-to-Mastery, lower is better) [{filter_label}]:")
        for agent_name, regret in regret_ttm.items():
            print(f"  {agent_name:12s}: {regret:8.4f}")

        regret_return = compute_regret_table(final_results, objective='return', feasibility_threshold=clamp_threshold)
        print(f"\nRegret summary (Raw return, lower is better) [{filter_label}]:")
        for agent_name, regret in regret_return.items():
            print(f"  {agent_name:12s}: {regret:8.4f}")

    print("\n" + "=" * 60)
    print("Experiment completed! Results saved to output/")
    print("=" * 60)

    return final_results, all_trajectories


def _parse_args():
    parser = argparse.ArgumentParser(description="Strategy comparison experiment")
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs per agent/profile (default: 10)')
    parser.add_argument(
        '--regret-thresholds',
        type=float,
        nargs='+',
        default=[0.90, 0.0],
        help='List of feasible_rate thresholds used for regret tables; values ≤ 0 disable filtering (default: 0.90 0.0)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    results, trajectories = run_experiment(
        num_runs=args.num_runs,
        regret_thresholds=tuple(args.regret_thresholds),
    )
