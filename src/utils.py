"""
PianoGym Utilities
Metric computation and plotting utilities (see PianoGym.md §5)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .metrics import compute_metrics as _compute_metrics, compute_regret_table


def compute_metrics(trajectories, mastery_window=3, optimal_return=0.0):
    """Wrap ``metrics.compute_metrics`` while preserving the original signature."""
    return _compute_metrics(trajectories, mastery_window=mastery_window, optimal_return=optimal_return)


def plot_comparison(results, save_path='output/comparison.png'):
    """Plot the strategy comparison figure (Time-to-Mastery / Total Reward / Independence Gain)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    agents = list(results.keys())
    ttm_means = [results[a].get('time_to_mastery_mean', 0.0) for a in agents]
    ttm_stds = [results[a].get('time_to_mastery_std', 0.0) for a in agents]

    reward_means = [results[a].get('total_reward_raw_mean', results[a].get('total_reward_mean', 0.0)) for a in agents]
    reward_stds = [results[a].get('total_reward_raw_std', results[a].get('total_reward_std', 0.0)) for a in agents]

    indep_means = [results[a].get('independence_gain_mean', 0.0) for a in agents]
    indep_stds = [results[a].get('independence_gain_std', 0.0) for a in agents]

    # Time-to-Mastery
    axes[0].bar(agents, ttm_means, yerr=ttm_stds, capsize=5, alpha=0.7)
    axes[0].set_ylabel('Time-to-Mastery (steps)')
    axes[0].set_title('Time-to-Mastery Comparison')
    axes[0].grid(axis='y', alpha=0.3)

    # Total Reward
    axes[1].bar(agents, reward_means, yerr=reward_stds, capsize=5, alpha=0.7, color='orange')
    axes[1].set_ylabel('Total Raw Reward')
    axes[1].set_title('Total Raw Reward Comparison')
    axes[1].grid(axis='y', alpha=0.3)

    # Independence Gain
    axes[2].bar(agents, indep_means, yerr=indep_stds, capsize=5, alpha=0.7, color='seagreen')
    axes[2].set_ylabel('Independence Gain')
    axes[2].set_title('Coordination Improvement')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_learning_curves(histories, save_path='output/learning_curves.png'):
    """Plot learning curves (reward over time).
    histories: dict of {agent_name: list of trajectories}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for agent_name, trajs in histories.items():
        series = []
        for traj in trajs:
            rewards = np.asarray(traj.get('raw_rewards', traj.get('rewards', [])), dtype=float)
            if rewards.size == 0:
                continue
            mastery_step = int(traj.get('mastery_step', 0))
            if mastery_step > 0:
                rewards = rewards[:min(mastery_step, rewards.size)]
            window = min(5, rewards.size)
            if window <= 1:
                smoothed = rewards.astype(float)
            elif rewards.size >= window:
                kernel = np.ones(window, dtype=float) / window
                smoothed = np.convolve(rewards, kernel, mode='valid')
            else:
                smoothed = rewards.astype(float)
            if smoothed.size:
                series.append(smoothed)
        if not series:
            continue
        min_len = min(len(s) for s in series if len(s) > 0)
        if min_len == 0:
            continue
        aligned = np.stack([s[:min_len] for s in series], axis=0)
        mean_curve = np.nanmean(aligned, axis=0)
        std_curve = np.nanstd(aligned, axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean_curve, label=agent_name, linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Sliding Avg Raw Reward')
    ax.set_title('Time-Aligned Learning Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_ope_results(ope_errors, save_path='output/ope_comparison.png'):
    """Plot OPE error curves for three metrics.

    Expected input structure:
    {
        dataset_name: {
            'coverage_metric': float (ESS or another coverage metric),
            'coverage': {...},
            'methods': {
                method: {'mse': float, 'mae': float, 'rmse': float, ...}
            }
        }
    }
    """
    if not ope_errors:
        return
    metrics = ['mse', 'mae', 'rmse']
    metric_titles = {'mse': 'MSE', 'mae': 'MAE', 'rmse': 'RMSE'}

    datasets = sorted(ope_errors.keys(), key=lambda name: ope_errors[name].get('coverage_metric', 0.0))
    methods_order = list(next(iter(ope_errors.values()))['methods'].keys())
    method_labels = [m.upper() for m in methods_order]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4), sharey=False, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    x_positions = np.arange(len(methods_order))
    xtick_labels = method_labels

    coverage_points = {m.upper(): {'coverage': [], 'rmse': []} for m in methods_order}
    dataset_labels = {}

    for dataset in datasets:
        coverage = ope_errors[dataset].get('coverage', {})
        ess = ope_errors[dataset].get('coverage_metric', coverage.get('ess', 0.0))
        mean_p = coverage.get('mean_prob', 0.0)
        dataset_labels[dataset] = f"{dataset} (ESS={ess:.0f}, μp={mean_p:.3f})"
        for method_key in methods_order:
            vals = ope_errors[dataset]['methods'][method_key]
            coverage_points[method_key.upper()]['coverage'].append(ess)
            coverage_points[method_key.upper()]['rmse'].append(vals.get('rmse', np.sqrt(vals.get('mse', 0.0))))

    for ax, metric in zip(axes, metrics):
        for dataset in datasets:
            values = [ope_errors[dataset]['methods'][method_key][metric] for method_key in methods_order]
            ax.plot(x_positions, values, marker='o', label=dataset_labels[dataset])
        ax.set_title(metric_titles[metric])
        ax.set_ylabel(metric_titles[metric])
        ax.grid(alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xtick_labels)

    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Coverage vs RMSE curve (ESS on the x-axis)
    if coverage_points:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for method_label, vals in coverage_points.items():
            coverage_vals = np.array(vals['coverage'])
            rmse_vals = np.array(vals['rmse'])
            order = np.argsort(coverage_vals)
            ax2.plot(coverage_vals[order], rmse_vals[order], marker='o', label=method_label)
        ax2.set_xlabel('Effective Sample Size (ESS)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('OPE RMSE vs Coverage')
        ax2.grid(alpha=0.3)
        ax2.legend()
        coverage_path = Path(save_path).with_name(Path(save_path).stem + '_coverage.png')
        fig2.savefig(coverage_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {coverage_path}")
        plt.close(fig2)


def save_results(results, save_path='output/results.txt'):
    """Save results as plain text."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        for agent_name, metrics in results.items():
            f.write(f"\n{'='*50}\n")
            f.write(f"Agent: {agent_name}\n")
            f.write(f"{'='*50}\n")
            for key, value in metrics.items():
                if not key.endswith('_raw'):
                    f.write(f"  {key:25s}: {value:8.3f}\n")

    print(f"Saved: {save_path}")
