"""
PianoGym Offline Policy Evaluation (OPE)
Implementation of IPS / SNIPS / DR / FQE (see PianoGym.md §6)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .safety import ExternalSafetyGuard


DEFAULT_GUARD_CONFIG = {"guard_delta": 0.08, "safety_margin": 0.05}


def _ensure_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr, dtype=float)


def _feature_vector(obs: Dict, action: int) -> np.ndarray:
    return np.concatenate([
        _ensure_numpy(obs['context']),
        _ensure_numpy(obs['action_features'])[action]
    ])


def _safe_prob(value: float) -> float:
    return np.clip(float(value), 1e-6, 1.0)


def compute_coverage_metrics(episodes: Sequence[Sequence[Dict]], target_policy=None) -> Dict[str, float]:
    """Compute coverage statistics for the behavior policy.

    Args:
        episodes: logged data
        target_policy: optional target policy used to compute ESS from importance weights

    Returns:
        A dictionary with behavior probability statistics and ESS.
    """
    probs = []
    for episode in episodes:
        for step in episode:
            probs.append(step['behavior_prob'])

    if not probs:
        return {'mean_prob': 0.0, 'min_prob': 0.0, 'max_prob': 0.0, 'ess': 0.0, 'num_steps': 0}

    prob_array = np.asarray(probs, dtype=float)
    sorted_probs = np.sort(prob_array)

    # ESS based on behavior probabilities, used as a coverage indicator.
    ess_den = np.sum(prob_array ** 2) + 1e-8
    ess_behavior = (np.sum(prob_array) ** 2) / ess_den

    result = {
        'mean_prob': float(np.mean(prob_array)),
        'min_prob': float(np.min(prob_array)),
        'max_prob': float(np.max(prob_array)),
        'p01_prob': float(sorted_probs[int(0.01 * (len(sorted_probs) - 1))]) if len(sorted_probs) > 1 else float(sorted_probs[0]),
        'p99_prob': float(sorted_probs[int(0.99 * (len(sorted_probs) - 1))]) if len(sorted_probs) > 1 else float(sorted_probs[-1]),
        'ess': float(ess_behavior),
        'num_steps': int(prob_array.size),
    }

    # If a target policy is provided, also compute ESS from importance weights.
    if target_policy is not None:
        weights = []
        for episode in episodes:
            for step in episode:
                mu = _safe_prob(step['behavior_prob'])
                pi = _safe_prob(target_policy.action_prob(step['obs'], step['action']))
                w = pi / mu
                weights.append(w)

        if weights:
            weight_array = np.asarray(weights, dtype=float)
            # ESS from importance weights: ESS = (sum w)^2 / sum w^2
            ess_weights = (np.sum(weight_array) ** 2) / (np.sum(weight_array ** 2) + 1e-8)
            result['ess_weights'] = float(ess_weights)
            result['mean_weight'] = float(np.mean(weight_array))

    return result


class FittedQEstimator:
    """Least-squares Fitted Q Evaluation."""

    def __init__(self, num_actions: int, gamma: float = 0.99, l2: float = 5e-3, iterations: int = 25):
        self.num_actions = num_actions
        self.gamma = gamma
        self.l2 = l2
        self.iterations = iterations
        self.theta = None
        self.feature_dim = None

    def fit(self, episodes: Sequence[Sequence[Dict]]) -> None:
        transitions = []

        for episode in episodes:
            for step in episode:
                phi = _feature_vector(step['obs'], step['action'])
                reward = step['reward']
                next_obs = step['next_obs']
                done = step['done']
                transitions.append((phi, reward, next_obs, done))

        if not transitions:
            raise ValueError("No transitions available for FQE.")

        self.feature_dim = len(transitions[0][0])
        theta = np.zeros(self.feature_dim)

        for _ in range(self.iterations):
            theta_prev = theta.copy()
            Phi_rows = []
            targets = []

            for phi, reward, next_obs, done in transitions:
                if done or next_obs is None:
                    target = reward
                else:
                    q_next = []
                    for a in range(self.num_actions):
                        next_phi = _feature_vector(next_obs, a)
                        q_next.append(next_phi @ theta_prev)
                    target = reward + self.gamma * np.max(q_next)

                Phi_rows.append(phi)
                targets.append(target)

            Phi = np.vstack(Phi_rows)
            y = np.array(targets)
            A = Phi.T @ Phi + self.l2 * np.eye(self.feature_dim)
            b = Phi.T @ y
            theta = np.linalg.solve(A, b)

        self.theta = theta

    def q_value(self, obs: Dict, action: int) -> float:
        if self.theta is None:
            raise RuntimeError("FittedQEstimator not fitted yet.")
        phi = _feature_vector(obs, action)
        return float(phi @ self.theta)

    def q_vector(self, obs: Dict) -> np.ndarray:
        return np.array([self.q_value(obs, a) for a in range(self.num_actions)])

    def state_value(self, obs: Dict, policy_probs: np.ndarray) -> float:
        q_vals = self.q_vector(obs)
        return float(np.dot(policy_probs, q_vals))


class OPEvaluator:
    """Offline policy evaluator."""

    def __init__(
        self,
        method: str = 'ips',
        num_actions: int | None = None,
        q_estimator: FittedQEstimator | None = None,
        clip_threshold: float | None = None,
    ):
        self.method = method.lower()
        self.num_actions = num_actions
        self.q_estimator = q_estimator
        self.clip_threshold = clip_threshold

        if self.method in {'dr', 'wdr', 'mrdr', 'cdr', 'fqe'} and self.q_estimator is None:
            raise ValueError(f"{self.method} requires a fitted Q estimator.")
        if self.method in {'ips', 'snips', 'dr', 'wdr', 'mrdr', 'wis', 'cips', 'cdr'} and self.num_actions is None:
            raise ValueError(f"{self.method} requires num_actions to be specified.")

    def evaluate(self, episode: Sequence[Dict], target_policy, gamma: float = 0.99) -> float:
        if self.method == 'ips':
            return self._ips(episode, target_policy, gamma)
        if self.method == 'cips':
            return self._ips(episode, target_policy, gamma, clip=self.clip_threshold)
        if self.method == 'snips':
            return self._snips(episode, target_policy, gamma)
        if self.method == 'dr':
            return self._dr(episode, target_policy, gamma)
        if self.method == 'cdr':
            return self._dr(episode, target_policy, gamma, clip=self.clip_threshold)
        if self.method in {'wdr', 'mrdr'}:  # 'mrdr' is kept as a backward-compatible alias.
            return self._wdr(episode, target_policy, gamma)
        if self.method == 'wis':
            return self._wis(episode, target_policy, gamma)
        if self.method == 'fqe':
            return self._fqe(episode, target_policy)
        raise ValueError(f"Unknown OPE method: {self.method}")

    def _ips(self, episode, target_policy, gamma, clip=None):
        # Accumulate in log-space to avoid overflow.
        log_w = 0.0
        total = 0.0
        for t, step in enumerate(episode):
            obs = step['obs']
            action = step['action']
            reward = step['reward']
            mu = _safe_prob(step['behavior_prob'])
            pi = _safe_prob(target_policy.action_prob(obs, action))

            # Log-space accumulation.
            log_ratio = np.log(pi) - np.log(mu)
            if clip is not None:
                log_ratio = min(log_ratio, np.log(clip))
            log_w += log_ratio

            # Apply clipping in log-space.
            if clip is not None:
                log_w = min(log_w, np.log(clip))

            w = np.exp(log_w)
            total += (gamma ** t) * w * reward
        return total

    def _snips(self, episode, target_policy, gamma):
        log_w = 0.0
        numer = 0.0
        denom = 0.0
        for t, step in enumerate(episode):
            obs = step['obs']
            action = step['action']
            reward = step['reward']
            mu = _safe_prob(step['behavior_prob'])
            pi = _safe_prob(target_policy.action_prob(obs, action))

            log_ratio = np.log(pi) - np.log(mu)
            if self.clip_threshold is not None:
                log_ratio = min(log_ratio, np.log(self.clip_threshold))
            log_w += log_ratio
            if self.clip_threshold is not None:
                log_w = min(log_w, np.log(self.clip_threshold))

            w = np.exp(log_w)
            numer += (gamma ** t) * w * reward
            denom += w
        return numer / (denom + 1e-8)

    def _dr(self, episode, target_policy, gamma, clip=None):
        estimator = self.q_estimator
        if estimator is None:
            raise RuntimeError("DR requires a fitted Q estimator.")

        total = 0.0
        log_w_prev = 0.0

        for t, step in enumerate(episode):
            obs = step['obs']
            action = step['action']
            reward = step['reward']
            mu = _safe_prob(step['behavior_prob'])

            pi_probs = target_policy.get_action_probabilities(obs)
            q_hat = estimator.q_value(obs, action)
            v_hat = estimator.state_value(obs, pi_probs)

            pi = _safe_prob(pi_probs[action])

            # Log-space accumulation.
            log_ratio = np.log(pi) - np.log(mu)
            if clip is not None:
                log_ratio = min(log_ratio, np.log(clip))
            log_w = log_w_prev + log_ratio
            if clip is not None:
                log_w = min(log_w, np.log(clip))

            w = np.exp(log_w)
            w_prev = np.exp(log_w_prev)

            total += (gamma ** t) * (w * (reward - q_hat) + w_prev * v_hat)
            log_w_prev = log_w

        return total

    def _wdr(self, episode, target_policy, gamma):
        """Weighted Doubly Robust (WDR)

        Note: this method was previously named MRDR, but the current implementation
        is a WDR-style weighted DR estimator. A true MRDR implementation would
        require a dedicated minimum-variance objective to train the Q-function.
        """
        estimator = self.q_estimator
        if estimator is None:
            raise RuntimeError("WDR requires a fitted Q estimator.")

        # Log-space accumulation.
        log_rho = 0.0
        log_rho_list = []
        for step in episode:
            mu = _safe_prob(step['behavior_prob'])
            pi = _safe_prob(target_policy.action_prob(step['obs'], step['action']))

            log_ratio = np.log(pi) - np.log(mu)
            if self.clip_threshold is not None:
                log_ratio = min(log_ratio, np.log(self.clip_threshold))
            log_rho += log_ratio
            if self.clip_threshold is not None:
                log_rho = min(log_rho, np.log(self.clip_threshold))

            log_rho_list.append(log_rho)

        # Compute the denominator with a log-sum-exp style trick.
        log_rho_arr = np.array(log_rho_list)
        max_log = np.max(log_rho_arr)
        denom = np.sum(np.exp(log_rho_arr - max_log)) * np.exp(max_log)

        total = 0.0
        prev_weight = 1.0

        for t, step in enumerate(episode):
            obs = step['obs']
            action = step['action']
            reward = step['reward']

            pi_probs = target_policy.get_action_probabilities(obs)
            q_hat = estimator.q_value(obs, action)
            v_hat = estimator.state_value(obs, pi_probs)

            rho_t = np.exp(log_rho_list[t])
            w_t = rho_t / (denom + 1e-8)
            total += (gamma ** t) * (w_t * (reward - q_hat) + prev_weight * v_hat)
            prev_weight = w_t
        return total

    def _wis(self, episode, target_policy, gamma):
        log_w = 0.0
        G = 0.0
        for t, step in enumerate(episode):
            obs = step['obs']
            action = step['action']
            reward = step['reward']
            mu = _safe_prob(step['behavior_prob'])
            pi = _safe_prob(target_policy.action_prob(obs, action))

            log_ratio = np.log(pi) - np.log(mu)
            if self.clip_threshold is not None:
                log_ratio = min(log_ratio, np.log(self.clip_threshold))
            log_w += log_ratio
            if self.clip_threshold is not None:
                log_w = min(log_w, np.log(self.clip_threshold))

            G += (gamma ** t) * reward

        w = np.exp(log_w)
        return w * G, w

    def _fqe(self, episode, target_policy):
        estimator = self.q_estimator
        if estimator is None:
            raise RuntimeError("FQE requires a fitted Q estimator.")

        initial_obs = episode[0]['obs']
        pi_probs = target_policy.get_action_probabilities(initial_obs)
        return estimator.state_value(initial_obs, pi_probs)


def collect_logged_data(env, behavior_policy, num_episodes: int = 10, seed: int | None = None,
                        save_path: str | Path | None = None) -> List[List[Dict]]:
    """Collect logged data with a behavior policy and return a list of episodes."""
    # The environment and agent manage their own randomness; ``seed`` is kept only for API compatibility.
    episodes: List[List[Dict]] = []

    for ep in range(num_episodes):
        obs = env.reset()
        behavior_policy.reset()
        done = False
        episode: List[Dict] = []
        guard = ExternalSafetyGuard(env, enabled=True, **DEFAULT_GUARD_CONFIG)
        guard.reset()

        while not done:
            action = behavior_policy.select_action(obs)
            probs = behavior_policy.get_action_probabilities(obs)
            executed_action, guard_decision = guard.enforce(int(action), obs)
            if probs is None:
                behavior_prob = 1.0 / env.cfg.num_actions
            else:
                probs_arr = np.asarray(probs, dtype=float)
                if 0 <= executed_action < probs_arr.size:
                    behavior_prob = float(probs_arr[executed_action])
                else:
                    behavior_prob = 0.0
                if behavior_prob <= 0:
                    behavior_prob = 1e-8

            next_obs, reward, done, info = env.step(executed_action)
            guard.annotate_info(info)
            next_obs_with_done = dict(next_obs)
            next_obs_with_done['done'] = done
            next_obs_with_done['_info'] = info

            episode.append({
                'obs': obs,
                'action': int(executed_action),
                'proposed_action': int(action),
                'reward': float(info.get('raw_reward', reward)),
                'next_obs': next_obs_with_done,
                'behavior_prob': behavior_prob,
                'done': bool(done),
                'info': info,
                'guard': guard_decision.as_dict(),
            })

            behavior_policy.update(obs, executed_action, reward, next_obs_with_done)
            obs = next_obs

        episodes.append(episode)

    behavior_policy.reset()

    if save_path is not None:
        save_logged_data(episodes, save_path)

    return episodes


def _serialize_observation(obs: Dict) -> Dict:
    data = dict(obs)
    data['context'] = _ensure_numpy(obs['context']).tolist()
    data['action_features'] = _ensure_numpy(obs['action_features']).tolist()
    return data


def save_logged_data(episodes: Sequence[Sequence[Dict]], path: str | Path) -> None:
    """Save logged data in JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        for ep_idx, episode in enumerate(episodes):
            for t, step in enumerate(episode):
                record = {
                    'episode': ep_idx,
                    't': t,
                    'obs': _serialize_observation(step['obs']),
                    'action': step['action'],
                    'reward': step['reward'],
                    'next_obs': _serialize_observation(step['next_obs']),
                    'behavior_prob': step['behavior_prob'],
                    'done': step['done'],
                }
                f.write(json.dumps(record) + '\n')


def evaluate_ope_accuracy(env, logged_data, target_policies, true_returns, gamma=0.99):
    """Evaluate OPE accuracy, optionally over multiple logged datasets."""
    if isinstance(logged_data, dict):
        datasets = logged_data.items()
    else:
        datasets = [('default', logged_data)]

    num_actions = env.cfg.num_actions
    overall_results = {}

    for dataset_name, episodes in datasets:
        fqe_estimator = FittedQEstimator(num_actions=num_actions, gamma=gamma)
        fqe_estimator.fit(episodes)

        dataset_results = {
            'coverage': compute_coverage_metrics(episodes),
            'methods': {},
            'weight_stats': {},
        }
        dataset_results['coverage_metric'] = dataset_results['coverage'].get('ess', 0.0)

        clip_cap = 10.0
        p99_clips = []
        for policy_name, policy in target_policies.items():
            policy.reset()
            per_step_weights = []
            final_weights = []
            for episode in episodes:
                rho = 1.0
                for step in episode:
                    mu = _safe_prob(step['behavior_prob'])
                    pi = _safe_prob(policy.action_prob(step['obs'], step['action']))
                    w = pi / mu
                    per_step_weights.append(w)
                    rho *= w
                final_weights.append(rho)
            weights_arr = np.asarray(per_step_weights, dtype=float) if per_step_weights else np.array([1.0])
            final_arr = np.asarray(final_weights, dtype=float) if final_weights else np.array([1.0])
            sorted_weights = np.sort(weights_arr)
            weight_info = {
                'max_weight': float(np.max(weights_arr)),
                'p99_weight': float(sorted_weights[int(0.99 * (len(sorted_weights) - 1))]) if len(sorted_weights) > 1 else float(sorted_weights[0]),
                'mean_weight': float(np.mean(weights_arr)),
                'final_weight_max': float(np.max(final_arr)),
                'final_weight_mean': float(np.mean(final_arr)),
                'num_steps': int(len(weights_arr)),
            }
            dataset_results['weight_stats'][policy_name] = weight_info
            # Only use reasonable p99 values (>= 1.0) so that unusually low outliers do not reduce the global clip.
            if np.isfinite(weight_info['p99_weight']) and weight_info['p99_weight'] >= 1.0:
                p99_clips.append(weight_info['p99_weight'])
            policy.reset()

        # Use the median p99 value as the clip threshold for robustness.
        if p99_clips:
            clip_threshold = min(clip_cap, np.median(p99_clips))
        else:
            clip_threshold = clip_cap
        methods = {
            'ips': OPEvaluator(method='ips', num_actions=num_actions),
            'cips': OPEvaluator(method='cips', num_actions=num_actions, clip_threshold=clip_threshold),
            'snips': OPEvaluator(method='snips', num_actions=num_actions),
            'wis': OPEvaluator(method='wis', num_actions=num_actions, clip_threshold=clip_threshold),
            'wdr': OPEvaluator(method='wdr', num_actions=num_actions, q_estimator=fqe_estimator),
            'dr': OPEvaluator(method='dr', num_actions=num_actions, q_estimator=fqe_estimator),
            'cdr': OPEvaluator(method='cdr', num_actions=num_actions, q_estimator=fqe_estimator, clip_threshold=clip_threshold),
            'fqe': OPEvaluator(method='fqe', num_actions=num_actions, q_estimator=fqe_estimator),
        }

        for method_name, evaluator in methods.items():
            estimates = {}
            for policy_name, policy in target_policies.items():
                episode_estimates = []
                policy.reset()
                if evaluator.method == 'wis':
                    weight_sum = 0.0
                    weighted_returns = 0.0
                    for episode in episodes:
                        wr, wt = evaluator.evaluate(episode, policy, gamma=gamma)
                        weighted_returns += wr
                        weight_sum += wt
                    estimate = weighted_returns / (weight_sum + 1e-8)
                else:
                    for episode in episodes:
                        estimate = evaluator.evaluate(episode, policy, gamma=gamma)
                        episode_estimates.append(estimate)
                    estimate = float(np.mean(episode_estimates)) if episode_estimates else 0.0
                policy.reset()
                estimates[policy_name] = float(estimate)

            errors = []
            for policy_name in target_policies.keys():
                true_val = true_returns.get(policy_name, 0.0)
                est_val = estimates.get(policy_name, 0.0)
                errors.append(est_val - true_val)

            errors = np.asarray(errors, dtype=float)
            mse = float(np.mean(errors ** 2))
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(mse))
            dataset_results['methods'][method_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'estimates': estimates,
            }

        overall_results[dataset_name] = dataset_results

    return overall_results
