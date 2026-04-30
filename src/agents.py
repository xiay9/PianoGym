"""
PianoGym Agents: PianoMPC, BayesianMAB, Thompson, LinUCB, DQN, CCB-DF, AutoCurriculum, Safe-AC
"""
import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    """Base policy class."""
    def __init__(self, num_actions, seed=None):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self._last_metadata = {}
        self.reset()

    @abstractmethod
    def select_action(self, obs):
        """Select an action given the observation."""
        pass

    @abstractmethod
    def update(self, obs, action, reward, next_obs):
        """Update policy parameters."""
        pass

    def reset(self):
        """Reset policy state."""
        pass

    def get_action_probabilities(self, obs):
        """Return the behavior distribution p(a|obs), uniform by default."""
        return np.ones(self.num_actions) / self.num_actions

    def action_prob(self, obs, action):
        probs = self.get_action_probabilities(obs)
        return probs[action]

    def _reset_metadata(self):
        """Clear per-step instrumentation metadata."""
        self._last_metadata = {}

    def _store_metadata(self, **metadata):
        """Record instrumentation details for the latest decision."""
        self._last_metadata = dict(metadata)

    def get_last_metadata(self):
        """Return metadata recorded during the previous action selection."""
        return dict(self._last_metadata)

    def set_guard_params(self, horizon=None, delta=None, safety_margin=None):
        """Override in subclasses that expose guard configuration."""
        return


class LinUCB(Agent):
    """LinUCB (linear contextual bandit) (see PianoGym.md §9).

    Fatigue penalty mechanism:
        penalty(a) = lambda_fatigue * max(0, f_pred(a) - threshold)

        ``lambda_fatigue`` is automatically rescaled to match the magnitude of
        the UCB score. The user-facing lambda should usually be in [0, 10] on a
        normalized scale; internally it is multiplied by ``scaling_factor``
        (default 50) to match the unnormalized UCB score.
    """
    def __init__(
        self,
        num_actions,
        context_dim,
        alpha=1.0,
        ridge=1e-3,
        lambda_fatigue=0.0,
        fatigue_threshold=0.75,
        rest_recovery=0.22,
        lambda_scaling=50.0,
        enable_guard=True,
        use_sherman_morrison=True,
        seed=None,
    ):
        self.d = context_dim
        self.alpha = alpha
        self.ridge = ridge
        # ``lambda_fatigue`` is user-facing and normalized; it is rescaled internally.
        self.lambda_fatigue_normalized = lambda_fatigue
        self.lambda_scaling = lambda_scaling
        self.lambda_fatigue = lambda_fatigue * lambda_scaling  # Effective lambda used internally.
        self.fatigue_threshold = fatigue_threshold
        self.rest_recovery = rest_recovery
        self.enable_guard = enable_guard  # Toggle the hard safety guard.
        self.guard_horizon = 3
        self.guard_delta = 0.0
        self.guard_safety_margin = 0.10
        self.use_sherman_morrison = use_sherman_morrison  # Toggle Sherman-Morrison optimization.
        super().__init__(num_actions, seed)

    def _predict_next_fatigue(self, current_fatigue, action_features):
        """Predict the next-step fatigue after taking action ``a``.

        Based on fatigue dynamics:
        f_{t+1} = clip(f_t + beta_a - rho * 1_{a=REST}, 0, 1)

        ``action_features`` layout:
        [q_vec (K dims), difficulty, coord_focus, beta, duration, type_idx_norm, transfer_scale]
        Assuming K=5, ``beta`` is at index 7.
        """
        # beta_a is stored at K + 2 in the feature vector:
        # K q-values + difficulty + coord_focus.
        # Assuming K=5, beta is at index 7.
        K = 5  # Skill dimension; should stay aligned with the environment.
        beta_idx = K + 2

        if len(action_features) > beta_idx:
            beta_a = action_features[beta_idx]
        else:
            beta_a = 0.1  # Default fallback value.

        # REST uses beta=0.0 and reduces fatigue.
        # Detect it via the duration feature (REST uses duration=1.0) or beta.
        duration_idx = beta_idx + 1
        is_rest = False
        if len(action_features) > duration_idx:
            # REST signature: beta=0 and duration=1.0.
            duration = action_features[duration_idx]
            is_rest = (abs(beta_a) < 1e-6 and abs(duration - 1.0) < 1e-6)

        if is_rest:
            return np.clip(current_fatigue - self.rest_recovery, 0.0, 1.0)
        else:
            return np.clip(current_fatigue + beta_a, 0.0, 1.0)

    def set_guard_params(self, horizon=None, delta=None, safety_margin=None):
        if horizon is not None:
            self.guard_horizon = max(1, int(horizon))
        if delta is not None:
            self.guard_delta = max(0.0, float(delta))
        if safety_margin is not None:
            self.guard_safety_margin = float(safety_margin)

    def _predict_peak(self, current_fatigue, action_features):
        f = current_fatigue
        peak = f
        for step in range(self.guard_horizon):
            feat = action_features
            f = self._predict_next_fatigue(f, feat)
            peak = max(peak, f)
        return peak

    def select_action(self, obs):
        self._reset_metadata()
        fatigue_est = float(obs.get('fatigue_est', obs.get('fatigue', 0.0)))
        action_features_list = obs.get('action_features', [])

        # Read the effective threshold from the environment at runtime.
        actual_threshold = obs.get('fatigue_threshold', self.fatigue_threshold)
        guard_threshold = actual_threshold + self.guard_delta

        safety_margin = self.guard_safety_margin
        K = 5
        beta_idx = K + 2

        candidates = []

        for a in range(self.num_actions):
            action_feat = action_features_list[a] if a < len(action_features_list) else np.zeros(1)
            f_next = self._predict_next_fatigue(fatigue_est, action_feat)
            peak_guard = self._predict_peak(f_next, action_feat)

            phi = self._phi(obs, a)
            if self.use_sherman_morrison:
                A_inv = self.A_inv[a]
            else:
                A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = phi @ theta + self.alpha * np.sqrt(max(phi @ A_inv @ phi, 0.0))

            violation = max(0.0, f_next - actual_threshold)
            penalty = self.lambda_fatigue * violation if self.lambda_fatigue > 0 else 0.0
            score = ucb - penalty

            if len(action_feat) > beta_idx:
                beta_a = action_feat[beta_idx]
            else:
                beta_a = 0.0

            candidates.append({
                'action': a,
                'score': score,
                'ucb': ucb,
                'penalty': penalty,
                'f_next': f_next,
                'peak_guard': peak_guard,
                'beta': beta_a,
            })

        base_best = max(candidates, key=lambda c: c['score'])

        if self.enable_guard:
            valid_candidates = [
                c for c in candidates
                if (c['f_next'] <= actual_threshold - safety_margin)
                and (c['peak_guard'] <= guard_threshold)
            ]
            guard_filtered = len(valid_candidates) < len(candidates)
            if not valid_candidates:
                # Fall back to the action with the lowest predicted peak fatigue.
                chosen = min(candidates, key=lambda c: c['peak_guard'])
            else:
                chosen = max(valid_candidates, key=lambda c: c['score'])
        else:
            guard_filtered = False
            chosen = base_best

        guard_pass = chosen['peak_guard'] <= guard_threshold

        self._store_metadata(
            selected_action=int(chosen['action']),
            base_action=int(base_best['action']),
            guard_enabled=bool(self.enable_guard),
            guard_filtered=bool(guard_filtered),
            guard_pass=bool(guard_pass),
            predicted_f_next=float(chosen['f_next']),
            predicted_peak_guard=float(chosen['peak_guard']),
            guard_threshold=float(guard_threshold),
            will_block=bool(not guard_pass),
            fatigue_threshold=float(actual_threshold),
            safety_margin=float(safety_margin),
            lambda_t=float(self.lambda_fatigue),
            penalty=float(chosen['penalty']),
        )

        return int(chosen['action'])

    def update(self, obs, action, reward, next_obs):
        phi = self._phi(obs, action)
        if self.use_sherman_morrison:
            # Sherman-Morrison incremental update:
            # (A + uu^T)^-1 = A^-1 - (A^-1 u u^T A^-1) / (1 + u^T A^-1 u)
            A_inv = self.A_inv[action]
            u = phi
            Au = A_inv @ u
            denom = 1.0 + u @ Au
            if abs(denom) > 1e-8:  # Numerical stability check.
                self.A_inv[action] = A_inv - np.outer(Au, Au) / denom
            else:
                # Degenerate case: recompute the full inverse.
                self.A[action] += np.outer(phi, phi)
                self.A_inv[action] = np.linalg.inv(self.A[action])
        else:
            self.A[action] += np.outer(phi, phi)
        self.b[action] += reward * phi

    def reset(self):
        self._reset_metadata()
        self.A = [np.eye(self.d) * self.ridge for _ in range(self.num_actions)]
        self.b = [np.zeros(self.d) for _ in range(self.num_actions)]
        if self.use_sherman_morrison:
            # Precompute and maintain the inverse matrices.
            self.A_inv = [np.eye(self.d) / self.ridge for _ in range(self.num_actions)]

    def _phi(self, obs, action):
        return np.concatenate([obs['context'], obs['action_features'][action]])

    def get_action_probabilities(self, obs):
        scores = []
        fatigue_est = float(obs.get('fatigue_est', obs.get('fatigue', 0.0)))
        action_features_list = obs.get('action_features', [])
        actual_threshold = obs.get('fatigue_threshold', self.fatigue_threshold)

        for a in range(self.num_actions):
            phi = self._phi(obs, a)
            if self.use_sherman_morrison:
                A_inv = self.A_inv[a]
            else:
                A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = phi @ theta + self.alpha * np.sqrt(max(phi @ A_inv @ phi, 0.0))

            # Action-specific fatigue penalty, kept consistent with ``select_action``.
            if self.lambda_fatigue > 0:
                action_feat = action_features_list[a] if a < len(action_features_list) else np.zeros(1)
                f_next = self._predict_next_fatigue(fatigue_est, action_feat)
                violation = max(0.0, f_next - actual_threshold)
                penalty = self.lambda_fatigue * violation
            else:
                penalty = 0.0

            scores.append(ucb - penalty)

        probs = np.zeros(self.num_actions)
        probs[int(np.argmax(scores))] = 1.0
        return probs

class ThompsonSampling(Agent):
    """Thompson Sampling (linear Gaussian) (see PianoGym.md §9)."""
    def __init__(self, num_actions, context_dim, v=1.0, ridge=1e-3, seed=None):
        self.d = context_dim
        self.v = v  # Noise variance.
        self.ridge = ridge
        super().__init__(num_actions, seed)

    def select_action(self, obs):
        samples = []

        for a in range(self.num_actions):
            phi = self._phi(obs, a)
            A_inv = np.linalg.inv(self.A[a])
            mean = A_inv @ self.b[a]
            cov = (self.v ** 2) * A_inv
            theta_sample = self.rng.multivariate_normal(mean, cov)
            samples.append(phi @ theta_sample)

        return np.argmax(samples)

    def update(self, obs, action, reward, next_obs):
        phi = self._phi(obs, action)
        self.A[action] += np.outer(phi, phi)
        self.b[action] += reward * phi

    def reset(self):
        self.A = [np.eye(self.d) * self.ridge for _ in range(self.num_actions)]
        self.b = [np.zeros(self.d) for _ in range(self.num_actions)]

    def _phi(self, obs, action):
        return np.concatenate([obs['context'], obs['action_features'][action]])

    def get_action_probabilities(self, obs):
        scores = []
        for a in range(self.num_actions):
            phi = self._phi(obs, a)
            A_inv = np.linalg.inv(self.A[a])
            mean = A_inv @ self.b[a]
            scores.append(phi @ mean)
        probs = np.zeros(self.num_actions)
        probs[int(np.argmax(scores))] = 1.0
        return probs

class DQNAgent(Agent):
    """Simplified DQN with a two-layer MLP and replay buffer."""
    def __init__(self, num_actions, state_dim, hidden_dim=64, gamma=0.99,
                 lr=1e-3, epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.995,
                 buffer_size=5000, batch_size=32, min_buffer_before_train=200,
                 tau=0.01, grad_clip=1.0, seed=None):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_buffer_before_train = min_buffer_before_train
        self.tau = tau
        self.grad_clip = grad_clip
        super().__init__(num_actions, seed)

    def reset(self):
        self.W1 = self.rng.normal(0, 0.1, size=(self.state_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self.rng.normal(0, 0.1, size=(self.hidden_dim, self.num_actions))
        self.b2 = np.zeros(self.num_actions)
        self.memory = []
        self.epsilon_current = self.epsilon
        self.tW1 = self.W1.copy()
        self.tb1 = self.b1.copy()
        self.tW2 = self.W2.copy()
        self.tb2 = self.b2.copy()

    def _state_vec(self, obs):
        state = np.asarray(obs['context'], dtype=float)
        # Dynamically adapt to the expected dimension when the environment changes the context size.
        if len(state) != self.state_dim:
            new_state = np.zeros(self.state_dim)
            copy_len = min(len(state), self.state_dim)
            new_state[:copy_len] = state[:copy_len]
            return new_state
        return state

    def _forward(self, state):
        h = np.tanh(state @ self.W1 + self.b1)
        q = h @ self.W2 + self.b2
        return h, q

    def select_action(self, obs):
        state = self._state_vec(obs)
        _, q_values = self._forward(state)

        if self.rng.random() < self.epsilon_current:
            return int(self.rng.integers(0, self.num_actions))
        return int(np.argmax(q_values))

    def update(self, obs, action, reward, next_obs):
        state = self._state_vec(obs)
        next_state = self._state_vec(next_obs)
        done = next_obs.get('done', False)

        transition = (state, action, reward, next_state, done)
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        self.memory.append(transition)

        self._train_step()
        self.epsilon_current = max(self.epsilon_min, self.epsilon_current * self.epsilon_decay)

    def _train_step(self):
        if len(self.memory) < max(self.batch_size, self.min_buffer_before_train):
            return

        idx = self.rng.choice(len(self.memory), size=self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.memory[i] for i in idx))
        states = np.stack(states)
        actions = np.array(actions, dtype=int)
        rewards = np.array(rewards, dtype=float)
        next_states = np.stack(next_states)
        dones = np.array(dones, dtype=bool)

        h, q_values = self._forward(states)
        _, next_q = self._forward_target(next_states)
        next_max = np.max(next_q, axis=1)
        targets = rewards + self.gamma * next_max * (~dones)

        td_errors = q_values[np.arange(self.batch_size), actions] - targets
        grad_q = np.zeros_like(q_values)
        grad_q[np.arange(self.batch_size), actions] = td_errors

        grad_W2 = h.T @ grad_q / self.batch_size
        grad_b2 = np.mean(grad_q, axis=0)

        grad_h = grad_q @ self.W2.T
        grad_z1 = grad_h * (1 - h ** 2)

        grad_W1 = states.T @ grad_z1 / self.batch_size
        grad_b1 = np.mean(grad_z1, axis=0)

        np.clip(grad_W2, -self.grad_clip, self.grad_clip, out=grad_W2)
        np.clip(grad_b2, -self.grad_clip, self.grad_clip, out=grad_b2)
        np.clip(grad_W1, -self.grad_clip, self.grad_clip, out=grad_W1)
        np.clip(grad_b1, -self.grad_clip, self.grad_clip, out=grad_b1)

        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

        self.tW1 = (1 - self.tau) * self.tW1 + self.tau * self.W1
        self.tb1 = (1 - self.tau) * self.tb1 + self.tau * self.b1
        self.tW2 = (1 - self.tau) * self.tW2 + self.tau * self.W2
        self.tb2 = (1 - self.tau) * self.tb2 + self.tau * self.b2

    def _forward_target(self, state):
        h = np.tanh(state @ self.tW1 + self.tb1)
        q = h @ self.tW2 + self.tb2
        return h, q

    def get_action_probabilities(self, obs):
        state = self._state_vec(obs)
        _, q_values = self._forward(state)
        greedy_actions = np.flatnonzero(q_values == np.max(q_values))
        probs = np.full(self.num_actions, self.epsilon_current / self.num_actions)
        if len(greedy_actions) == 0:
            return probs
        bonus = (1 - self.epsilon_current) / len(greedy_actions)
        for a in greedy_actions:
            probs[a] += bonus
        return probs

class ModelPredictivePlanner(Agent):
    """Simplified PianoMPC with short-horizon prediction and candidate-sequence scoring."""
    def __init__(self, num_actions, horizon=3, pool_size=6,
                 reward_weights=None, reward_norm=None,
                 fatigue_limit=0.75, time_weight=0.01,
                 fatigue_penalty=0.25, safety_margin=0.05,
                 guard_delta=0.08, guard_horizon=1,
                 rest_recovery=0.25, seed=None):
        self.horizon = horizon
        self.pool_size = pool_size
        self.reward_weights = np.array(reward_weights or [1.0, 0.8, 1.5])
        self.reward_norm = reward_norm or {'async': 80.0, 'dom_gap': 30.0}
        self.fatigue_limit = fatigue_limit
        self.time_weight = time_weight
        self.fatigue_penalty = fatigue_penalty
        self.safety_margin = safety_margin  # One-step fatigue cap: tau - safety_margin.
        self.guard_delta = guard_delta      # Peak fatigue cap: tau + guard_delta.
        self.guard_horizon = guard_horizon  # Guard lookahead horizon.
        self.rest_recovery = float(rest_recovery)
        self.skill_dim = None
        self.skill_est = None
        self.fatigue_est = 0.0
        self.retention_est = 0.5
        self.rest_action = num_actions - 1
        super().__init__(num_actions, seed)

    def reset(self):
        self._reset_metadata()
        if self.skill_dim is not None:
            self.skill_est = np.full(self.skill_dim, 0.25)
        else:
            self.skill_est = None
        self.fatigue_est = 0.0
        self.retention_est = 0.5

    def set_guard_params(self, horizon=None, delta=None, safety_margin=None):
        """Set guard parameters."""
        if horizon is not None:
            self.guard_horizon = max(1, int(horizon))
        if delta is not None:
            self.guard_delta = max(0.0, float(delta))
        if safety_margin is not None:
            self.safety_margin = float(safety_margin)

    def _predict_fnext(self, fatigue, feat):
        """Predict fatigue after executing an action."""
        K = self.skill_dim
        beta = float(feat[K + 2]) if len(feat) > K + 2 else 0.1
        duration = float(feat[K + 3]) if len(feat) > K + 3 else 0.0
        is_rest = (abs(beta) < 1e-6 and abs(duration - 1.0) < 1e-6)
        if is_rest:
            return np.clip(fatigue - self.rest_recovery, 0, 1)
        return np.clip(fatigue + beta, 0, 1)

    def select_action(self, obs):
        self._reset_metadata()
        features = np.asarray(obs['action_features'])
        self._ensure_dimensions(features)

        fatigue_est = obs.get('fatigue_est', self.fatigue_est)

        # 1) Use the actual environment threshold.
        actual_threshold = obs.get('fatigue_threshold', self.fatigue_limit)
        self._current_threshold = actual_threshold

        guard_threshold = actual_threshold + self.guard_delta   # Peak cap tau + delta.
        fnext_cap = actual_threshold - self.safety_margin       # One-step cap tau - epsilon.

        candidate_idxs = self._candidate_pool(features)

        safe_idxs = []          # All safe actions, including REST.
        safe_non_rest = []      # Safe actions excluding REST.
        best_viol = None        # (viol_score, idx, f_next, peak)

        # 2) Classify all candidates in one pass.
        for idx in candidate_idxs:
            feat = features[idx]
            f_next = self._predict_fnext(fatigue_est, feat)

            # Multi-step peak lookahead.
            peak = f_next
            if self.guard_horizon > 1:
                f_tmp = f_next
                for _ in range(self.guard_horizon - 1):
                    f_tmp = self._predict_fnext(f_tmp, feat)
                    peak = max(peak, f_tmp)

            is_safe = (peak <= guard_threshold) and (f_next <= fnext_cap)

            if is_safe:
                safe_idxs.append(idx)
                if idx != self.rest_action:
                    safe_non_rest.append(idx)
            else:
                # Violation score: one-step overflow + 0.5 * peak overflow.
                viol_score = max(0.0, f_next - fnext_cap) + 0.5 * max(0.0, peak - guard_threshold)
                if (best_viol is None) or (viol_score < best_viol[0]):
                    best_viol = (viol_score, idx, f_next, peak)

        # 3) Decision logic.
        if safe_non_rest:
            # If there are safe non-REST actions, choose the one with the best rollout score.
            best_score = -np.inf
            chosen_action = safe_non_rest[0]
            for idx in safe_non_rest:
                score = self._simulate(idx, features, fatigue_est)
                if score > best_score:
                    best_score = score
                    chosen_action = idx
            guard_triggered = False

        elif safe_idxs:
            # Only REST remains safe.
            rest_idx = self.rest_action
            if (best_viol is not None):
                # Compare whether resting now is better than allowing a slight violation.
                rest_score = self._simulate(rest_idx, features, fatigue_est)
                viol_score = self._simulate(best_viol[1], features, fatigue_est)
                if viol_score > rest_score:
                    chosen_action = best_viol[1]
                    guard_triggered = True   # Intentionally relaxed the guard in this case.
                else:
                    chosen_action = rest_idx
                    guard_triggered = False
            else:
                # REST is truly the only safe option.
                chosen_action = rest_idx
                guard_triggered = False

        else:
            # No safe action exists; choose the least violating one.
            if best_viol is not None:
                chosen_action = best_viol[1]
            else:
                chosen_action = self.rest_action
            guard_triggered = True

        # 4) Metadata.
        selected_feat = features[chosen_action]
        predicted_f_next = float(self._predict_fnext(fatigue_est, selected_feat))

        self._store_metadata(
            selected_action=int(chosen_action),
            guard_enabled=True,
            guard_triggered=bool(guard_triggered),
            guard_pass=not guard_triggered,
            predicted_f_next=predicted_f_next,
            guard_threshold=float(guard_threshold),
            fnext_cap=float(fnext_cap),
            fatigue_threshold=float(actual_threshold),
            safety_margin=float(self.safety_margin),
        )

        return chosen_action

    def update(self, obs, action, reward, next_obs):
        # 1) Make sure we understand the feature layout.
        features = np.asarray(obs['action_features'])
        self._ensure_dimensions(features)

        # 2) Read the features of the action actually executed at this step.
        feat = features[action]
        q_vec = feat[:self.skill_dim]
        difficulty = feat[self.skill_dim]

        # Base learning rate: higher difficulty implies slower improvement.
        alpha = 0.05 + 0.03 * difficulty

        # 3) Prefer the true fatigue from env info; fall back to the estimate otherwise.
        info_pkg = next_obs.get('_info') if isinstance(next_obs, dict) else None
        if info_pkg is not None and 'fatigue' in info_pkg:
            measured_f = float(info_pkg['fatigue'])
        else:
            measured_f = float(next_obs.get('fatigue_est', self.fatigue_est))

        # 4) Skill update: keep the original logic, but use the internal fatigue estimate.
        fatigue_factor = max(0.3, 1 - self.fatigue_est)
        gain = alpha * (1 - self.skill_est) * q_vec * fatigue_factor

        if action == self.rest_action:
            # Rest does not improve skills; it only clips away numerical drift.
            self.skill_est = np.clip(self.skill_est, 0, 1)
        else:
            self.skill_est = np.clip(self.skill_est + gain, 0, 1)

        # 5) Update the internal fatigue estimate with an EMA toward the measured value.
        self.fatigue_est = np.clip(
            0.7 * self.fatigue_est + 0.3 * measured_f,
            0.0,
            1.0
        )

        # 6) If this step was REST, apply one extra deterministic recovery update.
        if action == self.rest_action:
            self.fatigue_est = np.clip(self.fatigue_est - self.rest_recovery, 0.0, 1.0)

        # 7) Retention follows the same rule: prefer the measured value when available.
        if info_pkg is not None and 'retention' in info_pkg:
            measured_r = float(info_pkg['retention'])
        else:
            measured_r = float(next_obs.get('retention_est', self.retention_est))

        self.retention_est = np.clip(
            0.85 * self.retention_est + 0.15 * measured_r,
            0.0,
            1.0
        )

    def _ensure_dimensions(self, features):
        if self.skill_dim is None:
            self.skill_dim = features.shape[1] - 6
            self.skill_est = np.full(self.skill_dim, 0.25)

    def _candidate_pool(self, features):
        gains = []
        residual = 1 - self.skill_est
        for idx in range(self.num_actions):
            if idx == self.rest_action:
                continue
            q_vec = features[idx][:self.skill_dim]
            difficulty = features[idx][self.skill_dim]
            score = np.dot(residual, q_vec) - 0.2 * difficulty
            gains.append((score, idx))
        gains.sort(reverse=True)
        top = [idx for _, idx in gains[:max(1, self.pool_size)]]
        if self.rest_action not in top:
            top.append(self.rest_action)
        return top

    def _simulate(self, first_idx, features, fatigue_est):
        # Copy the internal state for virtual rollouts.
        skill = self.skill_est.copy()
        fatigue = float(fatigue_est)
        retention = float(self.retention_est)
        total = 0.0
        idx = int(first_idx)

        # Use the same asynchrony weights as the environment, truncating if needed.
        base_w_async = np.array([0.4, 0.4, 0.1, 0.1, 0.0], dtype=float)
        if self.skill_dim < base_w_async.size:
            w_async = base_w_async[:self.skill_dim]
        else:
            # If there are more than five skill dimensions, assign small tail weights.
            pad = np.full(self.skill_dim - base_w_async.size, 0.05, dtype=float)
            w_async = np.concatenate([base_w_async, pad], axis=0)

        for step in range(self.horizon):
            feat = features[idx]
            # Basic feature unpacking.
            q_vec       = feat[:self.skill_dim]
            difficulty  = float(feat[self.skill_dim])
            coord_focus = float(feat[self.skill_dim + 1])
            beta        = float(feat[self.skill_dim + 2])
            duration    = float(feat[self.skill_dim + 3])

            # 1) Build exercise-specific rhythm demand and a left/right amplification factor.
            # Without env-level ``phi_a`` / ``psi_a``, infer separable proxies from the features.
            phi_a = 60.0 + 40.0 * difficulty              # 60 to 100; higher difficulty is stricter.
            psi_a = 15.0 + 20.0 * abs(coord_focus)        # 15 to 35; stronger hand bias increases dom gap.

            # 2) Virtual skill and fatigue evolution.
            alpha = 0.05 + 0.03 * difficulty
            if idx == self.rest_action:
                # Rest.
                fatigue = np.clip(fatigue - self.rest_recovery, 0.0, 1.0)
                skill_gain = np.zeros_like(skill)
            else:
                # Practice.
                fatigue = np.clip(fatigue + beta, 0.0, 1.0)
                fatigue_factor = max(0.3, 1.0 - fatigue)
                skill_gain = alpha * (1.0 - skill) * q_vec * fatigue_factor

            skill = np.clip(skill + skill_gain, 0.0, 1.0)
            retention = np.clip(0.85 * retention + 0.15 * float(np.mean(skill)), 0.0, 1.0)

            # 3) Generate a synthetic observation.
            async_est = max(
                20.0,
                phi_a * (1.0 - float(np.dot(w_async, skill))) * (1.0 + 0.3 * fatigue)
            )

            if self.skill_dim >= 2:
                dom_est = psi_a * (skill[1] - skill[0])
            else:
                dom_est = 0.0

            # Fidelity proxy: higher q @ x and lower difficulty/fatigue are better.
            fid_est = np.clip(
                np.dot(q_vec, skill) - 0.25 * difficulty - 0.30 * fatigue + 0.25,
                0.0,
                1.0
            )

            # 4) Compute the penalty using the current true threshold.
            threshold = getattr(self, "_current_threshold", self.fatigue_limit)
            reward = -(
                self.reward_weights[0] * async_est / self.reward_norm['async']
                + self.reward_weights[1] * abs(dom_est) / self.reward_norm['dom_gap']
                + self.reward_weights[2] * (1.0 - fid_est)
                + self.time_weight * duration
                + self.fatigue_penalty * max(0.0, fatigue - threshold)
            )
            total += reward

            # 5) Decide whether to switch actions at the next step.
            if step < self.horizon - 1:
                idx = self._next_action(skill, fatigue, features)

        return float(total)

    def _next_action(self, skill, fatigue, features):
        threshold = getattr(self, '_current_threshold', self.fatigue_limit)
        if fatigue > threshold + 0.1:
            return self.rest_action
        residual = 1 - skill
        best_score = -np.inf
        best_idx = self.rest_action
        for idx in range(self.num_actions):
            if idx == self.rest_action:
                continue
            feat = features[idx]
            q_vec = feat[:self.skill_dim]
            difficulty = feat[self.skill_dim]
            score = np.dot(residual, q_vec) - 0.25 * difficulty
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def get_action_probabilities(self, obs):
        probs = np.zeros(self.num_actions)
        action = self.select_action(obs)
        probs[action] = 1.0
        return probs


# ===== Additional baselines for the comparison experiment =====
class BayesianMAB(Agent):
    """Bayesian Multi-Armed Bandit (Algorithms 2023)

    Uses a Gaussian posterior for real-valued rewards and an exponential
    discount factor for nonstationary environments. Thompson sampling is used
    to balance exploration and exploitation.

    Note: an earlier version used a Beta distribution plus a sigmoid mapping
    for real-valued rewards; this version uses a Gaussian posterior to better
    match the literature.
    """
    def __init__(self, num_actions, discount=0.995, prior_mu=0.0, prior_sigma=1.0,
                 sigma_noise=0.5, seed=None):
        self.discount = discount  # Exponential discount factor for nonstationarity.
        self.prior_mu = prior_mu  # Prior mean.
        self.prior_sigma = prior_sigma  # Prior standard deviation.
        self.sigma_noise = sigma_noise  # Observation noise standard deviation.
        super().__init__(num_actions, seed)

    def reset(self):
        # Gaussian posterior parameters (mu, sigma^2) for each arm.
        self.mu = np.full(self.num_actions, self.prior_mu, dtype=float)
        self.sigma_sq = np.full(self.num_actions, self.prior_sigma ** 2, dtype=float)
        # Running statistics.
        self.reward_count = np.zeros(self.num_actions)

    def select_action(self, obs):
        # Thompson sampling: draw once from each arm's Gaussian posterior.
        samples = []
        for a in range(self.num_actions):
            # Sample from N(mu_a, sigma_a^2).
            sigma = np.sqrt(max(self.sigma_sq[a], 1e-6))
            theta_sample = self.rng.normal(self.mu[a], sigma)
            samples.append(theta_sample)
        return int(np.argmax(samples))

    def update(self, obs, action, reward, next_obs):
        # Gaussian posterior update (Bayesian inference for a Normal with known variance).
        # Apply discounting to every arm by inflating variance to simulate forgetting.
        self.sigma_sq *= (1.0 / self.discount)

        # Bayesian update for the chosen arm.
        # Likelihood: reward ~ N(mu_a, sigma_noise^2)
        # Posterior: N(mu_post, sigma_post^2)
        sigma_obs_sq = self.sigma_noise ** 2
        precision_prior = 1.0 / max(self.sigma_sq[action], 1e-6)
        precision_obs = 1.0 / sigma_obs_sq

        # Posterior precision = prior precision + observation precision.
        precision_post = precision_prior + precision_obs
        sigma_sq_post = 1.0 / precision_post

        # Posterior mean = (prior precision * prior mean + observation precision * observed reward) / posterior precision.
        mu_post = (precision_prior * self.mu[action] + precision_obs * reward) / precision_post

        self.mu[action] = mu_post
        self.sigma_sq[action] = sigma_sq_post
        self.reward_count[action] += 1.0

    def get_action_probabilities(self, obs):
        # Return the greedy policy under the posterior mean, used for behavior-probability estimation.
        probs = np.zeros(self.num_actions)
        probs[int(np.argmax(self.mu))] = 1.0
        return probs

class CCB_DF(Agent):
    """Counterfactual Contextual Bandit with Delayed Feedback (NCA 2024)

    Uses a delayed-feedback queue and counterfactual importance weights to
    address delayed-observation problems. The base learner is LinUCB.

    Reference: Cai et al., "Counterfactual contextual bandit for recommendation
    under delayed feedback", Neural Computing and Applications, 2024.
    """
    def __init__(self, num_actions, context_dim, alpha=1.0, ridge=1e-3,
                 delay_window=1, weight_clip=10.0, seed=None):
        self.d = context_dim
        self.alpha = alpha
        self.ridge = ridge
        self.delay_window = delay_window  # Delay in steps.
        self.weight_clip = weight_clip  # Importance-weight clipping threshold.
        super().__init__(num_actions, seed)

    def reset(self):
        self._reset_metadata()
        # LinUCB parameters.
        self.A = [np.eye(self.d) * self.ridge for _ in range(self.num_actions)]
        self.b = [np.zeros(self.d) for _ in range(self.num_actions)]
        # Delayed-feedback queue storing (phi, action, timestep)-style entries.
        self.delay_queue = []

    def _phi(self, obs, action):
        return np.concatenate([obs['context'], obs['action_features'][action]])

    def select_action(self, obs):
        self._reset_metadata()
        # LinUCB action selection.
        scores = []
        for a in range(self.num_actions):
            phi = self._phi(obs, a)
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = phi @ theta + self.alpha * np.sqrt(max(phi @ A_inv @ phi, 0.0))
            scores.append(ucb)

        action = int(np.argmax(scores))

        # Push (phi, action) into the delayed queue.
        phi_t = self._phi(obs, action)
        behavior_prob = 1.0  # The current implementation is deterministic.
        self.delay_queue.append({
            'phi': phi_t,
            'action': action,
            'obs': obs,
            'selected_action': action,  # Store the action proposed by the agent.
            'behavior_prob': behavior_prob,
        })

        return action

    def update(self, obs, action, reward, next_obs):
        # Record the executed action for the current step so guard overrides can be detected later.
        if self.delay_queue:
            # Update the most recent queue entry with the actual executed action.
            current_step_data = self.delay_queue[-1]
            current_step_data['executed_action'] = action

        # Delayed-feedback update: pair the current reward with the data from t-delay_window.
        if len(self.delay_queue) > self.delay_window:
            # Pop the delayed record.
            delayed_data = self.delay_queue.pop(0)
            phi_prev = delayed_data['phi']
            a_selected = delayed_data['selected_action']
            a_executed = delayed_data.get('executed_action', a_selected)

            # Skip the update if the guard changed the action, since the counterfactual weight is then invalid.
            if a_selected != a_executed:
                return

            # Counterfactual importance weight: w_t = 1 / P_b(a|x).
            prob_behavior = float(delayed_data.get('behavior_prob', 0.0))
            if prob_behavior <= 0.0:
                prob_behavior = 1e-6
            weight = 1.0 / max(prob_behavior, 1e-6)
            weight = np.clip(weight, 0.0, self.weight_clip)  # Clip the weight.

            # Weighted update.
            self.A[a_selected] += weight * np.outer(phi_prev, phi_prev)
            self.b[a_selected] += weight * reward * phi_prev

    def get_action_probabilities(self, obs):
        scores = []
        for a in range(self.num_actions):
            phi = self._phi(obs, a)
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            score = phi @ theta
            scores.append(score)
        probs = np.zeros(self.num_actions)
        probs[int(np.argmax(scores))] = 1.0
        return probs


class SafeActorCritic(Agent):
    """Safe Reinforcement Learning with Actor-Critic (Robotics 2024)

    Uses a cost critic and a dual variable lambda to solve a constrained MDP.
    This is a lightweight Actor-Critic implementation for small-scale problems.
    """
    def __init__(self, num_actions, state_dim, hidden_dim=32,
                 lr_actor=1e-3, lr_critic=1e-3, lr_cost=1e-3, lr_lambda=1e-2,
                 gamma=0.99, cost_limit=0.75, lambda_init=0.1, seed=None):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_cost = lr_cost
        self.lr_lambda = lr_lambda
        self.gamma = gamma
        self.cost_limit = cost_limit
        self.lambda_init = lambda_init
        super().__init__(num_actions, seed)

    def reset(self):
        # Actor network: state -> action logits.
        self.W_actor = self.rng.normal(0, 0.1, size=(self.state_dim, self.hidden_dim))
        self.b_actor = np.zeros(self.hidden_dim)
        self.W_actor_out = self.rng.normal(0, 0.1, size=(self.hidden_dim, self.num_actions))
        self.b_actor_out = np.zeros(self.num_actions)

        # Critic network: state -> V(s).
        self.W_critic = self.rng.normal(0, 0.1, size=(self.state_dim, self.hidden_dim))
        self.b_critic = np.zeros(self.hidden_dim)
        self.W_critic_out = self.rng.normal(0, 0.1, size=(self.hidden_dim, 1))
        self.b_critic_out = np.zeros(1)

        # Cost-critic network: state -> C(s), predicting cumulative cost.
        self.W_cost = self.rng.normal(0, 0.1, size=(self.state_dim, self.hidden_dim))
        self.b_cost = np.zeros(self.hidden_dim)
        self.W_cost_out = self.rng.normal(0, 0.1, size=(self.hidden_dim, 1))
        self.b_cost_out = np.zeros(1)

        # Dual variable lambda (Lagrange multiplier).
        self.lambda_dual = self.lambda_init

        # Cached history.
        self.last_state = None

    def _state_vec(self, obs):
        state = np.asarray(obs['context'], dtype=float)
        # Dynamically adapt to the expected dimension when the context size changes.
        if len(state) != self.state_dim:
            new_state = np.zeros(self.state_dim)
            copy_len = min(len(state), self.state_dim)
            new_state[:copy_len] = state[:copy_len]
            return new_state
        return state

    def _forward_actor(self, state):
        h = np.tanh(state @ self.W_actor + self.b_actor)
        logits = h @ self.W_actor_out + self.b_actor_out
        return h, logits

    def _forward_critic(self, state):
        h = np.tanh(state @ self.W_critic + self.b_critic)
        v = (h @ self.W_critic_out + self.b_critic_out)[0]
        return h, v

    def _forward_cost(self, state):
        h = np.tanh(state @ self.W_cost + self.b_cost)
        c = (h @ self.W_cost_out + self.b_cost_out)[0]
        return h, c

    def select_action(self, obs):
        state = self._state_vec(obs)
        _, logits = self._forward_actor(state)

        # Sample from the softmax policy.
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        action = self.rng.choice(self.num_actions, p=probs)
        self.last_state = state
        return int(action)

    def update(self, obs, action, reward, next_obs):
        if self.last_state is None:
            return

        state = self.last_state
        next_state = self._state_vec(next_obs)

        # Compute the current cost as fatigue violation.
        info_pkg = next_obs.get('_info', {}) if isinstance(next_obs, dict) else {}
        fatigue = info_pkg.get('fatigue', next_obs.get('fatigue', next_obs.get('fatigue_est', 0.0)))
        cost = max(0.0, fatigue - self.cost_limit)

        # Critic update (TD learning).
        _, v_current = self._forward_critic(state)
        _, v_next = self._forward_critic(next_state)
        td_error = reward + self.gamma * v_next - v_current

        # Cost-critic update.
        _, c_current = self._forward_cost(state)
        _, c_next = self._forward_cost(next_state)
        cost_td_error = cost + self.gamma * c_next - c_current

        # Actor update (policy gradient minus dual-variable-weighted cost).
        h_actor, logits = self._forward_actor(state)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        # Advantage estimate: A = r - lambda*c - V.
        advantage = td_error - self.lambda_dual * cost_td_error

        # Policy gradient: grad log pi(a|s) * A.
        grad_logits = -probs.copy()
        grad_logits[action] += 1.0
        grad_logits *= advantage

        # Update the actor with a simplified gradient step.
        grad_W_actor_out = np.outer(h_actor, grad_logits) * self.lr_actor
        self.W_actor_out += grad_W_actor_out
        self.b_actor_out += grad_logits * self.lr_actor

        # Update the critic.
        h_critic, _ = self._forward_critic(state)
        grad_critic_out = h_critic * td_error * self.lr_critic
        self.W_critic_out += grad_critic_out.reshape(-1, 1)
        self.b_critic_out[0] += td_error * self.lr_critic

        # Update the cost critic.
        h_cost, _ = self._forward_cost(state)
        grad_cost_out = h_cost * cost_td_error * self.lr_cost
        self.W_cost_out += grad_cost_out.reshape(-1, 1)
        self.b_cost_out[0] += cost_td_error * self.lr_cost

        # Update the dual variable lambda via dual ascent: lambda += lr * (cost - threshold).
        cost_violation = cost - 0.0  # The target is to keep cost close to zero.
        self.lambda_dual = np.clip(
            self.lambda_dual + self.lr_lambda * cost_violation,
            0.0, 10.0
        )

    def get_action_probabilities(self, obs):
        state = self._state_vec(obs)
        _, logits = self._forward_actor(state)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        return probs

class AutoCurriculum(Agent):
    """Automatic Curriculum Learning (Applied Sciences 2023)

    A difficulty-based state machine that adjusts the difficulty stage according
    to learner performance. The policy advances when performance is strong and
    regresses when performance drops.
    """
    def __init__(self, num_actions, difficulty_stages=None,
                 advance_threshold=0.7, regress_threshold=0.3,
                 stability_count=3, seed=None):
        self.difficulty_stages = difficulty_stages or self._default_stages()
        self.advance_threshold = advance_threshold  # Promotion threshold.
        self.regress_threshold = regress_threshold  # Regression threshold.
        self.stability_count = stability_count  # Required number of stable observations.
        super().__init__(num_actions, seed)

    def _default_stages(self):
        """Default difficulty-stage definitions."""
        return [
            {'name': 'easy', 'difficulty_range': (0.0, 0.4), 'min_performance': 0.6},
            {'name': 'medium', 'difficulty_range': (0.35, 0.7), 'min_performance': 0.5},
            {'name': 'hard', 'difficulty_range': (0.65, 1.0), 'min_performance': 0.4},
        ]

    def reset(self):
        self.current_stage = 0  # Start from the easiest stage.
        self.performance_buffer = []  # Recent performance history.
        self.stable_count = 0  # Stability counter.

    def select_action(self, obs):
        # Get the current difficulty stage.
        stage = self.difficulty_stages[self.current_stage]
        min_diff, max_diff = stage['difficulty_range']

        # Filter actions whose difficulty matches the current stage.
        action_features = obs.get('action_features', [])
        candidates = []

        for a in range(self.num_actions):
            if a >= len(action_features):
                continue
            feat = action_features[a]
            # Assume ``difficulty`` is stored at position K (with K=5).
            if len(feat) > 5:
                difficulty = feat[5]
                if min_diff <= difficulty <= max_diff:
                    candidates.append(a)

        # If nothing matches exactly, relax the range.
        if not candidates:
            candidates = list(range(min(self.num_actions, len(action_features))))

        # Randomly choose among candidate actions.
        return int(self.rng.choice(candidates))

    def update(self, obs, action, reward, next_obs):
        # Score performance by mapping reward into [0, 1].
        # Assume rewards lie in [-5, 0].
        performance = np.clip((reward + 5.0) / 5.0, 0.0, 1.0)
        self.performance_buffer.append(performance)

        # Maintain a fixed-length sliding window.
        if len(self.performance_buffer) > 10:
            self.performance_buffer.pop(0)

        # Compute average performance.
        if len(self.performance_buffer) >= 5:
            avg_performance = np.mean(self.performance_buffer)
            stage = self.difficulty_stages[self.current_stage]

            # Promotion logic.
            if avg_performance >= self.advance_threshold:
                self.stable_count += 1
                if self.stable_count >= self.stability_count:
                    if self.current_stage < len(self.difficulty_stages) - 1:
                        self.current_stage += 1
                        self.performance_buffer.clear()
                        self.stable_count = 0
            # Regression logic.
            elif avg_performance < self.regress_threshold:
                self.stable_count += 1
                if self.stable_count >= self.stability_count:
                    if self.current_stage > 0:
                        self.current_stage -= 1
                        self.performance_buffer.clear()
                        self.stable_count = 0
            else:
                self.stable_count = 0

    def get_action_probabilities(self, obs):
        # Return a point mass over the selected candidate action.
        probs = np.zeros(self.num_actions)
        action = self.select_action(obs)
        probs[action] = 1.0
        return probs


# ===== Helper utilities =====
def get_agent(name, num_actions, context_dim=None, seed=None, **kwargs):
    """Factory function that creates the eight retained agents.
    
    Available agents:
    - linucb: LinUCB (requires context_dim)
    - thompson: ThompsonSampling (requires context_dim)  
    - dqn: DQNAgent (requires state_dim)
    - pianoMPC: ModelPredictivePlanner
    - bayesianmab: BayesianMAB
    - ccb_df: CCB_DF (requires context_dim)
    - safe_ac: SafeActorCritic (requires state_dim)
    - autocurriculum: AutoCurriculum
    """
    # Extract guard parameters used only by LinUCB and PianoMPC.
    guard_horizon = kwargs.pop('guard_horizon', None)
    guard_delta = kwargs.pop('guard_delta', None)
    guard_safety_margin = kwargs.pop('guard_safety_margin', None)
    
    # Remove deprecated FatigueGuardWrapper-related arguments.
    kwargs.pop('enable_fatigue_guard', None)
    kwargs.pop('fatigue_threshold', None)

    agents = {
        'linucb': LinUCB,
        'thompson': ThompsonSampling,
        'dqn': DQNAgent,
        'pianoMPC': ModelPredictivePlanner,
        'bayesianmab': BayesianMAB,
        'ccb_df': CCB_DF,
        'safe_ac': SafeActorCritic,
        'autocurriculum': AutoCurriculum,
    }

    if name not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys())}")

    # Construct the agent according to its parameter requirements.
    if name in ['linucb', 'thompson', 'ccb_df']:
        if context_dim is None:
            raise ValueError(f"{name} requires context_dim")
        agent = agents[name](num_actions, context_dim, seed=seed, **kwargs)
    elif name in ['dqn', 'safe_ac']:
        state_dim = kwargs.pop('state_dim', None)
        if state_dim is None:
            raise ValueError(f"{name} requires state_dim")
        agent = agents[name](num_actions, state_dim, seed=seed, **kwargs)
    else:
        # bayesianmab, pianoMPC, autocurriculum
        agent = agents[name](num_actions, seed=seed, **kwargs)

    # Apply guard parameters for agents with built-in guard support (LinUCB and PianoMPC).
    if hasattr(agent, "set_guard_params"):
        if any(param is not None for param in (guard_horizon, guard_delta, guard_safety_margin)):
            agent.set_guard_params(
                horizon=guard_horizon,
                delta=guard_delta,
                safety_margin=guard_safety_margin,
            )

    return agent
