"""
PianoGym Environment
Core environment: state transitions, observation generation, and reward computation (see PianoGym.md §3-4)
"""
import numpy as np
from configs import cfg


class PianoGymEnv:
    def __init__(self, config=None, seed=None):
        self.cfg = config or cfg
        self.rng = np.random.default_rng(seed)
        self.action_features = None
        self.obs_context_dim = 0
        self.last_metrics = None
        self.profile_info = None
        self.constraint_enabled = False
        self.fatigue_threshold = self.cfg.fatigue_threshold
        self.constraint_violation = 0.0
        self.current_constraint_penalty = 0.0
        self.enable_nonstationary = self.cfg.enable_nonstationary
        self.nonstationary_events = 0
        self._refresh_action_features()
        self.context_count = 0
        self.context_mean = None
        self.context_M2 = None
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_M2 = 1e-6
        self.last_raw_reward = 0.0
        self.reset()

    @property
    def action_feature_dim(self):
        return self.cfg.action_feature_dim

    def reset(self, profile=None, transfer_strength=None,
              enforce_constraint=None, enable_nonstationary=None):
        """Reset the environment.
        profile: str/dict learner profile
        transfer_strength: {'weak','medium','strong'} or float
        """
        if transfer_strength is not None:
            self.cfg.set_transfer_strength(transfer_strength)
            self._refresh_action_features()

        self.profile_info = self.cfg.resolve_profile(profile)

        # Initialize state.
        init_range = self.profile_info.get('init_skill_range', self.cfg.init_skill_range)
        self.x = self.rng.uniform(init_range[0], init_range[1], self.cfg.K)
        # Apply left-hand weakness.
        self.x[0] -= self.profile_info.get('left_weakness', 0.0)
        self.x = np.clip(self.x, 0, 1)

        fatigue_bias = self.profile_info.get('fatigue_bias', 0.0)
        retention_bias = self.profile_info.get('retention_bias', 0.0)
        self.f = np.clip(fatigue_bias, 0, 1)
        self.r = np.clip(np.mean(self.x) + retention_bias, 0, 1)

        self.step_count = 0
        self.mastery_count = 0  # Consecutive mastery counter.
        self.done = False
        self.last_metrics = None
        self.nonstationary_events = 0

        self.fatigue_threshold = self.profile_info.get('fatigue_threshold', self.cfg.fatigue_threshold)
        constraint_flag = self.profile_info.get('enforce_constraint', False) or self.cfg.enable_fatigue_constraint
        if enforce_constraint is not None:
            constraint_flag = bool(enforce_constraint)
        self.constraint_enabled = constraint_flag
        self.constraint_violation = max(0.0, self.f - self.fatigue_threshold) if self.constraint_enabled else 0.0
        self.current_constraint_penalty = self.cfg.constraint_penalty * self.constraint_violation

        nonstationary_flag = self.profile_info.get('enable_nonstationary', self.cfg.enable_nonstationary)
        if enable_nonstationary is not None:
            nonstationary_flag = bool(enable_nonstationary)
        self.enable_nonstationary = nonstationary_flag
        self.last_raw_reward = 0.0

        return self._observe()

    def step(self, action_idx):
        """Execute one environment step.
        action_idx: int in [0, num_actions-1]
        Returns: (obs, reward, done, info)
        """
        assert 0 <= action_idx < self.cfg.num_actions, f"Invalid action: {action_idx}"

        params = self.cfg.exercise_params[action_idx]

        # 1. Skill update (§3.1).
        self._update_skills(params)

        # 2. Fatigue and retention update (§3.2).
        self._update_fatigue_retention(params)

        # 3. Forgetting.
        self._apply_forgetting()

        # 4. Nonstationary perturbations.
        self._apply_nonstationary_effects()

        # 5. Constraint check.
        if self.constraint_enabled:
            self.constraint_violation = max(0.0, self.f - self.fatigue_threshold)
            self.current_constraint_penalty = self.cfg.constraint_penalty * self.constraint_violation
        else:
            self.constraint_violation = 0.0
            self.current_constraint_penalty = 0.0

        # 6. Observation and reward.
        obs = self._observe(params)
        reward = self._compute_reward(obs, params)

        # 7. Mastery check.
        self.step_count += 1
        is_mastery = self._check_mastery(obs)
        if is_mastery:
            self.mastery_count += 1
        else:
            self.mastery_count = 0

        if self.mastery_count >= self.cfg.mastery_window:
            self.done = True
        elif self.step_count >= self.cfg.max_steps:
            self.done = True

        info = {
            'true_skills': self.x.copy(),
            'fatigue': self.f,
            'retention': self.r,
            'mastery_count': self.mastery_count,
            'action_name': params['name'],
            'constraint_violation': self.constraint_violation,
            'fatigue_threshold': self.fatigue_threshold,
            'nonstationary_events': self.nonstationary_events,
            'raw_reward': self.last_raw_reward,
        }

        # Guard checks can be added in debug mode if needed.
        # In ablations such as No_REST, fatigue violations may be expected and should not trigger warnings.

        return obs, reward, self.done, info

    # ===== Internal methods =====

    def _update_skills(self, params):
        """Skill update (Eq. §3.1)."""
        d_a = params['difficulty']
        q_a = params['q_a']
        alpha_a = params['alpha_a']
        tau_a = params['tau_a']
        transfer_vec = params.get('transfer_vector', np.zeros_like(self.x))

        # Difficulty-matching gate g(q, x, d).
        match_score = np.dot(q_a, self.x) - d_a + self.cfg.gate_offset
        g = 1.0 / (1.0 + np.exp(-self.cfg.kappa * match_score))  # sigmoid

        # Main learning gain.
        fatigue_factor = max(self.cfg.min_fatigue_factor, 1 - self.cfg.gamma_f * self.f)
        main_gain = alpha_a * (1 - self.x) * q_a * g * fatigue_factor

        # Transfer effects: intrinsic transfer plus exercise interference.
        transfer_matrix_effect = self.cfg.M @ transfer_vec
        transfer = tau_a + transfer_matrix_effect

        # Noise.
        noise = self.rng.normal(0, self.cfg.sigma_x, self.cfg.K)

        # Update.
        delta_x = main_gain + transfer + noise
        self.x = np.clip(self.x + delta_x, 0, 1)

    def _update_fatigue_retention(self, params):
        """Fatigue and retention update (Eq. §3.2)."""
        beta_a = params['beta_a']
        if params.get('is_rest', False):
            rest_recovery = getattr(self.cfg, 'rho', None)
            if rest_recovery is None:
                rest_recovery = getattr(self.cfg, 'rest_recovery', 0.2)
            self.f = np.clip(self.f - rest_recovery, 0, 1)
        else:
            self.f = np.clip(self.f + beta_a, 0, 1)

        # Retention (EMA).
        self.r = self.cfg.lambda_r * self.r + (1 - self.cfg.lambda_r) * np.mean(self.x)

    def _apply_forgetting(self):
        """Apply forgetting decay."""
        forget_factor = self.cfg.eta_forget * (1 - self.r)
        self.x = self.x * (1 - forget_factor)
        self.x = np.clip(self.x, 0, 1)

    def _apply_nonstationary_effects(self):
        if not self.enable_nonstationary:
            return
        schedule = self.cfg.nonstationary
        start = schedule.get('start_step', 0)
        cycle = schedule.get('cycle', 0)
        if self.step_count < start or cycle <= 0:
            return
        if (self.step_count - start) % cycle != 0:
            return

        decay = schedule.get('decay', 0.0)
        boost = schedule.get('boost', 0.0)
        noise = schedule.get('noise', 0.0)

        drift = self.rng.normal(0, noise, size=self.cfg.K)
        self.x = np.clip(self.x * (1 - decay) + drift, 0, 1)
        self.f = np.clip(self.f * (1 - boost), 0, 1)
        self.r = np.clip(self.r * (1 - decay / 2) + boost, 0, 1)
        self.nonstationary_events += 1

    def _observe(self, params=None):
        """Generate an observation (Eq. §3.3).
        Returns: {'async', 'dom_gap', 'fid', 'context'}
        """
        if params is None:
            # Initial observation at reset.
            async_init = 100.0
            dom_gap_init = self.cfg.xi
            fid_init = 0.3
            fatigue_proxy = np.clip(self.f + self.rng.normal(0, self.cfg.fatigue_noise), 0, 1)
            retention_proxy = np.clip(self.r + self.rng.normal(0, self.cfg.retention_noise), 0, 1)
            progress_proxy = 0.0
            context_raw = np.array([
                async_init / self.cfg.reward_norm['async'],
                dom_gap_init / self.cfg.reward_norm['dom_gap'],
                fid_init,
                fatigue_proxy,
                retention_proxy,
                progress_proxy
            ])
            context = self._standardize_context(context_raw)
            self.obs_context_dim = len(context)
            metrics = {'async': async_init, 'dom_gap': dom_gap_init, 'fid': fid_init}
            self.last_metrics = metrics
            return {
                'async': async_init,
                'dom_gap': dom_gap_init,
                'fid': fid_init,
                'fatigue_est': fatigue_proxy,
                'retention_est': retention_proxy,
                'fatigue_threshold': self.fatigue_threshold,
                'context': context,
                'action_features': self.action_features.copy()
            }

        # Asynchrony (ms)
        if len(self.cfg.async_weights) != self.cfg.K:
            raise ValueError("cfg.async_weights size must match number of skills")
        w = self.cfg.async_weights
        mu_async = params['phi_a'] * (1 - np.dot(w, self.x)) * (1 + 0.3 * self.f)
        async_obs = max(0, self.rng.normal(mu_async, self.cfg.sigma_async))

        # Dominance Gap (ms)
        mu_dom = params['psi_a'] * (self.x[1] - self.x[0]) + self.cfg.xi
        dom_gap = self.rng.normal(mu_dom, self.cfg.sigma_dom)

        # Fidelity [0,1]
        eta = self.cfg.eta
        logit = eta['eta0'] + eta['eta1'] * np.dot(params['q_a'], self.x) \
                - eta['eta2'] * params['difficulty'] - eta['eta3'] * self.f
        fid = 1.0 / (1.0 + np.exp(-logit))
        fid = np.clip(fid + self.rng.normal(0, 0.05), 0, 1)

        fatigue_proxy = np.clip(self.f + self.rng.normal(0, self.cfg.fatigue_noise), 0, 1)
        retention_proxy = np.clip(self.r + self.rng.normal(0, self.cfg.retention_noise), 0, 1)
        prev_fid = self.last_metrics['fid'] if self.last_metrics else fid
        progress_proxy = fid - prev_fid

        context_raw = np.array([
            async_obs / self.cfg.reward_norm['async'],
            dom_gap / self.cfg.reward_norm['dom_gap'],
            fid,
            fatigue_proxy,
            retention_proxy,
            progress_proxy
        ])

        context = self._standardize_context(context_raw)

        metrics = {'async': async_obs, 'dom_gap': dom_gap, 'fid': fid}
        self.last_metrics = metrics

        return {
            'async': async_obs,
            'dom_gap': dom_gap,
            'fid': fid,
            'fatigue_est': fatigue_proxy,
            'retention_est': retention_proxy,
            'fatigue_threshold': self.fatigue_threshold,
            'context': context,
            'action_features': self.action_features.copy()
        }

    def _compute_reward(self, obs, params):
        """Reward computation (Eq. §4)."""
        w = self.cfg.w_reward
        async_norm = np.clip(obs['async'] / self.cfg.reward_norm['async'], 0, 1.5)
        dom_norm = np.clip(abs(obs['dom_gap']) / self.cfg.reward_norm['dom_gap'], 0, 1.5)
        fid_norm = np.clip(1 - obs['fid'], 0, 1.0)
        penalty = w[0] * async_norm + w[1] * dom_norm + w[2] * fid_norm

        time_cost = self.cfg.lambda_c * params['c_a']
        raw_reward = -(penalty + time_cost + self.current_constraint_penalty)
        standardized_reward = self._standardize_reward(raw_reward)
        self.last_raw_reward = raw_reward
        return standardized_reward

    def _check_mastery(self, obs):
        """Check whether the current observation meets the mastery criteria."""
        thresh = self.cfg.mastery_thresh
        return (obs['async'] <= thresh['async'] and
                obs['fid'] >= thresh['fid'] and
                abs(obs['dom_gap']) <= thresh['dom_gap'])

    def _standardize_context(self, vector):
        vec = np.asarray(vector, dtype=float)

        base_mean = getattr(self.cfg, 'context_mean', None)
        base_std = getattr(self.cfg, 'context_std', None)
        if (
            base_mean is not None
            and base_std is not None
            and len(base_mean) == len(vec)
            and len(base_std) == len(vec)
        ):
            mean = np.asarray(base_mean, dtype=float)
            std = np.asarray(base_std, dtype=float)
            std = np.where(std <= 1e-6, 1.0, std)
            return np.clip((vec - mean) / std, -3.5, 3.5)

        if self.context_mean is None:
            self.context_mean = vec.astype(float).copy()
            self.context_M2 = np.zeros_like(vec, dtype=float)
            self.context_count = 0

        self.context_count += 1
        delta = vec - self.context_mean
        self.context_mean += delta / self.context_count
        delta2 = vec - self.context_mean
        self.context_M2 += delta * delta2

        if self.context_count < 2:
            std = np.ones_like(vec, dtype=float)
        else:
            variance = self.context_M2 / (self.context_count - 1)
            std = np.sqrt(np.maximum(variance, 1e-6))

        return np.clip((vec - self.context_mean) / std, -3.5, 3.5)

    def _standardize_reward(self, reward):
        if self.reward_count < 1:
            standardized = 0.0
        else:
            variance = self.reward_M2 / self.reward_count
            std = max(np.sqrt(max(variance, 1e-6)), 1e-3)
            standardized = np.clip((reward - self.reward_mean) / std, -5.0, 5.0)

        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_M2 += delta * delta2

        return standardized

    def _refresh_action_features(self):
        self.action_features = np.array([p['feature_vector'] for p in self.cfg.exercise_params])
