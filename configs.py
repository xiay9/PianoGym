"""
PianoGym Configuration
Centralized task parameters, learner profiles, transfer strengths, and constraint settings.
"""
from __future__ import annotations

import copy
from typing import Dict, Iterable, Optional

import numpy as np


class Config:
    def __init__(self):
        # ===== Skill dimensions and names =====
        self.K = 5  # [x_L, x_R, x_2:3, x_3:4, x_switch]
        self.skill_names = ['Left', 'Right', 'Poly_2:3', 'Poly_3:4', 'Switch']

        # ===== Action space (exercise types) =====
        self.difficulties = [0.3, 0.6, 0.85]
        self.beta_schedule = [0.06, 0.09, 0.13]
        self.duration_schedule = [1.2, 1.6, 2.4]

        # ===== Learning dynamics parameters =====
        self.alpha_range = [0.04, 0.12]
        self.gamma_f = 0.4
        self.kappa = 5.0
        self.gate_offset = 0.1
        self.min_fatigue_factor = 0.25
        self.sigma_x = 0.01

        # ===== Fatigue and forgetting =====
        self.beta_range = [0.05, 0.08, 0.12]
        self.rho = 0.25
        self.lambda_r = 0.9
        self.eta_forget = 0.02

        # ===== Observation noise and baselines =====
        self.sigma_async = 10.0
        self.sigma_dom = 8.0
        self.phi_range = [40, 80]
        self.psi_range = [20, 40]
        self.xi = 5.0

        self.eta = {'eta0': -1.0, 'eta1': 2.0, 'eta2': 0.8, 'eta3': 0.6}

        # ===== Normalization constants =====
        self.async_ref = 120.0
        self.dom_ref = 60.0
        self.fatigue_noise = 0.05
        self.retention_noise = 0.05
        self.reward_norm = {
            'async': 80.0,
            'dom_gap': 30.0,
            'fid': 1.0,
        }
        self.context_mean = np.array([0.6, 0.0, 0.6, 0.35, 0.55, 0.0])
        self.context_std = np.array([0.3, 0.4, 0.2, 0.2, 0.2, 0.1])
        self.async_weights = np.array([0.4, 0.4, 0.1, 0.1, 0.0])

        # ===== Reward and constraints =====
        self.w_reward = [1.0, 0.8, 1.5]
        self.lambda_c = 0.01
        self.transfer_strength_levels = {'weak': 0.6, 'medium': 1.0, 'strong': 1.4}
        self.transfer_scale = 1.0

        self.enable_fatigue_constraint = True
        self.fatigue_threshold = 0.75
        self.constraint_penalty = 0.30
        # Align rest recovery with the paper symbol rho while preserving backward compatibility.
        self.rest_recovery = self.rho

        self.enable_nonstationary = True
        self.nonstationary = {
            'start_step': 120,
            'cycle': 80,
            'decay': 0.02,
            'boost': 0.015,
            'noise': 0.01,
        }

        # ===== Learner profiles =====
        self.profile_defs = {
            'balanced': {
                'name': 'balanced',
                'init_skill_range': (0.12, 0.28),
                'left_weakness': 0.0,
                'fatigue_bias': 0.0,
                'fatigue_threshold': 0.75,
                'retention_bias': 0.0,
                'enforce_constraint': True,
            },
            'mild_left_weak': {
                'name': 'mild_left_weak',
                'init_skill_range': (0.12, 0.26),
                'left_weakness': 0.12,
                'fatigue_bias': 0.05,
                'fatigue_threshold': 0.7,
                'retention_bias': -0.03,
                'enforce_constraint': True,
            },
            'severe_left_weak': {
                'name': 'severe_left_weak',
                'init_skill_range': (0.1, 0.22),
                'left_weakness': 0.22,
                'fatigue_bias': 0.08,
                'fatigue_threshold': 0.65,
                'retention_bias': -0.06,
                'enforce_constraint': True,
            },
        }

        # ===== Mastery criteria and environment settings =====
        self.mastery_thresh = {'async': 60.0, 'fid': 0.80, 'dom_gap': 20.0}
        self.mastery_window = 3
        self.max_steps = 320
        self.init_skill_range = [0.1, 0.3]

        # ===== Transfer matrix and exercise bank generation =====
        self.M = self._init_transfer_matrix(self.transfer_scale)
        self.exercise_params = self._init_exercises()
        self.num_actions = len(self.exercise_params)
        self.action_feature_dim = len(self.exercise_params[0]['feature_vector'])
        self.task_tag = 'basic'

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------
    def _base_type_definitions(self) -> Iterable[Dict]:
        return [
            {
                'name': 'Left_Independence',
                'q': [1.0, 0.2, 0.1, 0.1, 0.3],
                'alpha': 0.07,
                'tau': [0.0, 0.03, 0.0, 0.0, 0.04],
                'transfer': [1.0, 0.2, 0.0, 0.0, 0.3],
                'coord_focus': 0.2,
                'fatigue_scale': 1.0,
            },
            {
                'name': 'Right_Independence',
                'q': [0.2, 1.0, 0.1, 0.1, 0.3],
                'alpha': 0.07,
                'tau': [0.03, 0.0, 0.0, 0.0, 0.04],
                'transfer': [0.2, 1.0, 0.0, 0.0, 0.3],
                'coord_focus': 0.2,
                'fatigue_scale': 1.0,
            },
            {
                'name': 'Poly_2:3',
                'q': [0.3, 0.3, 1.0, 0.1, 0.4],
                'alpha': 0.05,
                'tau': [0.0, 0.0, 0.0, 0.04, 0.02],
                'transfer': [0.2, 0.2, 1.0, 0.1, 0.3],
                'coord_focus': 0.6,
                'fatigue_scale': 1.1,
            },
            {
                'name': 'Poly_3:4',
                'q': [0.3, 0.3, 0.1, 1.0, 0.4],
                'alpha': 0.05,
                'tau': [0.0, 0.0, 0.04, 0.0, 0.02],
                'transfer': [0.2, 0.2, 0.1, 1.0, 0.3],
                'coord_focus': 0.6,
                'fatigue_scale': 1.1,
            },
            {
                'name': 'Switching',
                'q': [0.4, 0.4, 0.2, 0.2, 1.0],
                'alpha': 0.06,
                'tau': [0.02, 0.02, 0.03, 0.03, 0.0],
                'transfer': [0.3, 0.3, 0.2, 0.2, 1.0],
                'coord_focus': 0.8,
                'fatigue_scale': 1.2,
            },
            {
                'name': 'Phase_Inversion',
                'q': [0.5, 0.5, 0.4, 0.4, 0.6],
                'alpha': 0.04,
                'tau': [-0.02, -0.02, 0.03, 0.03, 0.01],
                'transfer': [0.4, 0.4, 0.3, 0.3, 0.5],
                'coord_focus': 0.7,
                'fatigue_scale': 1.0,
            },
            {
                'name': 'Accent_Shaping',
                'q': [0.3, 0.3, 0.5, 0.5, 0.6],
                'alpha': 0.05,
                'tau': [0.01, 0.01, 0.02, 0.02, 0.02],
                'transfer': [0.2, 0.2, 0.4, 0.4, 0.5],
                'coord_focus': 0.5,
                'fatigue_scale': 0.9,
            },
            {
                'name': 'Tempo_Shift',
                'q': [0.6, 0.6, 0.2, 0.2, 0.3],
                'alpha': 0.06,
                'tau': [0.02, 0.02, 0.01, 0.01, 0.02],
                'transfer': [0.5, 0.5, 0.2, 0.2, 0.3],
                'coord_focus': 0.4,
                'fatigue_scale': 1.3,
            },
        ]

    def _init_exercises(self):
        params = []
        types = list(self._base_type_definitions())
        self.num_exercise_types = len(types)

        for type_idx, typ in enumerate(types):
            q_vec = np.array(typ['q'])
            base_tau = np.array(typ['tau'])
            base_transfer = np.array(typ.get('transfer', typ['q']))
            coord_focus = typ.get('coord_focus', 0.5)
            fatigue_scale = typ.get('fatigue_scale', 1.0)

            tau_scaled = self.transfer_scale * base_tau
            transfer_vector = self.transfer_scale * base_transfer

            for diff_idx, d in enumerate(self.difficulties):
                beta = self.beta_schedule[diff_idx] * fatigue_scale
                duration = self.duration_schedule[diff_idx]
                alpha = typ['alpha'] * (1.0 + 0.1 * diff_idx)
                alpha = np.clip(alpha, self.alpha_range[0], self.alpha_range[1])

                feature_vector = np.concatenate([
                    q_vec,
                    [
                        d,
                        coord_focus,
                        beta,
                        duration,
                        type_idx / max(1, self.num_exercise_types - 1),
                        self.transfer_scale,
                    ],
                ])

                params.append({
                    'name': f"{typ['name']}_d{d:.2f}",
                    'difficulty': d,
                    'type_index': type_idx,
                    'q_a': q_vec,
                    'alpha_a': alpha * np.ones(self.K),
                    'tau_a': tau_scaled,
                    'transfer_vector': transfer_vector,
                    'beta_a': beta,
                    'c_a': duration,
                    'phi_a': self.phi_range[0] + d * (self.phi_range[1] - self.phi_range[0]),
                    'psi_a': self.psi_range[0] + d * (self.psi_range[1] - self.psi_range[0]),
                    'coord_focus': coord_focus,
                    'feature_vector': feature_vector,
                })

        # REST action
        rest_feature = np.concatenate([
            np.zeros(self.K),
            [0.0, 0.0, 0.0, 1.0, 1.0, self.transfer_scale],
        ])
        params.append({
            'name': 'REST',
            'difficulty': 0.0,
            'type_index': self.num_exercise_types,
            'q_a': np.zeros(self.K),
            'alpha_a': np.zeros(self.K),
            'tau_a': np.zeros(self.K),
            'transfer_vector': np.zeros(self.K),
            'beta_a': 0.0,
            'c_a': 1.0,
            'phi_a': 0.0,
            'psi_a': 0.0,
            'coord_focus': 0.0,
            'feature_vector': rest_feature,
            'is_rest': True,
        })

        return params

    def _init_transfer_matrix(self, scale: float):
        M = np.zeros((self.K, self.K))
        M[0, 1] = M[1, 0] = 0.01 * scale
        M[2, 3] = M[3, 2] = 0.03 * scale
        M[4, :4] = 0.02 * scale
        M[:4, 4] = 0.015 * scale
        return M

    # ------------------------------------------------------------------
    # Configuration utilities
    # ------------------------------------------------------------------
    def set_transfer_strength(self, level: str | float):
        if isinstance(level, str):
            if level not in self.transfer_strength_levels:
                raise ValueError(f"Unknown transfer strength: {level}")
            self.transfer_scale = self.transfer_strength_levels[level]
        else:
            self.transfer_scale = float(level)

        self.M = self._init_transfer_matrix(self.transfer_scale)
        self.exercise_params = self._init_exercises()
        self.num_actions = len(self.exercise_params)
        self.action_feature_dim = len(self.exercise_params[0]['feature_vector'])
        return self

    def resolve_profile(self, profile: Optional[str | Dict] = None) -> Dict:
        if profile is None:
            return copy.deepcopy(self.profile_defs['balanced'])
        if isinstance(profile, str):
            if profile not in self.profile_defs:
                raise ValueError(f"Unknown profile: {profile}")
            return copy.deepcopy(self.profile_defs[profile])
        base = copy.deepcopy(self.profile_defs.get(profile.get('name', 'balanced'), {}))
        base.update(profile)
        return base

    def clone(self, **overrides):
        variant = copy.deepcopy(self)
        transfer_strength = overrides.pop('transfer_strength', None)
        for key, value in overrides.items():
            setattr(variant, key, value)
        if transfer_strength is not None:
            variant.set_transfer_strength(transfer_strength)
        else:
            variant.exercise_params = variant._init_exercises()
            variant.num_actions = len(variant.exercise_params)
            variant.action_feature_dim = len(variant.exercise_params[0]['feature_vector'])
        return variant


# Singleton instance for convenient direct use.
cfg = Config()
