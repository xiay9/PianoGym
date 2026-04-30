"""
Shared safety guard utilities to enforce fatigue limits at the episode runner.

The guard operates outside of individual agents so that every baseline
experiences the same filtering logic. It inspects the current observation to
estimate next-step fatigue for each action and, when necessary, projects an
unsafe proposal back to REST.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class GuardWindow:
    """Pre-computed guard view of the current step."""

    fatigue_est: float
    threshold: float
    guard_threshold: float
    step_cap: float
    safe_actions: Tuple[int, ...]
    relaxed_actions: Tuple[int, ...]
    predicted_next_fatigue: Dict[int, float]


@dataclass
class GuardDecision:
    """Decision record returned after enforcing the guard."""

    original_action: int
    final_action: int
    replaced: bool
    reason: str
    fatigue_est: float
    threshold: float
    guard_threshold: float
    step_cap: float
    predicted_next_fatigue: Dict[int, float]

    def as_dict(self) -> Dict[str, object]:
        payload = {
            "original_action": int(self.original_action),
            "final_action": int(self.final_action),
            "replaced": bool(self.replaced),
            "reason": self.reason,
            "fatigue_est": float(self.fatigue_est),
            "threshold": float(self.threshold),
            "guard_threshold": float(self.guard_threshold),
            "step_cap": float(self.step_cap),
            "predicted_next_fatigue": {
                int(idx): float(val) for idx, val in self.predicted_next_fatigue.items()
            },
        }
        return payload


class ExternalSafetyGuard:
    """Episode-level fatigue guard that projects unsafe actions to REST."""

    def __init__(
        self,
        env,
        *,
        enabled: bool = True,
        guard_delta: float = 0.08,
        safety_margin: float = 0.05,
        rest_action: Optional[int] = None,
        rest_recovery: Optional[float] = None,
    ) -> None:
        self.env = env
        self.enabled = bool(enabled)
        self.guard_delta = float(guard_delta)
        self.safety_margin = max(0.0, float(safety_margin))
        self.rest_action = (
            int(rest_action) if rest_action is not None else env.cfg.num_actions - 1
        )
        cfg_rest = getattr(env.cfg, "rest_recovery", None)
        self.rest_recovery = float(
            rest_recovery if rest_recovery is not None else cfg_rest if cfg_rest is not None else 0.25
        )
        self.skill_dim = getattr(env.cfg, "K", len(env.cfg.exercise_params[0]["q_a"]))
        self._last_window: Optional[GuardWindow] = None
        self._last_decision: Optional[GuardDecision] = None

    def reset(self) -> None:
        self._last_window = None
        self._last_decision = None

    # ------------------------------------------------------------------ #
    # Window computation
    # ------------------------------------------------------------------ #
    def _window(self, obs: Dict) -> GuardWindow:
        fatigue_est = float(obs.get("fatigue_est", obs.get("fatigue", 0.0)))
        threshold = float(obs.get("fatigue_threshold", getattr(self.env, "fatigue_threshold", 0.75)))
        guard_threshold = min(1.0, threshold + self.guard_delta)
        step_cap = max(0.0, threshold - self.safety_margin)

        features = np.asarray(obs.get("action_features", ()))
        if features.size == 0:
            total_actions = getattr(self.env.cfg, "num_actions", len(self.env.cfg.exercise_params))
            safe_all = tuple(range(total_actions))
            window = GuardWindow(
                fatigue_est=fatigue_est,
                threshold=threshold,
                guard_threshold=guard_threshold,
                step_cap=step_cap,
                safe_actions=safe_all,
                relaxed_actions=safe_all,
                predicted_next_fatigue={},
            )
            self._last_window = window
            return window

        predicted: Dict[int, float] = {}
        safe: List[int] = []
        relaxed: List[int] = []

        params_list: Iterable[Dict] = getattr(self.env.cfg, "exercise_params", [])

        for idx in range(len(features)):
            if idx >= len(params_list):
                params = {}
            else:
                params = params_list[idx]
            feat = features[idx]
            is_rest = bool(params.get("is_rest", False))
            beta = float(params.get("beta_a", feat[self.skill_dim + 2] if len(feat) > self.skill_dim + 2 else 0.0))
            duration = float(params.get("c_a", feat[self.skill_dim + 3] if len(feat) > self.skill_dim + 3 else 1.0))

            if is_rest or (abs(beta) < 1e-6 and abs(duration - 1.0) < 1e-6):
                next_f = max(0.0, fatigue_est - self.rest_recovery)
            else:
                next_f = min(1.0, fatigue_est + max(beta, 0.0))

            predicted[idx] = next_f

            if is_rest or next_f <= step_cap:
                safe.append(idx)
            if is_rest or next_f <= guard_threshold:
                relaxed.append(idx)

        if self.rest_action not in safe:
            safe.append(self.rest_action)
        if self.rest_action not in relaxed:
            relaxed.append(self.rest_action)

        window = GuardWindow(
            fatigue_est=fatigue_est,
            threshold=threshold,
            guard_threshold=guard_threshold,
            step_cap=step_cap,
            safe_actions=tuple(sorted(set(safe))),
            relaxed_actions=tuple(sorted(set(relaxed))),
            predicted_next_fatigue=predicted,
        )
        self._last_window = window
        return window

    # ------------------------------------------------------------------ #
    # Guard application
    # ------------------------------------------------------------------ #
    def safe_actions(self, obs: Dict) -> Tuple[int, ...]:
        """Return the primary safe action set for logging or agent hints."""
        return self._window(obs).safe_actions

    def relaxed_actions(self, obs: Dict) -> Tuple[int, ...]:
        """Return relaxed safe actions (<= τ + δ)."""
        return self._window(obs).relaxed_actions

    def enforce(self, proposed_action: int, obs: Dict) -> Tuple[int, GuardDecision]:
        """Project an action through the guard and return the executed action."""
        window = self._window(obs)
        if not self.enabled:
            decision = GuardDecision(
                original_action=int(proposed_action),
                final_action=int(proposed_action),
                replaced=False,
                reason="disabled",
                fatigue_est=window.fatigue_est,
                threshold=window.threshold,
                guard_threshold=window.guard_threshold,
                step_cap=window.step_cap,
                predicted_next_fatigue=dict(window.predicted_next_fatigue),
            )
            self._last_decision = decision
            return proposed_action, decision

        safe = set(window.safe_actions)
        relaxed = set(window.relaxed_actions)

        final_action = int(proposed_action)
        replaced = False
        reason = "safe"

        if final_action not in safe:
            # Allow relaxed actions only when nothing else (besides REST) survives.
            non_rest_safe = [idx for idx in safe if idx != self.rest_action]
            if final_action in relaxed and not non_rest_safe:
                reason = "relaxed"
            else:
                final_action = self.rest_action
                replaced = final_action != proposed_action
                reason = "rest_fallback"

        decision = GuardDecision(
            original_action=int(proposed_action),
            final_action=int(final_action),
            replaced=replaced,
            reason=reason,
            fatigue_est=window.fatigue_est,
            threshold=window.threshold,
            guard_threshold=window.guard_threshold,
            step_cap=window.step_cap,
            predicted_next_fatigue=dict(window.predicted_next_fatigue),
        )
        self._last_decision = decision
        return final_action, decision

    def annotate_info(self, info: Dict) -> None:
        """Attach the latest decision to the environment info payload."""
        if info is None or not isinstance(info, dict):
            return
        if self._last_decision is None:
            return
        info.setdefault("guard", self._last_decision.as_dict())

