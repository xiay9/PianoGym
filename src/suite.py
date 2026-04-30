"""
PianoGym Suite
Predefined task combinations: learner profile × transfer strength × constraint setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from configs import Config


@dataclass(frozen=True)
class TaskSpec:
    name: str
    category: str
    profile: str
    transfer_strength: str
    enforce_constraint: bool
    enable_nonstationary: bool


def build_task_config(spec: TaskSpec) -> Config:
    cfg = Config()
    cfg.task_tag = spec.name
    cfg.set_transfer_strength(spec.transfer_strength)
    cfg.enable_nonstationary = spec.enable_nonstationary
    cfg.enable_fatigue_constraint = spec.enforce_constraint
    if spec.enforce_constraint:
        profile = cfg.resolve_profile(spec.profile)
        cfg.fatigue_threshold = profile.get('fatigue_threshold', cfg.fatigue_threshold)
    return cfg


def suite_specs() -> List[TaskSpec]:
    categories = [
        ('basic', 'balanced', False, False),
        ('shift', 'mild_left_weak', False, True),
        ('safe', 'severe_left_weak', True, True),
    ]
    transfer_levels = ['weak', 'medium', 'strong']

    specs: List[TaskSpec] = []
    for cat, profile, constraint, nonstationary in categories:
        for level in transfer_levels:
            name = f"{cat}-p{profile}-m{level}"
            specs.append(TaskSpec(
                name=name,
                category=cat,
                profile=profile,
                transfer_strength=level,
                enforce_constraint=constraint,
                enable_nonstationary=nonstationary,
            ))
    return specs


def build_suite() -> Iterable[tuple[TaskSpec, Config]]:
    for spec in suite_specs():
        yield spec, build_task_config(spec)
