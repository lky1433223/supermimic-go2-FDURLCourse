"""Curricula act at episode boundaries; evaluation always uses fixed parameters."""

from typing import Protocol
from bisect import bisect_right

import torch


class Curriculum(Protocol):
    def on_reset(self, env_ids: torch.Tensor, *, environment_steps: int, metrics: dict) -> dict:
        """Return task parameter updates for resetting environments, never mutate physics mid-step."""
        ...

    def state_dict(self) -> dict: ...


class FixedCurriculum:
    def on_reset(self, env_ids, *, environment_steps, metrics):
        return {}

    def state_dict(self):
        return {"name": "none"}


class PushForceCurriculum:
    """Progress by collected transitions; a new range takes effect only on episode reset."""

    def __init__(self, config):
        from rl_go2.tasks.push_walk import validate_force_range
        self.stages = config['stages']
        self.thresholds = [stage['after_steps'] for stage in self.stages]
        if (not self.thresholds or self.thresholds[0] != 0 or
                any(not isinstance(x, int) or x < 0 for x in self.thresholds) or
                any(b <= a for a, b in zip(self.thresholds, self.thresholds[1:]))):
            raise ValueError('Curriculum thresholds must start at zero and strictly increase')
        for stage in self.stages:
            validate_force_range(stage['force_range_n'])
        self.stage = 0

    def on_reset(self, env_ids, *, environment_steps, metrics):
        self.stage = bisect_right(self.thresholds, environment_steps) - 1
        return {'push_force_range': self.stages[self.stage]['force_range_n']}

    def state_dict(self):
        return {'name': 'push_force', 'stage': self.stage,
                'force_range_n': self.stages[self.stage]['force_range_n']}


def make_curriculum(config):
    if config['name'] == 'push_force':
        return PushForceCurriculum(config)
    if config["name"] != "none":
        raise NotImplementedError(f"Unknown curriculum {config['name']!r}; available: none, push_force")
    return FixedCurriculum()
