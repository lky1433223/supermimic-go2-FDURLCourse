"""Curricula act at episode boundaries; evaluation always uses fixed parameters."""

from typing import Protocol

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


def make_curriculum(config):
    if config["name"] != "none":
        raise NotImplementedError("Only fixed difficulty is implemented; add a Curriculum implementation first")
    return FixedCurriculum()
