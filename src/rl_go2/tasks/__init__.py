"""Task selection and lifecycle contract. Adding a task does not change the environment."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch
    from rl_go2.state import RobotState


TASK_TYPES = {"flat_walk": "rl_go2.tasks.flat_walk:FlatWalkTask",
              "push_walk": "rl_go2.tasks.push_walk:PushWalkTask"}


class Task(Protocol):
    observation_terms: list[tuple[str, int]]

    def __init__(self, config: dict, default_positions: torch.Tensor, num_envs: int,
                 device: torch.device, dt: float, seed: int) -> None: ...

    @staticmethod
    def validate_config(config: dict, robot: dict) -> None: ...

    def reset(self, env_ids: torch.Tensor, state: RobotState, updates: dict) -> None: ...
    def observe(self, state: RobotState, previous_action: torch.Tensor) -> torch.Tensor: ...
    def reward(self, state: RobotState, action: torch.Tensor,
               previous_action: torch.Tensor, torque: torch.Tensor) -> torch.Tensor: ...
    def terminated(self, state: RobotState) -> torch.Tensor: ...
    def external_force(self, state: RobotState, episode_steps: torch.Tensor) -> torch.Tensor | None:
        """World-frame base COM force held over the next control step; never an actor input."""
        ...
    def on_step_end(self, state: RobotState, episode_steps: torch.Tensor) -> None:
        """Update task state for the next observation, after automatic resets."""
        ...

    def context(self) -> dict[str, torch.Tensor]:
        """Task-specific goals/phase used for optional model-learning transitions."""
        ...


def validate_task_name(name):
    if not isinstance(name, str) or name not in TASK_TYPES:
        raise ValueError(f"Unknown task {name!r}; available tasks: {', '.join(sorted(TASK_TYPES))}")


def get_task_type(name) -> type[Task]:
    validate_task_name(name)
    module, attribute = TASK_TYPES[name].split(":")
    return getattr(import_module(module), attribute)
