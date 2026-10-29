"""Reserved API only; no SuperMimic reproduction or training is implemented."""

from typing import Protocol

import torch

from rl_go2.env import Transition
from rl_go2.state import RobotState


class WorldModel(Protocol):
    def predict(self, state: RobotState, action: torch.Tensor, context: dict) -> RobotState: ...

    def update(self, transitions: list[Transition]) -> dict[str, float]: ...


def train(config, output):
    raise NotImplementedError(
        "SuperMimic is an interface only. Implement world-model fitting and imagined-rollout policy updates first."
    )
