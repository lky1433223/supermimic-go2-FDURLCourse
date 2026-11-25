"""Episode-constant delays for batched control-step signals."""

from dataclasses import dataclass
from typing import NamedTuple

import torch


class StepDelayState(NamedTuple):
    """Per-environment delay in steps and a newest-first [env, lag, feature] queue."""

    steps: torch.Tensor
    history: torch.Tensor


@dataclass(frozen=True)
class EpisodeStepDelay:
    """Sample one lag per environment at reset and carry it through the episode."""

    min_steps: int
    max_steps: int

    @classmethod
    def from_bounds(cls, bounds: object, field: str) -> "EpisodeStepDelay | None":
        """Validate inclusive integer bounds. A zero-only range requires no queue."""
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(x) is not int for x in bounds) or not 0 <= bounds[0] <= bounds[1]):
            raise ValueError(f'{field} must be two ordered nonnegative integers')
        return cls(*bounds) if bounds[1] else None

    def reset(self, initial: torch.Tensor) -> StepDelayState:
        """Sample lags and populate the queue from the initial [env, feature] batch."""
        shape = (initial.shape[0],)
        if self.min_steps == self.max_steps:
            steps = torch.full(shape, self.min_steps, device=initial.device, dtype=torch.long)
        else:
            steps = torch.randint(self.min_steps, self.max_steps + 1, shape, device=initial.device)
        return StepDelayState(steps, initial[:, None].repeat(1, self.max_steps, 1))

    def select(self, current: torch.Tensor, state: StepDelayState) -> torch.Tensor:
        """Select the current or queued signal for each environment, with gradients."""
        lag = (state.steps - 1).clamp_min(0)
        index = lag[:, None, None].expand(-1, 1, current.shape[-1])
        past = state.history.gather(1, index).squeeze(1)
        return torch.where(state.steps[:, None] == 0, current, past)

    def advance(self, current: torch.Tensor, state: StepDelayState) -> StepDelayState:
        """Append the current signal without mutating the rollout's input state."""
        history = torch.cat((current[:, None], state.history[:, :self.max_steps - 1]), dim=1)
        return StepDelayState(state.steps, history)


def action_delay_from_config(config: dict) -> EpisodeStepDelay | None:
    """Read optional action delay bounds in whole policy steps."""
    bounds = config.get('domain_randomization', {}).get('action_delay_steps', [0, 0])
    return EpisodeStepDelay.from_bounds(bounds, 'domain_randomization.action_delay_steps')
