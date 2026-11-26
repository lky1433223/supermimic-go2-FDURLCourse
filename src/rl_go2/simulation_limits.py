"""Limits for identifying invalid real-simulator transitions."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math

import torch

from rl_go2.state import RobotState


@dataclass(frozen=True)
class SimulationStateLimits:
    """Upper limits on terrain-relative height and world-frame speed."""

    max_base_height_m: float
    max_linear_speed_mps: float
    max_vertical_speed_mps: float

    @classmethod
    def from_config(cls, settings: dict | None) -> 'SimulationStateLimits | None':
        """Read all three positive limits when simulation filtering is enabled."""
        if settings is None:
            return None
        names = {field.name for field in fields(cls)}
        if not isinstance(settings, Mapping) or settings.keys() != names:
            raise ValueError(f'simulation_state_limits requires {", ".join(sorted(names))}')
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in settings.values()):
            raise ValueError('simulation_state_limits values must be finite and positive')
        return cls(**settings)

    def classify(self, state: RobotState, clearance_m: torch.Tensor,
                 invalid_numeric: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-environment reason masks; overlapping reasons are allowed."""
        velocity = state.linear_velocity
        usable = ~invalid_numeric
        return {
            'nonfinite_or_degenerate': invalid_numeric | ~clearance_m.isfinite(),
            'height': usable & (clearance_m > self.max_base_height_m),
            'linear_speed': usable & (velocity.square().sum(dim=-1) > self.max_linear_speed_mps**2),
            'vertical_speed': usable & (velocity[:, 2].abs() > self.max_vertical_speed_mps),
        }


def invalid_numeric_state(state: RobotState) -> torch.Tensor:
    """Reject nonfinite values and quaternions too small to represent rotation."""
    components = (state.position, state.quaternion, state.linear_velocity, state.angular_velocity,
                  state.joint_position, state.joint_velocity)
    finite = torch.cat(components, dim=-1).isfinite().all(dim=-1)
    norm_squared = state.quaternion.square().sum(dim=-1)
    return ~finite | ~norm_squared.isfinite() | (norm_squared <= 1e-12)
