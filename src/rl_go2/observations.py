"""The same actor input construction for simulation and SDK low-level telemetry."""

from typing import NamedTuple

import torch

from rl_go2.state import RobotState, rotate_inverse


def project_gravity(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the world down unit vector in the body frame for batched wxyz attitudes."""
    gravity = torch.zeros_like(quaternion[..., 1:])
    gravity[..., 2] = -1
    return rotate_inverse(quaternion, gravity)


class SensorReadings(NamedTuple):
    """Batched body gyro, wxyz attitude, joint angle, joint speed and body gravity direction."""

    angular_velocity: torch.Tensor
    attitude: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    projected_gravity: torch.Tensor

    @classmethod
    def from_state(cls, state: RobotState) -> "SensorReadings":
        """Convert simulation ground truth to the frames used by Go2 telemetry."""
        return cls(rotate_inverse(state.quaternion, state.angular_velocity), state.quaternion,
                   state.joint_position, state.joint_velocity, project_gravity(state.quaternion))

OBSERVATION_TERMS = [
    ("base_angular_velocity_body", 3), ("projected_gravity_body", 3), ("commands", 3),
    ("joint_position_offset", 12), ("joint_velocity", 12), ("previous_action", 12),
]


def policy_observation(quaternion, angular_velocity_body, joint_position, joint_velocity,
                       commands, previous_action, default_position, scales, *,
                       projected_gravity: torch.Tensor | None = None) -> torch.Tensor:
    """Build the actor input, optionally using a precomputed noisy gravity observation."""
    if projected_gravity is None:
        projected_gravity = project_gravity(quaternion)
    command_scales = commands.new_tensor(scales["commands"])
    return torch.cat((angular_velocity_body * scales["angular_velocity"],
                      projected_gravity, commands * command_scales,
                      (joint_position - default_position) * scales["joint_position"],
                      joint_velocity * scales["joint_velocity"], previous_action), dim=-1)
