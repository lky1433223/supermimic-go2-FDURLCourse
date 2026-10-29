"""The same actor input construction for simulation and SDK low-level telemetry."""

import torch

from rl_go2.state import rotate_inverse

OBSERVATION_TERMS = [
    ("base_angular_velocity_body", 3), ("projected_gravity_body", 3), ("commands", 3),
    ("joint_position_offset", 12), ("joint_velocity", 12), ("previous_action", 12),
]


def policy_observation(quaternion, angular_velocity_body, joint_position, joint_velocity,
                       commands, previous_action, default_position, scales):
    gravity = torch.zeros_like(angular_velocity_body)
    gravity[:, 2] = -1
    command_scales = commands.new_tensor(scales["commands"])
    return torch.cat((angular_velocity_body * scales["angular_velocity"],
                      rotate_inverse(quaternion, gravity), commands * command_scales,
                      (joint_position - default_position) * scales["joint_position"],
                      joint_velocity * scales["joint_velocity"], previous_action), dim=-1)
