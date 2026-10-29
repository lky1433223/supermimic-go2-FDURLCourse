from dataclasses import dataclass, fields

import torch


@dataclass
class RobotState:
    """Batched state: base origin in world frame, quaternion wxyz, joints in canonical order."""

    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor

    def clone(self):
        return RobotState(**{f.name: getattr(self, f.name).detach().clone() for f in fields(self)})


def rotate_inverse(quaternion, vector):
    """Rotate world-frame vectors into a body's frame (unit wxyz quaternion)."""
    xyz = quaternion[..., 1:]
    t = 2 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector - quaternion[..., :1] * t + torch.linalg.cross(xyz, t, dim=-1)
