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

    def where(self, mask: torch.Tensor, fallback: "RobotState") -> "RobotState":
        """Use this state on true environment rows and the fallback elsewhere."""
        return RobotState(*(torch.where(mask[:, None], getattr(self, f.name), getattr(fallback, f.name))
                            for f in fields(self)))

    def select(self, indices: torch.Tensor) -> "RobotState":
        """Select the supplied batch rows without changing state fields."""
        return RobotState(*(getattr(self, f.name)[indices] for f in fields(self)))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two rotation quaternions in wxyz order."""
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1)), dim=-1)


def rotate_inverse(quaternion, vector):
    """Rotate world-frame vectors into a body's frame (unit wxyz quaternion)."""
    xyz = quaternion[..., 1:]
    t = 2 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector - quaternion[..., :1] * t + torch.linalg.cross(xyz, t, dim=-1)
