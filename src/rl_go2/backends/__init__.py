from dataclasses import dataclass
from typing import Protocol

import torch

from rl_go2.state import RobotState, rotate_inverse


@dataclass(frozen=True)
class LinkPointForces:
    """Point forces for one physics step, indexed by environment and backend link_names.

    Both tensors have shape (num_envs, num_links, 3), with robot links sorted
    by name. Points are offsets in meters from the authored link origins.
    Force vectors are in world-frame newtons. Use a zero force for any link
    without a load.
    """

    point_local_m: torch.Tensor
    force_world_n: torch.Tensor

    def validate(self, num_envs: int, num_links: int) -> None:
        """Reject malformed or non-finite point and force batches."""
        shape = (num_envs, num_links, 3)
        for name, value in (("point_local_m", self.point_local_m), ("force_world_n", self.force_world_n)):
            if value.shape != shape or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be a finite floating-point tensor of shape {shape}")

    def wrenches_world(self, link_quaternions: torch.Tensor) -> torch.Tensor:
        """Return world-frame force and moment about each link origin, in N and N·m."""
        inverse = torch.cat((link_quaternions[..., :1], -link_quaternions[..., 1:]), dim=-1)
        offsets = rotate_inverse(inverse, self.point_local_m)
        moments = torch.linalg.cross(offsets, self.force_world_n, dim=-1)
        return torch.cat((self.force_world_n, moments), dim=-1)


class PhysicsBackend(Protocol):
    num_envs: int
    device: torch.device
    link_names: tuple[str, ...]

    def read_state(self) -> RobotState: ...
    def read_link_quaternions(self) -> torch.Tensor:
        """Return world orientations of authored link frames as (num_envs, num_links, 4), wxyz."""
        ...
    def reset(self, env_ids: torch.Tensor) -> None: ...
    def set_foot_ground_friction(self, base: torch.Tensor, multipliers: torch.Tensor) -> None: ...
    def set_foot_friction_multipliers(self, multipliers: torch.Tensor, env_ids: torch.Tensor) -> None: ...
    def set_base_com_offset(self, offsets: torch.Tensor) -> None: ...
    def set_base_mass_scale(self, scales: torch.Tensor) -> torch.Tensor: ...
    def step(
        self,
        torque: torch.Tensor,
        *,
        base_force: torch.Tensor | None = None,
        link_point_forces: LinkPointForces | None = None,
    ) -> None:
        """Advance one physics step with optional external forces.

        base_force has shape (num_envs, 3) in world-frame newtons and acts
        at the base-link center of mass. It adds to any link point forces.
        """
        ...
    def close(self) -> None: ...


def make_backend(config, viewer=False):
    name = config["backend"]
    if name == "isaac":
        raise NotImplementedError("Isaac Sim backend is reserved; no runtime is installed")
    if name == "genesis":
        from .genesis import GenesisBackend
        cls = GenesisBackend
    elif name == "mujoco":
        from .mujoco import MujocoBackend
        cls = MujocoBackend
    else:
        raise ValueError(f"Unknown backend: {name}")
    entities = [e for e in config["scene"]["entities"] if e.get("robot")]
    if len(entities) != 1 or entities[0]["id"] != config["robot"]["entity_id"]:
        raise ValueError("Expected exactly one robot matching robot.entity_id")
    from pathlib import Path
    if not Path(entities[0]["file"]).is_file():
        raise FileNotFoundError("Go2 asset missing. Run: uv run python scripts/fetch_go2.py")
    return cls(config, viewer=viewer)
