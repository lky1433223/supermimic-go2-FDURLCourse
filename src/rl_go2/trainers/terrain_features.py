"""Local height features read from the static scene used for simulation."""

import torch
from torch import nn

from rl_go2.state import rotate_inverse


class TerrainHeightScan(nn.Module):
    """Sample a heading-aligned 9×6 grid around the base, in metres."""

    def __init__(self, scene: dict) -> None:
        super().__init__()
        self.size = 54
        x = torch.arange(-3, 6, dtype=torch.float32) / 10
        y = torch.arange(-5, 6, 2, dtype=torch.float32) / 20
        xy = torch.cartesian_prod(x, y)
        offsets = torch.cat((xy, xy.new_zeros((self.size, 1))), dim=1)
        self.register_buffer('offsets', offsets, persistent=False)

        planes = [entity for entity in scene['entities'] if entity['type'] == 'plane']
        if len(planes) != 1 or any(planes[0]['rotation'][1:3]):
            raise ValueError('Terrain height scans require one horizontal support plane')
        ground_z = planes[0]['position'][2]
        self.register_buffer('ground_z', torch.tensor(ground_z, dtype=torch.float32), persistent=False)

        boxes = [entity for entity in scene['entities'] if entity['type'] == 'box']
        self.register_buffer('centers', torch.tensor([box['position'] for box in boxes], dtype=torch.float32).reshape(-1, 3), persistent=False)
        self.register_buffer('half_sizes', torch.tensor([box['size'] for box in boxes], dtype=torch.float32).reshape(-1, 3) / 2, persistent=False)
        self.register_buffer('rotations', torch.tensor([box['rotation'] for box in boxes], dtype=torch.float32).reshape(-1, 4), persistent=False)

    def forward(self, position: torch.Tensor, inverse_heading: torch.Tensor) -> torch.Tensor:
        """Return ground minus base height as [batch, 54], in X-major order.

        Position is the world-frame base origin [batch, 3]. Inverse heading
        is the conjugate of the yaw-only wxyz quaternion [batch, 4].
        """
        offsets = self.offsets.expand(position.shape[0], -1, -1)
        world_offsets = rotate_inverse(inverse_heading[:, None, :].expand(-1, self.size, -1), offsets)
        world_xy = position[:, None, :2] + world_offsets[..., :2]
        surface = torch.zeros_like(world_xy[..., 0]) + self.ground_z
        world_points = torch.cat((world_xy, torch.zeros_like(world_xy[..., :1])), dim=-1)
        for index in range(self.centers.shape[0]):
            quaternion = self.rotations[index].expand(position.shape[0], self.size, -1)
            local_origin = rotate_inverse(quaternion, world_points - self.centers[index])
            direction = torch.zeros_like(local_origin)
            direction[..., 2] = 1
            local_direction = rotate_inverse(quaternion, direction)
            parallel = local_direction.abs() < 1e-6
            safe_direction = torch.where(parallel, torch.ones_like(local_direction), local_direction)
            first = (-self.half_sizes[index] - local_origin) / safe_direction
            second = (self.half_sizes[index] - local_origin) / safe_direction
            lower = torch.where(parallel, -torch.inf, first.minimum(second)).amax(dim=-1)
            upper = torch.where(parallel, torch.inf, first.maximum(second)).amin(dim=-1)
            parallel_inside = (~parallel | (local_origin.abs() <= self.half_sizes[index])).all(dim=-1)
            intersects = parallel_inside & (lower <= upper)
            surface = surface.maximum(torch.where(intersects, upper, self.ground_z))
        return surface - position[:, 2:3]
