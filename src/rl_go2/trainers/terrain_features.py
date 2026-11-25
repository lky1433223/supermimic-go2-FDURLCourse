"""Local height features with a static spatial index over collision surfaces."""


import torch
from torch import nn

from rl_go2.state import rotate_inverse


def _surface_index(boxes: list[dict], cell_size_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Index conservative box footprints, including both sides of cell boundaries."""
    footprints = []
    for box in boxes:
        quaternion = torch.tensor(box['rotation'], dtype=torch.float32)
        # Rows are local axes expressed in world coordinates.
        inverse = quaternion.clone()
        inverse[1:] *= -1
        axes = rotate_inverse(inverse.expand(3, -1), torch.eye(3))
        radius = (axes.abs() * torch.tensor(box['size'])[:, None] / 2).sum(0)
        center = torch.tensor(box['position'])
        lower = torch.floor((center[:2] - radius[:2]) / cell_size_m).long()
        upper = torch.floor((center[:2] + radius[:2]) / cell_size_m).long()
        footprints.append((lower.tolist(), upper.tolist()))
    if not footprints:
        return torch.zeros((1, 1, 1), dtype=torch.long), torch.zeros(2, dtype=torch.long)
    origin = [min(bounds[0][axis] for bounds in footprints) for axis in range(2)]
    extent = [max(bounds[1][axis] for bounds in footprints) - origin[axis] + 1 for axis in range(2)]
    cells: dict[tuple[int, int], list[int]] = {}
    for surface, (lower, upper) in enumerate(footprints, 1):
        for x in range(lower[0], upper[0] + 1):
            for y in range(lower[1], upper[1] + 1):
                cells.setdefault((x - origin[0], y - origin[1]), []).append(surface)
    width = max(map(len, cells.values()))
    index = torch.zeros((*extent, width), dtype=torch.long)
    for (x, y), surfaces in cells.items():
        index[x, y, :len(surfaces)] = torch.tensor(surfaces)
    return index, torch.tensor(origin, dtype=torch.long)


def _surface_equations(boxes: list[dict], ground_z: float) -> tuple[torch.Tensor, ...]:
    """Precompute vertical intersection bounds and side constraints for each box.

    Each coefficient triplet evaluates a*x + b*y + c. Surface zero is
    the ground, also used for padding spatial cells and out-of-map queries.
    """
    lower = [torch.tensor([[0., 0., -torch.inf]]).expand(3, -1)]
    upper = [torch.tensor([[0., 0., ground_z]]).expand(3, -1)]
    sides = [torch.zeros(3, 3)]
    limits = [torch.full((3,), torch.inf)]
    for box in boxes:
        quaternion = torch.tensor(box['rotation'], dtype=torch.float32)
        inverse = quaternion.clone()
        inverse[1:] *= -1
        axes = rotate_inverse(inverse.expand(3, -1), torch.eye(3))
        center = torch.tensor(box['position'], dtype=torch.float32)
        half_size = torch.tensor(box['size'], dtype=torch.float32) / 2
        projection = (axes * center).sum(-1)
        vertical = axes[:, 2]
        parallel = vertical.abs() < 1e-6
        divisor = torch.where(parallel, torch.ones_like(vertical), vertical)
        coefficients = -axes[:, :2] / divisor[:, None]
        low = (projection - half_size * vertical.sign()) / divisor
        high = (projection + half_size * vertical.sign()) / divisor
        low_planes = torch.cat((coefficients, low[:, None]), -1)
        high_planes = torch.cat((coefficients, high[:, None]), -1)
        low_planes[parallel] = torch.tensor([0., 0., -torch.inf])
        high_planes[parallel] = torch.tensor([0., 0., torch.inf])
        lower.append(low_planes)
        upper.append(high_planes)
        sides.append(torch.cat((axes[:, :2], -projection[:, None]), -1))
        limits.append(torch.where(parallel, half_size, torch.inf))
    return torch.stack(lower), torch.stack(upper), torch.stack(sides), torch.stack(limits)


class TerrainHeightScan(nn.Module):
    """Sample a heading-aligned 9×6 grid around the base, in metres."""

    def __init__(self, scene: dict) -> None:
        super().__init__()
        self.size = 54
        self.cell_size_m = 0.30
        x = torch.arange(-3, 6, dtype=torch.float32) / 10
        y = torch.arange(-5, 6, 2, dtype=torch.float32) / 20
        xy = torch.cartesian_prod(x, y)
        self.register_buffer('offsets', torch.cat((xy, xy.new_zeros((self.size, 1))), 1), persistent=False)
        planes = [entity for entity in scene['entities'] if entity['type'] == 'plane']
        if len(planes) != 1 or any(planes[0]['rotation'][1:3]):
            raise ValueError('Terrain height scans require one horizontal support plane')
        ground_z = planes[0]['position'][2]
        self.register_buffer('ground_z', torch.tensor(ground_z, dtype=torch.float32), persistent=False)
        boxes = [entity for entity in scene['entities'] if entity['type'] == 'box']
        index, origin = _surface_index(boxes, self.cell_size_m)
        self.register_buffer('surface_index', index, persistent=False)
        self.register_buffer('index_origin', origin, persistent=False)
        self.register_buffer('index_extent', torch.tensor(index.shape[:2]), persistent=False)
        for name, value in zip(('lower_planes', 'upper_planes', 'side_planes', 'side_limits'),
                               _surface_equations(boxes, ground_z)):
            self.register_buffer(name, value, persistent=False)

    def heights(self, world_xy: torch.Tensor) -> torch.Tensor:
        """Return exact surface Z for [..., 2] world points, including box side caps."""
        cells = torch.floor(world_xy / self.cell_size_m).long() - self.index_origin
        inside = ((cells >= 0) & (cells < self.index_extent)).all(-1)
        safe_cells = cells.clamp_min(0).minimum(self.index_extent - 1)
        candidates = self.surface_index[safe_cells[..., 0], safe_cells[..., 1]]
        candidates = torch.where(inside[..., None], candidates, 0)
        point = world_xy[..., None, None, :]
        lower = self.lower_planes[candidates]
        upper = self.upper_planes[candidates]
        sides = self.side_planes[candidates]
        low_z = ((lower[..., :2] * point).sum(-1) + lower[..., 2]).amax(-1)
        high_z = ((upper[..., :2] * point).sum(-1) + upper[..., 2]).amin(-1)
        side_distance = ((sides[..., :2] * point).sum(-1) + sides[..., 2]).abs()
        intersects = (low_z <= high_z) & (side_distance <= self.side_limits[candidates]).all(-1)
        return torch.where(intersects, high_z, self.ground_z).amax(-1).maximum(self.ground_z)

    def forward(self, position: torch.Tensor, inverse_heading: torch.Tensor) -> torch.Tensor:
        """Return ground minus base height as [batch, 54], in X-major order.

        Position is the world-frame base origin [batch, 3]. Inverse heading
        is the conjugate of the yaw-only wxyz quaternion [batch, 4].
        """
        offsets = self.offsets.expand(position.shape[0], -1, -1)
        world_offsets = rotate_inverse(inverse_heading[:, None, :].expand(-1, self.size, -1), offsets)
        world_xy = position[:, None, :2] + world_offsets[..., :2]
        return self.heights(world_xy) - position[:, 2:3]
