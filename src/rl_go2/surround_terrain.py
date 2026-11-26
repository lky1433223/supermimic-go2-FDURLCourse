"""Concurrent obstacle patches for the graded terrain course."""

from collections.abc import Mapping, Sequence
from math import ceil, cos, isfinite, pi, radians, sin, sqrt, tan
from random import Random

from rl_go2.scene_geometry import inclined_box


MAX_TERRAIN_LEVEL = 9
_LEVEL_ZERO_EXIT_M = 1.6
_PATCH_SPACING_M = 20.0
# Leave room for the 0.5 s upright check after reaching the elevated finish.
_LANDING_WIDTH_M = 1.2
_TERRAIN_FIELDS = {
    'step': {'height_levels_m', 'length_m'},
    'ramp': {'slope_levels_deg', 'run_m', 'platform_length_m'},
    'stairs': {'step_height_levels_m', 'tread_depth_m', 'num_steps'},
    'rough_ground': {'map_count', 'seed', 'tile_size_m', 'height_increment_m',
                     'max_height_levels', 'max_adjacent_delta_levels'},
}


def _positive_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive')
    return float(value)


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f'{label} must be a positive integer')
    return value


def _levels(parameters: Mapping, key: str, *, integers: bool = False) -> list:
    values = parameters.get(key)
    if not isinstance(values, list) or len(values) != MAX_TERRAIN_LEVEL:
        raise ValueError(f'{key} must specify {MAX_TERRAIN_LEVEL} terrain levels')
    validate = _positive_integer if integers else _positive_number
    levels = [validate(value, key) for value in values]
    if levels != sorted(levels):
        raise ValueError(f'{key} must be ordered from lowest to highest')
    return levels


def _box(
    name: str, x: float, y: float, length: float, width: float, height: float, friction: float,
) -> dict:
    return {'id': name, 'type': 'box', 'fixed': True, 'position': [x, y, height / 2],
            'rotation': [1, 0, 0, 0], 'size': [length, width, height], 'friction': friction}


def _ring_boxes(
    name: str, y: float, inner: float, outer: float, heights: Sequence[float], friction: float,
) -> list[dict]:
    """Cover a square ring with four adjacent non-overlapping rectangles."""
    center = (inner + outer) / 2
    band = outer - inner
    strips = ((0.0, center, 2 * outer, band), (center, 0.0, band, 2 * inner),
              (0.0, -center, 2 * outer, band), (-center, 0.0, band, 2 * inner))
    return [_box(f'{name}-{side}', x, y + offset_y, length, width, height, friction)
            for side, ((x, offset_y, length, width), height) in enumerate(zip(strips, heights, strict=True))
            if height > 0]


def _ramp_slabs(name: str, y: float, inner: float, run: float, slope: float, friction: float) -> list[dict]:
    """Cover all directions with uphill slabs, using the higher surface at corners."""
    radius = inner + run
    rise = run * tan(slope)
    slab = inclined_box(name, (inner + run / 2, 0.0, rise / 2), slope,
                        (run / cos(slope), 2 * radius, 0.05), friction)
    distance, _, z = slab['position']
    pitch_w, _, pitch_y, _ = slab['rotation']
    slabs = []
    for side, yaw in enumerate((0.0, pi / 2, pi, -pi / 2)):
        cy, sy = cos(yaw / 2), sin(yaw / 2)
        slabs.append({
            **slab,
            'id': f'{name}-up-{side}',
            'position': [distance * cos(yaw), y + distance * sin(yaw), z],
            'rotation': [cy * pitch_w, -sy * pitch_y, cy * pitch_y, sy * pitch_w],
        })
    return slabs


def _rough_ring_levels(rings: int, height: int, delta: int, seed: int) -> list[list[int]]:
    """Sample strip heights with bounded rises at all cardinal seams and at both edges."""
    rng = Random(seed)
    peak_ring = (rings - 1) // 2
    if height > (peak_ring + 1) * delta:
        raise ValueError('The rough band is too narrow for the maximum height and adjacent delta')
    peak = (peak_ring, rng.randrange(4))
    assigned = {peak: height}
    remaining = [(ring, side) for ring in range(rings) for side in range(4) if (ring, side) != peak]
    rng.shuffle(remaining)
    for ring, side in remaining:
        # A raised innermost ring closes flat routes out of the spawn platform.
        low = int(ring == 0)
        high = min(height, (ring + 1) * delta, (rings - ring) * delta)
        for (other_ring, other_side), value in assigned.items():
            side_distance = abs(side - other_side)
            # Adjacent sides may touch one ring in or out at a square corner.
            distance = max(abs(ring - other_ring), min(side_distance, 4 - side_distance))
            low = max(low, value - distance * delta)
            high = min(high, value + distance * delta)
        assigned[ring, side] = rng.randint(low, high)
    return [[assigned[ring, side] for side in range(4)] for ring in range(rings)]


def _height_map(
    inner_tiles: int, rings: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Expand the square strip geometry into the reusable XY height-map representation."""
    extent = inner_tiles + len(rings)
    heights = []
    for x in range(-extent, extent + 1):
        row = []
        for y in range(-extent, extent + 1):
            ring = max(abs(x), abs(y)) - inner_tiles - 1
            if ring < 0:
                row.append(0.0)
                continue
            if abs(y) >= abs(x):
                side = 0 if y > 0 else 2
            else:
                side = 1 if x > 0 else 3
            row.append(rings[ring][side])
        heights.append(row)
    return heights


def _rough_ground(
    name: str, y: float, center: float, tile: float, increment: float,
    height: int, delta: int, seed: int, friction: float,
) -> tuple[list[dict], float, float, dict]:
    # Odd grids keep a tile centered on the spawn point. Snap the platform outwards
    # to its nearest tile boundary and keep the rough band about five 0.30 m tiles wide.
    inner_tiles = max(0, ceil(center / tile - 0.5))
    inner = (inner_tiles + 0.5) * tile
    ring_count = ceil(1.5 / tile)
    outer = inner + ring_count * tile
    levels = _rough_ring_levels(ring_count, height, delta, seed)
    heights = [[value * increment for value in ring] for ring in levels]
    boxes = [box for index, ring in enumerate(heights)
             for box in _ring_boxes(f'{name}-ring-{index + 1}', y, inner + index * tile,
                                    inner + (index + 1) * tile, ring, friction)]
    metadata = {
        'seed': seed, 'height_map_m': _height_map(inner_tiles, heights), 'tile_size_m': tile,
        'start_x_m': -outer, 'center_y_m': y, 'base_z_m': 0.0,
        'max_height_m': height * increment, 'max_adjacent_delta_m': delta * increment,
    }
    return boxes, inner, outer, metadata


def _level_specs(kind: str, parameters: Mapping, seed: int) -> list[list[tuple]]:
    """Return the per-level arguments for all geometry variants of one category."""
    if kind == 'rough_ground':
        count = _positive_integer(parameters.get('map_count', 2), 'rough_ground.map_count')
        tile = _positive_number(parameters.get('tile_size_m'), 'rough_ground.tile_size_m')
        increment = _positive_number(parameters.get('height_increment_m'), 'rough_ground.height_increment_m')
        rough_seed = parameters.get('seed', seed)
        if type(rough_seed) is not int:
            raise ValueError('rough_ground.seed must be an integer')
        heights = _levels(parameters, 'max_height_levels', integers=True)
        deltas = _levels(parameters, 'max_adjacent_delta_levels', integers=True)
        return [[(tile, increment, height, delta, rough_seed + level * count + variant)
                 for variant in range(count)]
                for level, (height, delta) in enumerate(zip(heights, deltas, strict=True))]
    if kind == 'step':
        length = _positive_number(parameters.get('length_m'), 'step.length_m')
        return [[(length, height)] for height in _levels(parameters, 'height_levels_m')]
    if kind == 'stairs':
        depth = _positive_number(parameters.get('tread_depth_m'), 'stairs.tread_depth_m')
        steps = _positive_integer(parameters.get('num_steps'), 'stairs.num_steps')
        return [[(depth, steps, height)] for height in _levels(parameters, 'step_height_levels_m')]
    run = _positive_number(parameters.get('run_m'), 'ramp.run_m')
    platform = _positive_number(parameters.get('platform_length_m'), 'ramp.platform_length_m')
    slopes = _levels(parameters, 'slope_levels_deg')
    if slopes[-1] >= 90:
        raise ValueError('Ramp slopes must be less than 90 degrees')
    return [[(run, platform, radians(slope))] for slope in slopes]


def _obstacles(
    kind: str, name: str, y: float, center: float, dimensions: tuple, friction: float,
) -> tuple[list[dict], float, float, dict | None]:
    if kind == 'rough_ground':
        return _rough_ground(name, y, center, *dimensions, friction)
    if kind == 'step':
        length, height = dimensions
        outer = center + length
        return _ring_boxes(name, y, center, outer, (height,) * 4, friction), center, outer, None
    if kind == 'stairs':
        tread, count, rise = dimensions
        outer = center + count * tread
        boxes = []
        for step in range(count):
            inner = center + step * tread
            end = inner + tread + (_LANDING_WIDTH_M if step + 1 == count else 0.0)
            height = (step + 1) * rise
            boxes.extend(_ring_boxes(f'{name}-tread-{step + 1}', y, inner, end, (height,) * 4, friction))
        return boxes, center, outer, None
    run, platform, slope = dimensions
    outer = center + run + platform
    height = run * tan(slope)
    boxes = _ramp_slabs(name, y, center, run, slope, friction)
    boxes.extend(_ring_boxes(f'{name}-landing', y, center + run, outer + _LANDING_WIDTH_M,
                             (height,) * 4, friction))
    return boxes, center, outer, None


def build_curriculum_patches(
    terrains: Mapping[str, Mapping], settings: Mapping, seed: int,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Build separated geometry, spawn metadata, and rough height maps for levels 0–9."""
    if not isinstance(settings, Mapping):
        raise ValueError('terrain_curriculum must be a mapping')
    center = _positive_number(settings.get('spawn_half_width_m', 1.0), 'terrain_curriculum.spawn_half_width_m')
    if terrains.keys() - _TERRAIN_FIELDS.keys():
        raise ValueError('Terrain curriculum level 0 provides the flat ground for each obstacle type')
    patch_specs = []
    for kind, parameters in terrains.items():
        if not isinstance(parameters, Mapping) or parameters.keys() - (_TERRAIN_FIELDS[kind] | {'weight', 'friction'}):
            raise ValueError(f'Unsupported curriculum terrain settings for {kind}')
        weight = _positive_number(parameters.get('weight', 1.0), f'{kind}.weight')
        friction = parameters.get('friction', 0.8)
        variants = [[()], *_level_specs(kind, parameters, seed)]
        patch_weight = weight / sum(len(group) for group in variants)
        for level, group in enumerate(variants):
            for variant, dimensions in enumerate(group):
                patch_specs.append((kind, level, variant, dimensions, patch_weight, friction))

    # Genesis's default floor has finite collision bounds. Spread patches over
    # both axes to keep the course near the origin even with several variants.
    columns = max(1, ceil(sqrt(len(patch_specs))))
    rows = ceil(len(patch_specs) / columns)
    boxes = []
    patches = []
    height_maps = {}
    for index, (kind, level, variant, dimensions, patch_weight, friction) in enumerate(patch_specs):
        row, column = divmod(index, columns)
        x = (column - (columns - 1) / 2) * _PATCH_SPACING_M
        y = (row - (rows - 1) / 2) * _PATCH_SPACING_M
        name = f'{kind}-level-{level}-{variant}'
        inner, outer = center, max(center + 0.1, _LEVEL_ZERO_EXIT_M)
        if level:
            obstacles, inner, outer, height_map = _obstacles(kind, name, y, center, dimensions, friction)
            for box in obstacles:
                box['position'][0] += x
            boxes.extend(obstacles)
            if height_map is not None:
                height_map['start_x_m'] += x
                height_maps[name] = height_map
        patches.append({
            'name': name, 'type': kind, 'level': level, 'weight': patch_weight,
            'position': [x, y, 0.42], 'rotation': [1, 0, 0, 0],
            'inner_half_width_m': inner, 'outer_half_width_m': outer,
        })
    return boxes, patches, height_maps
