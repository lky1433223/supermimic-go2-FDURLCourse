"""Resolve authored terrain into fixed boxes in the scene's Z-up frame."""

from collections.abc import Mapping
import math
from random import Random


STEP_FIELDS = {"length_m", "width_m", "height_m", "start_x_m", "center_y_m", "base_z_m"}
STAIRS_FIELDS = {"step_height_m", "tread_depth_m", "num_steps", "width_m",
                 "start_x_m", "center_y_m", "base_z_m"}
RAMP_FIELDS = {"run_m", "width_m", "platform_length_m", "thickness_m", "slope_deg",
               "start_x_m", "center_y_m", "base_z_m"}
ROUGH_GROUND_INTEGERS = {"rows", "columns", "seed", "max_height_level", "max_adjacent_delta_levels"}
ROUGH_GROUND_FIELDS = ROUGH_GROUND_INTEGERS | {
    "tile_size_m", "height_increment_m", "start_x_m", "center_y_m", "base_z_m",
}


def terrain_parameters(entity: dict, overrides: Mapping[str, float], fields: set[str]) -> dict:
    """Validate the declared units-bearing fields and apply instance overrides."""
    kind, entity_id = entity["type"], entity["id"]
    if entity.keys() - (fields | {"id", "type", "friction"}) or overrides.keys() - fields:
        raise ValueError(f"Unsupported {kind} fields or overrides on {entity_id}")
    values = {**entity, **overrides}
    if any(type(values.get(key)) not in (int, float) or not math.isfinite(values[key]) for key in fields):
        raise ValueError(f"{kind.capitalize()} {entity_id} must declare finite dimensions and position")
    return values


def step_box(entity: dict, overrides: Mapping[str, float]) -> dict:
    """Resolve an axis-aligned step with its front edge and base held fixed."""
    values = terrain_parameters(entity, overrides, STEP_FIELDS)
    length, width, height = (values[key] for key in ("length_m", "width_m", "height_m"))
    if min(length, width, height) <= 0:
        raise ValueError(f"Step {entity['id']} dimensions must be positive")
    return {
        "id": entity["id"], "type": "box", "fixed": True,
        "position": [values["start_x_m"] + length / 2, values["center_y_m"], values["base_z_m"] + height / 2],
        "size": [length, width, height], "friction": values.get("friction", 0.8),
    }


def stairs_boxes(entity: dict, overrides: Mapping[str, float]) -> list[dict]:
    """Build adjacent solid treads ascending along world +X from the base plane.

    Each tread has the same depth and rise. The last tread ends at the
    configured depth of the flight, with a vertical drop to the base plane.
    """
    values = terrain_parameters(entity, overrides, STAIRS_FIELDS)
    count = values['num_steps']
    if type(count) is not int or count < 1:
        raise ValueError(f"Stairs {entity['id']} num_steps must be a positive integer")
    rise, depth, width = (values[key] for key in ('step_height_m', 'tread_depth_m', 'width_m'))
    if min(rise, depth, width) <= 0:
        raise ValueError(f"Stairs {entity['id']} dimensions must be positive")
    return [step_box({
        'id': f"{entity['id']}-{index}", 'type': 'step',
        'start_x_m': values['start_x_m'] + index * depth,
        'center_y_m': values['center_y_m'], 'base_z_m': values['base_z_m'],
        'length_m': depth, 'width_m': width, 'height_m': (index + 1) * rise,
        'friction': values.get('friction', 0.8),
    }, {}) for index in range(count)]


def inclined_box(name: str, top_center_m: tuple[float, float, float], slope_rad: float,
                 size_m: tuple[float, float, float], friction: float) -> dict:
    """Place a fixed slab by the center of its top face, with rise along +X."""
    x, y, z = top_center_m
    half_thickness = size_m[2] / 2
    # A rising surface has normal (-sin(slope), 0, cos(slope)).
    return {
        "id": name, "type": "box", "fixed": True, "friction": friction,
        "position": [x + half_thickness * math.sin(slope_rad), y, z - half_thickness * math.cos(slope_rad)],
        "rotation": [math.cos(slope_rad / 2), 0, -math.sin(slope_rad / 2), 0],
        "size": list(size_m),
    }


def ramp_boxes(entity: dict, overrides: Mapping[str, float]) -> tuple[dict, dict, dict]:
    """Build an uphill slab, a level deck and a downhill slab along +X."""
    values = terrain_parameters(entity, overrides, RAMP_FIELDS)
    run, width, deck, thickness = (values[key] for key in ("run_m", "width_m", "platform_length_m", "thickness_m"))
    slope_deg = values["slope_deg"]
    name = entity["id"]
    if min(run, width, deck, thickness) <= 0 or not 0 < slope_deg < 90:
        raise ValueError(f"Ramp {name} needs positive dimensions and a slope between 0 and 90 degrees")

    slope = math.radians(slope_deg)
    rise = run * math.tan(slope)
    x, y, z = (values[key] for key in ("start_x_m", "center_y_m", "base_z_m"))
    friction = values.get("friction", 0.8)
    slab = (run / math.cos(slope), width, thickness)
    return (
        inclined_box(f"{name}-up", (x + run / 2, y, z + rise / 2), slope, slab, friction),
        inclined_box(f"{name}-platform", (x + run + deck / 2, y, z + rise), 0, (deck, width, rise), friction),
        inclined_box(f"{name}-down", (x + run + deck + run / 2, y, z + rise / 2), -slope, slab, friction),
    )


def sample_height_levels(rows: int, columns: int, max_height: int, max_delta: int, seed: int) -> list[list[int]]:
    """Sample a grid bounded by its cardinal neighbors and level entry and exit."""
    rng = Random(seed)
    levels: list[list[int]] = []
    for row_index in range(rows):
        # Leave enough rows to reach level zero at both ends of the patch.
        row_max = min(max_height, (row_index + 1) * max_delta, (rows - row_index) * max_delta)
        row: list[int] = []
        for column in range(columns):
            neighbors = []
            if row_index:
                neighbors.append(levels[-1][column])
            if column:
                neighbors.append(row[-1])
            lower = max(0, max(neighbors, default=0) - max_delta)
            upper = min(row_max, min(neighbors, default=row_max) + max_delta)
            row.append(rng.randint(lower, upper))
        levels.append(row)
    if not any(map(any, levels)):
        levels[rng.randrange(rows)][rng.randrange(columns)] = 1
    return levels


def rough_ground_boxes(entity: dict, overrides: Mapping[str, float]) -> tuple[list[dict], dict]:
    """Create tiles above the supporting plane and return the resolved height map in metres."""
    values = terrain_parameters(entity, overrides, ROUGH_GROUND_FIELDS)
    name = entity["id"]
    if any(type(values[key]) is not int for key in ROUGH_GROUND_INTEGERS):
        raise ValueError(f"Rough ground {name} requires integer dimensions in tiles, seed and height levels")
    rows, columns = values["rows"], values["columns"]
    max_height, max_delta = values["max_height_level"], values["max_adjacent_delta_levels"]
    tile, increment = values["tile_size_m"], values["height_increment_m"]
    if min(rows, columns, max_height, max_delta, tile, increment) <= 0:
        raise ValueError(f"Rough ground {name} requires positive dimensions, height and adjacent delta")

    seed = values["seed"]
    heights = [[level * increment for level in row]
               for row in sample_height_levels(rows, columns, max_height, max_delta, seed)]
    start_x, center_y, base_z = (values[key] for key in ("start_x_m", "center_y_m", "base_z_m"))
    start_y = center_y - columns * tile / 2
    boxes = []
    for row, row_heights in enumerate(heights):
        for column, height in enumerate(row_heights):
            if height == 0:
                continue
            boxes.append({
                "id": f"{name}-{row}-{column}", "type": "box", "fixed": True,
                "position": [start_x + (row + 0.5) * tile, start_y + (column + 0.5) * tile, base_z + height / 2],
                "size": [tile, tile, height], "friction": values.get("friction", 0.8),
            })
    metadata = {"seed": seed, "height_map_m": heights, "tile_size_m": tile,
                "start_x_m": start_x, "center_y_m": center_y, "base_z_m": base_z,
                "max_height_m": max_height * increment, "max_adjacent_delta_m": max_delta * increment}
    return boxes, metadata


def resolve_terrain_entities(
    entities: list[dict], overrides: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict], dict[str, dict]]:
    """Expand terrain declarations and return geometry and generated height maps."""
    names = [entity["id"] for entity in entities]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate authored scene entity id")
    remaining = dict(overrides)
    resolved = []
    height_maps = {}
    for entity in entities:
        kind, name = entity["type"], entity["id"]
        if kind == "step":
            resolved.append(step_box(entity, remaining.pop(name, {})))
        elif kind == "stairs":
            resolved.extend(stairs_boxes(entity, remaining.pop(name, {})))
        elif kind == "ramp":
            resolved.extend(ramp_boxes(entity, remaining.pop(name, {})))
        elif kind == "rough_ground":
            boxes, height_maps[name] = rough_ground_boxes(entity, remaining.pop(name, {}))
            resolved.extend(boxes)
        else:
            resolved.append(entity)
    if remaining:
        raise ValueError(f"Overrides require named terrain entities: {sorted(remaining)}")
    return resolved, height_maps
