"""Resolve authored terrain into fixed boxes in the scene's Z-up frame."""

from collections.abc import Mapping
import math


STEP_FIELDS = {"length_m", "width_m", "height_m", "start_x_m", "center_y_m", "base_z_m"}
RAMP_FIELDS = {"run_m", "width_m", "platform_length_m", "thickness_m", "slope_deg",
               "start_x_m", "center_y_m", "base_z_m"}


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


def resolve_terrain_entities(entities: list[dict], overrides: Mapping[str, Mapping[str, float]]) -> list[dict]:
    """Expand terrain declarations and reject overrides for unknown or fixed entities."""
    names = [entity["id"] for entity in entities]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate authored scene entity id")
    remaining = dict(overrides)
    resolved = []
    for entity in entities:
        kind, name = entity["type"], entity["id"]
        if kind == "step":
            resolved.append(step_box(entity, remaining.pop(name, {})))
        elif kind == "ramp":
            resolved.extend(ramp_boxes(entity, remaining.pop(name, {})))
        else:
            resolved.append(entity)
    if remaining:
        raise ValueError(f"Overrides require named step or ramp entities: {sorted(remaining)}")
    return resolved
