"""Small YAML loader; all authored asset paths are relative to their YAML."""

from collections.abc import Mapping
import math
from pathlib import Path

import yaml

from rl_go2.scene_geometry import resolve_terrain_entities
from rl_go2.tasks import validate_task_name

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAT_SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "flat.yaml"
FOOT_NAMES = ("FR", "FL", "RR", "RL")
FRICTION_TERMS = ("sliding", "torsional", "rolling")
MODEL_PARAMETER_KEYS = ("foot_ground_friction", "base_mass_kg", "base_com_offset_m")


def base_com_offset_ranges(config):
    domain = config.get("domain_randomization", {})
    if not isinstance(domain, dict):
        raise ValueError("domain_randomization must be a mapping")
    section = domain.get("base_com", {})
    if not isinstance(section, dict) or not isinstance(section.get("enabled", False), bool):
        raise ValueError("base_com must be a mapping with a boolean enabled flag")
    if not section.get("enabled", False):
        return None
    ranges = section.get("offset_ranges_m")
    if (not isinstance(ranges, dict) or set(ranges) != {"x", "y", "z"}
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                   or not all(type(x) in (int, float) and abs(x) <= 1e30 for x in pair)
                   or pair[0] > pair[1] for pair in ranges.values())):
        raise ValueError("base_com.offset_ranges_m must specify finite ordered pairs for x, y and z")
    return [ranges[axis] for axis in ("x", "y", "z")]


def base_mass_scale_range(config):
    domain = config.get("domain_randomization", {})
    if not isinstance(domain, dict):
        raise ValueError("domain_randomization must be a mapping")
    section = domain.get("base_mass", {})
    if not isinstance(section, dict) or not isinstance(section.get("enabled", False), bool):
        raise ValueError("base_mass must be a mapping with a boolean enabled flag")
    if not section.get("enabled", False):
        return None
    bounds = section.get("scale_range")
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or not all(type(x) in (int, float) and 0 < x <= 1e30 for x in bounds)
            or bounds[0] > bounds[1]):
        raise ValueError("base_mass.scale_range must be a finite positive ordered pair")
    return bounds


def foot_friction_switch_seconds(config):
    friction = config.get("domain_randomization", {}).get("foot_ground_friction", {})
    interval = friction.get("switch_interval_seconds")
    if not friction.get("enabled") or interval is None:
        return None
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or not all(type(value) in (int, float) and math.isfinite(value) for value in interval)
            or not 0 < interval[0] <= interval[1]):
        raise ValueError("switch_interval_seconds must be a positive ordered pair")
    if any(lo != hi for foot in friction["base_ranges"].values() for lo, hi in foot.values()):
        raise ValueError("Timed friction switching requires fixed per-foot base_ranges")
    return interval


def foot_friction_ranges(config):
    domain = config.get("domain_randomization", {})
    if not isinstance(domain, dict):
        raise ValueError("domain_randomization must be a mapping")
    section = domain.get("foot_ground_friction", {})
    if not isinstance(section, dict) or not isinstance(section.get("enabled", False), bool):
        raise ValueError("foot_ground_friction must be a mapping with a boolean enabled flag")
    if not section.get("enabled", False):
        return None

    def bounds(pair, label):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) and x >= 0 for x in pair)
                or pair[0] > pair[1]):
            raise ValueError(f"{label} must be a finite nonnegative [min, max] pair")
        return pair

    feet = section.get("base_ranges", {})
    if (not isinstance(feet, dict) or set(feet) != set(FOOT_NAMES)
            or any(not isinstance(feet[name], dict) or set(feet[name]) != set(FRICTION_TERMS) for name in FOOT_NAMES)):
        raise ValueError("base_ranges must specify sliding, torsional and rolling for FR, FL, RR and RL")
    base = [[bounds(feet[name][term], f"{name}.{term}") for term in FRICTION_TERMS] for name in FOOT_NAMES]
    ratio = bounds(section.get("env_multiplier_range"), "env_multiplier_range")
    if min(foot[0][0] for foot in base) * ratio[0] < 0.01:
        raise ValueError("Effective sliding friction must be at least 0.01 (Genesis contact floor)")
    largest_base = max(pair[1] for foot in base for pair in foot)
    if max(largest_base, ratio[1], largest_base * ratio[1]) > 3.4028234663852886e38:
        raise ValueError("Friction bounds and their effective products must fit in float32")
    foot_friction_switch_seconds(config)
    return base, ratio


def read_yaml(path):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def load_scene(path: str | Path, *, entity_overrides: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Load backend-ready geometry, optionally overriding named terrain parameters."""
    path = Path(path).resolve()
    doc = read_yaml(path)
    if doc.get("schema") != "rl-go2.scene/v1":
        raise ValueError("Expected schema: rl-go2.scene/v1 (Unisis-style subset)")
    return validate_scene(doc["scene"], path, entity_overrides=entity_overrides)


def validate_scene(scene: dict, path: Path, *, entity_overrides=None) -> dict:
    """Validate resolved or authored scene geometry and resolve asset paths."""
    unknown = scene.keys() - {"sim_options", "entities", "sensors", "terrain_patches", "height_maps"}
    if unknown:
        raise ValueError(f"Unsupported scene fields: {sorted(unknown)}")
    if scene.get("sensors"):
        raise NotImplementedError("Sensor loading is reserved for future tasks")
    sim = scene["sim_options"]
    if sim.keys() - {"dt", "gravity"} or not math.isfinite(sim["dt"]) or sim["dt"] <= 0:
        raise ValueError("sim_options supports positive dt and gravity only")
    if len(sim["gravity"]) != 3 or not all(math.isfinite(x) for x in sim["gravity"]):
        raise ValueError("gravity must be three finite values")
    seen = set()
    scene["entities"], height_maps = resolve_terrain_entities(scene["entities"], entity_overrides or {})
    if height_maps:
        scene["height_maps"] = height_maps
    for entity in scene["entities"]:
        eid = entity["id"]
        if eid in seen:
            raise ValueError(f"Duplicate entity id: {eid}")
        seen.add(eid)
        kind = entity["type"]
        if kind not in {"plane", "box", "mjcf"}:
            raise NotImplementedError(f"Scene entity type not implemented: {kind}")
        allowed = {"id", "type", "position", "rotation"}
        allowed |= {"robot", "file"} if kind == "mjcf" else {"fixed", "friction"}
        if kind == "box":
            allowed.add("size")
        if entity.keys() - allowed:
            raise ValueError(f"Unsupported fields on entity {eid}: {entity.keys() - allowed}")
        if kind == "mjcf" and not entity.get("robot"):
            raise NotImplementedError("MJCF currently supports the robot entity only")
        if kind != "mjcf" and not entity.get("fixed", True):
            raise NotImplementedError("Dynamic props are not implemented yet")
        entity.setdefault("position", [0, 0, 0])
        entity.setdefault("rotation", [1, 0, 0, 0])
        if len(entity["position"]) != 3 or len(entity["rotation"]) != 4:
            raise ValueError(f"Invalid pose for {eid}")
        if not all(math.isfinite(x) for x in entity["position"] + entity["rotation"]):
            raise ValueError(f"Pose must be finite: {eid}")
        if not math.isclose(sum(x*x for x in entity["rotation"]), 1.0, abs_tol=1e-5):
            raise ValueError(f"rotation must be a unit wxyz quaternion: {eid}")
        if kind == "box" and (len(entity["size"]) != 3 or any(
            not math.isfinite(size) or size <= 0 for size in entity["size"]
        )):
            raise ValueError(f"Box size must contain three positive full lengths: {eid}")
        if "friction" in entity and (not math.isfinite(entity["friction"]) or entity["friction"] < 0):
            raise ValueError(f"friction must be finite and nonnegative: {eid}")
        if "file" in entity:
            entity["file"] = str((path.parent / entity["file"]).resolve())
    return scene


def validate_scene_robot(scene: dict, robot: dict) -> None:
    """Require a single robot matching the controller configuration."""
    entities = [entity for entity in scene["entities"] if entity.get("robot")]
    if len(entities) != 1 or entities[0]["id"] != robot["entity_id"]:
        raise ValueError("Scene must contain exactly one robot matching robot.entity_id")


def resolve_run_asset(file: str) -> str:
    """Resolve a saved asset path, falling back to this checkout for copied runs."""
    original = Path(file)
    if original.is_file():
        return file
    parts = original.parts
    if 'assets' not in parts:
        raise FileNotFoundError(f'Cannot relocate scene asset: {original}')
    local = REPO_ROOT.joinpath(*parts[parts.index('assets'):])
    if not local.is_file():
        raise FileNotFoundError(f'Scene asset missing: {original} (local candidate: {local})')
    return str(local)


def joint_position_bounds(
    robot: dict,
) -> tuple[tuple[float, float], ...] | None:
    """Return physical joint bounds in radians, ordered by robot.joint_names."""
    if "joint_position_limits" not in robot:
        return None
    ranges = robot["joint_position_limits"]
    names = robot["joint_names"]
    margin = robot.get("joint_limit_margin", 0.0)
    if not isinstance(ranges, dict) or set(ranges) != set(names):
        raise ValueError("Joint limits must cover joint_names")
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
        raise ValueError("Joint limit margin must be finite and nonnegative")

    defaults = robot["default_joint_positions"]
    resets = robot.get("reset_joint_positions", defaults)
    bounds = []
    for name, default, reset in zip(names, defaults, resets, strict=True):
        pair = ranges[name]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(type(angle) in (int, float) and math.isfinite(angle) for angle in pair)):
            raise ValueError(f"Joint limits for {name} must be two finite angles in radians")
        lower, upper = pair
        if lower + margin >= upper - margin:
            raise ValueError(f"Joint limits for {name} must be ordered with room for the margin")
        if not lower + margin <= default <= upper - margin:
            raise ValueError(f"Default joint positions must lie inside the joint limit margin: {name}")
        if not lower <= reset <= upper:
            raise ValueError(f"Reset joint positions must lie inside the physical joint limits: {name}")
        bounds.append((lower, upper))
    return tuple(bounds)


def _apply_robot_overrides(robot: dict, overrides: object) -> None:
    """Apply per-joint experiment overrides to the resolved robot configuration."""
    if not isinstance(overrides, dict) or overrides.keys() - {"joint_position_limits"}:
        raise ValueError("robot_overrides supports only joint_position_limits")
    limits = overrides.get("joint_position_limits", {})
    if not isinstance(limits, dict) or limits.keys() - set(robot["joint_names"]):
        raise ValueError("robot_overrides.joint_position_limits must use known joint names")
    if limits:
        robot["joint_position_limits"] = {**robot.get("joint_position_limits", {}), **limits}


def load_experiment(path):
    path = Path(path).resolve()
    config = read_yaml(path)
    robot_path = path.parent / config["robot"]
    for key in ("robot", "task"):
        config[key] = read_yaml(path.parent / config[key])
    robot = config["robot"]
    _apply_robot_overrides(robot, config.pop("robot_overrides", {}))
    scene = config['scene']
    if isinstance(scene, dict):
        from rl_go2.training_scene import build_training_scene

        scene = build_training_scene(scene, config['robot'], robot_path, config['seed'])
        config['scene'] = validate_scene(scene, path)
    else:
        config['scene'] = load_scene(path.parent / scene)
    validate_scene_robot(config["scene"], robot)
    names = robot["joint_names"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Robot joint_names must be nonempty and unique")
    for key in ("default_joint_positions", "torque_limits"):
        if len(robot[key]) != len(names):
            raise ValueError(f"{key} length must match joint_names")
    if "reset_joint_positions" in robot:
        positions = robot["reset_joint_positions"]
        if (not isinstance(positions, list) or len(positions) != len(names)
                or not all(type(angle) in (int, float) and math.isfinite(angle) for angle in positions)):
            raise ValueError("reset_joint_positions must provide one finite angle per joint in joint_names order")
    joint_position_bounds(robot)
    if min(robot["torque_limits"]) <= 0 or robot["kp"] < 0 or robot["kd"] < 0:
        raise ValueError("Invalid PD gains or torque limits")
    decimation = robot["control_decimation"]
    if not isinstance(decimation, int) or decimation < 1:
        raise ValueError("control_decimation must be a positive integer")
    if robot["action_clip"] <= 0 or robot["action_scale"] <= 0:
        raise ValueError("action_clip and action_scale must be positive")
    if 'courses' in config:
        _load_courses(config, path, robot_path)
    else:
        _validate_task_and_randomization(config)
    return config


def _validate_task_and_randomization(config: dict) -> None:
    task = config["task"]
    validate_task_name(task["name"])
    if not math.isfinite(task["episode_seconds"]) or task["episode_seconds"] <= 0:
        raise ValueError("episode_seconds must be finite and positive")
    foot_friction_ranges(config)
    base_mass_scale_range(config)
    base_com_offset_ranges(config)
    if 'action_delay_steps' in config.get('domain_randomization', {}):
        from rl_go2.step_delay import action_delay_from_config

        action_delay_from_config(config)
    if 'observation_noise' in config.get('domain_randomization', {}):
        from rl_go2.observation_noise import ObservationNoise

        ObservationNoise.from_config(config)
    from rl_go2.point_push import PointPushConfig

    push = PointPushConfig.from_config(config)
    if push is not None and (config['trainer']['name'] != 'supermimic'
                             or not config['trainer'].get('condition_on_link_wrenches')
                             or task['name'] != 'flat_walk'):
        raise ValueError('point_push requires SuperMimic flat_walk with condition_on_link_wrenches')


def _load_courses(config: dict, path: Path, robot_path: Path) -> None:
    """Resolve course files into cumulative, self-contained stage settings."""
    from rl_go2.tasks import get_task_type
    from rl_go2.training_scene import build_training_scene

    if (config['trainer']['name'] != 'supermimic' or config['task']['name'] != 'flat_walk'
            or config['curriculum']['name'] != 'none'):
        raise ValueError('courses require SuperMimic flat_walk and curriculum.name: none')
    if 'iterations' in config['trainer'] or 'domain_randomization' in config:
        raise ValueError('Set iteration budgets in courses entries and domain randomization in course files')
    entries = config['courses']
    if not isinstance(entries, list) or not entries:
        raise ValueError('courses must be a nonempty list of config and iterations entries')
    task, scene, domain = config['task'], config['scene'], {}
    task_type = get_task_type(task['name'])
    stages = []
    total = 0
    for entry in entries:
        if (not isinstance(entry, dict) or entry.keys() != {'config', 'iterations'}
                or not isinstance(entry['config'], str)
                or type(entry['iterations']) is not int or entry['iterations'] < 1):
            raise ValueError('Each course needs a config path and a positive integer iteration budget')
        course_path = path.parent / entry['config']
        course = read_yaml(course_path)
        unknown = course.keys() - {'command_ranges', 'domain_randomization', 'terrains'}
        if not course or unknown:
            raise ValueError(f'Invalid course settings in {course_path}: {sorted(unknown)}')
        if 'command_ranges' in course:
            task = {**task, 'command_ranges': course['command_ranges']}
        changes = course.get('domain_randomization', {})
        if not isinstance(changes, dict) or changes.keys() - {
            'foot_ground_friction', 'base_mass', 'base_com', 'observation_noise',
            'action_delay_steps', 'point_push',
        }:
            raise ValueError(f'Invalid domain_randomization in {course_path}')
        domain = {**domain, **changes}
        if 'terrains' in course:
            spec = {'sim_options': scene['sim_options'], 'terrains': course['terrains']}
            scene = build_training_scene(spec, config['robot'], robot_path, config['seed'])
            scene = validate_scene(scene, path)
            validate_scene_robot(scene, config['robot'])
        total += entry['iterations']
        stage = {'name': course_path.stem, 'end_iteration': total, 'iterations': entry['iterations'],
                 'settings': {'task': task, 'scene': scene, 'domain_randomization': domain}}
        task_type.validate_config(task, config['robot'])
        _validate_task_and_randomization({**config, **stage['settings']})
        stages.append(stage)
    config.update(stages[0]['settings'])
    config['courses'] = stages
    config['trainer']['iterations'] = total


def config_at_iteration(config: dict, iteration: int | None) -> dict:
    """Return the effective settings for a saved checkpoint's last training iteration."""
    if iteration is not None:
        for stage in config.get('courses', ()):
            if iteration <= stage['end_iteration']:
                return {**config, **stage['settings']}
    return config
