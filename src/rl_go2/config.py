"""Small YAML loader; all authored asset paths are relative to their YAML."""

from pathlib import Path
import math

import yaml

from rl_go2.tasks import validate_task_name


def read_yaml(path):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def load_scene(path):
    path = Path(path).resolve()
    doc = read_yaml(path)
    if doc.get("schema") != "rl-go2.scene/v1":
        raise ValueError("Expected schema: rl-go2.scene/v1 (Unisis-style subset)")
    scene = doc["scene"]
    unknown = scene.keys() - {"sim_options", "entities", "sensors"}
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
        if kind == "box" and (len(entity["size"]) != 3 or min(entity["size"]) <= 0):
            raise ValueError(f"Box size must contain three positive full lengths: {eid}")
        if "friction" in entity and (not math.isfinite(entity["friction"]) or entity["friction"] < 0):
            raise ValueError(f"friction must be finite and nonnegative: {eid}")
        if "file" in entity:
            entity["file"] = str((path.parent / entity["file"]).resolve())
    return scene


def load_experiment(path):
    path = Path(path).resolve()
    config = read_yaml(path)
    config["scene"] = load_scene(path.parent / config["scene"])
    for key in ("robot", "task"):
        config[key] = read_yaml(path.parent / config[key])
    robot = config["robot"]
    entities = [e for e in config["scene"]["entities"] if e.get("robot")]
    if len(entities) != 1 or entities[0]["id"] != robot["entity_id"]:
        raise ValueError("Scene must contain exactly one robot matching robot.entity_id")
    names = robot["joint_names"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Robot joint_names must be nonempty and unique")
    for key in ("default_joint_positions", "torque_limits"):
        if len(robot[key]) != len(names):
            raise ValueError(f"{key} length must match joint_names")
    if min(robot["torque_limits"]) <= 0 or robot["kp"] < 0 or robot["kd"] < 0:
        raise ValueError("Invalid PD gains or torque limits")
    decimation = robot["control_decimation"]
    if not isinstance(decimation, int) or decimation < 1:
        raise ValueError("control_decimation must be a positive integer")
    if robot["action_clip"] <= 0 or robot["action_scale"] <= 0:
        raise ValueError("action_clip and action_scale must be positive")
    task = config["task"]
    validate_task_name(task["name"])
    if not math.isfinite(task["episode_seconds"]) or task["episode_seconds"] <= 0:
        raise ValueError("episode_seconds must be finite and positive")
    return config
