from importlib.metadata import version
import json
from pathlib import Path
import hashlib

from rl_go2.tasks import get_task_type


def save_run_spec(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    asset = next(e["file"] for e in config["scene"]["entities"] if e.get("robot"))
    meshes = sorted((Path(asset).parent / "assets").glob("*"))
    metadata = {
        "schema": "rl-go2.policy/v1", "quaternion_order": "wxyz", "world_up": "z", "units": "SI",
        "task": config["task"]["name"],
        "observation_terms": get_task_type(config["task"]["name"]).observation_terms, "control": config["robot"],
        "control_dt": config["scene"]["sim_options"]["dt"] * config["robot"]["control_decimation"],
        "versions": {name: version(name) for name in ("torch", "genesis-world", "mujoco", "rsl-rl-lib")},
        "asset_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(asset), *meshes]},
    }
    (output / "policy_spec.json").write_text(json.dumps(metadata, indent=2) + "\n")
