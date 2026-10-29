from typing import Protocol

import torch

from rl_go2.state import RobotState


class PhysicsBackend(Protocol):
    num_envs: int
    device: torch.device

    def read_state(self) -> RobotState: ...
    def reset(self, env_ids: torch.Tensor) -> None: ...
    def step(self, torque: torch.Tensor) -> None: ...
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
