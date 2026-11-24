"""Load SuperMimic model weights for a new training run."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from rl_go2.config import base_com_offset_ranges, base_mass_scale_range, foot_friction_ranges

if TYPE_CHECKING:
    from rl_go2.trainers.supermimic import Trainer


def initialize_weights(trainer: "Trainer", checkpoint: Path) -> None:
    """Initialize networks and normalizers, adding zero-weight wrench inputs if needed."""
    _validate_source_config(checkpoint, trainer.env.config)
    saved = torch.load(checkpoint, map_location=trainer.env.device, weights_only=True)
    if saved.get("schema") != "rl-go2.supermimic/v2":
        raise ValueError("Expected a SuperMimic v2 checkpoint")

    model_state = dict(saved["world_model"])
    target_state = trainer.model.state_dict()
    if trainer.model.wrench_norm is not None:
        _initialize_wrench_inputs(model_state, target_state)
    trainer.model.load_state_dict(model_state)
    trainer.actor.load_state_dict(saved["actor"])
    trainer.critic.load_state_dict(saved["critic"])
    print(f"Initialized SuperMimic weights from {checkpoint}; optimizers and replay start fresh.", flush=True)


def _validate_source_config(checkpoint: Path, current: dict) -> None:
    source = checkpoint.parent / "config.json"
    if not source.is_file():
        source = checkpoint.parent / "remote-config.json"
    original = json.loads(source.read_text())
    for key in ("base_link", "joint_names", "default_joint_positions", "kp", "kd", "torque_limits",
                "action_scale", "action_clip", "control_decimation", "joint_position_limits",
                "joint_limit_margin"):
        if original["robot"].get(key) != current["robot"].get(key):
            raise ValueError(f"Checkpoint robot setting differs: {key}")
    if original["scene"]["sim_options"]["dt"] != current["scene"]["sim_options"]["dt"]:
        raise ValueError("Checkpoint physics timestep differs")
    for key in ("name", "observation_scales"):
        if original["task"].get(key) != current["task"].get(key):
            raise ValueError(f"Checkpoint task setting differs: {key}")
    for reader in (foot_friction_ranges, base_mass_scale_range, base_com_offset_ranges):
        if (reader(original) is None) != (reader(current) is None):
            raise ValueError("Checkpoint physical feature layout differs")


def _initialize_wrench_inputs(
    model_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor],
) -> None:
    wrench_width = target_state["wrench_norm.mean"].numel()
    base_width = target_state["input_norm.mean"].numel()
    weight = model_state["net.0.weight"]
    if weight.shape[1] == base_width:
        model_state["net.0.weight"] = F.pad(weight, (0, wrench_width))
        for name in ("mean", "var", "count"):
            key = f"wrench_norm.{name}"
            model_state[key] = target_state[key]
    elif model_state["input_norm.mean"].numel() == base_width + wrench_width:
        # A checkpoint from the initial wrench implementation used one normalizer.
        for name in ("mean", "var"):
            base_key = f"input_norm.{name}"
            model_state[base_key], model_state[f"wrench_norm.{name}"] = model_state[base_key].split(
                (base_width, wrench_width)
            )
        model_state["wrench_norm.count"] = model_state["input_norm.count"].clone()
