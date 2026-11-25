"""Load SuperMimic model weights for a new training run."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from rl_go2.config import base_com_offset_ranges, base_mass_scale_range, config_at_iteration, foot_friction_ranges

if TYPE_CHECKING:
    from rl_go2.trainers.supermimic import Trainer


def initialize_weights(trainer: "Trainer", checkpoint: Path) -> None:
    """Initialize networks and normalizers, adding zero-weight physics inputs if needed."""
    saved = torch.load(checkpoint, map_location=trainer.env.device, weights_only=True)
    if saved.get("schema") != "rl-go2.supermimic/v2":
        raise ValueError("Expected a SuperMimic v2 checkpoint")
    _validate_source_config(checkpoint, trainer.env.config, saved.get('iteration'))

    model_state = dict(saved["world_model"])
    target_state = trainer.model.state_dict()
    if trainer.model.wrench_norm is not None:
        _initialize_wrench_inputs(model_state, target_state)
    if trainer.model.terrain_norm is not None and 'terrain_norm.mean' not in model_state:
        width = target_state['terrain_norm.mean'].numel()
        model_state['net.0.weight'] = F.pad(model_state['net.0.weight'], (0, width))
        for name in ('mean', 'var', 'count'):
            key = f'terrain_norm.{name}'
            model_state[key] = target_state[key]
    trainer.model.load_state_dict(model_state)
    trainer.actor.load_state_dict(saved["actor"])
    trainer.critic.load_state_dict(saved["critic"])
    print(f"Initialized SuperMimic weights from {checkpoint}; optimizers and replay start fresh.", flush=True)


def _validate_source_config(checkpoint: Path, current: dict, iteration: int | None) -> None:
    source = checkpoint.parent / "config.json"
    if not source.is_file():
        source = checkpoint.parent / "remote-config.json"
    original = config_at_iteration(json.loads(source.read_text()), iteration)
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
    terrain = model_state.get('terrain_norm.mean')
    terrain_width = 0 if terrain is None else terrain.numel()
    if weight.shape[1] == base_width + terrain_width:
        # Terrain channels follow any link-wrench channels in the first layer.
        new_wrench_columns = weight.new_zeros(weight.shape[0], wrench_width)
        model_state["net.0.weight"] = torch.cat((weight[:, :base_width], new_wrench_columns, weight[:, base_width:]), 1)
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
