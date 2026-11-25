"""Transfer SuperMimic learning state between courses with different inputs."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from rl_go2.config import MODEL_PARAMETER_KEYS

if TYPE_CHECKING:
    from rl_go2.trainers.supermimic import Trainer


InputColumns = tuple[tuple[str, int], ...]
InputLayouts = dict[str, InputColumns]


def input_layouts(trainer: "Trainer") -> InputLayouts:
    """Describe the input columns in the same order as the model and critic."""
    context = trainer.env.model_context()
    physical = tuple((key, context[key][0].numel()) for key in MODEL_PARAMETER_KEYS if key in context)
    state_action_width = trainer.model.input_norm.mean.numel() - sum(width for _, width in physical)
    shared = (('state_action', state_action_width), *physical)
    critic_context = tuple((key, context[key][0].numel()) for key in sorted(context)
                           if key not in MODEL_PARAMETER_KEYS)
    model = shared
    for name, normalizer in (('wrenches', trainer.model.wrench_norm), ('terrain', trainer.model.terrain_norm)):
        if normalizer is not None:
            model += ((name, normalizer.mean.numel()),)
    return {'world_model': model, 'critic': shared + critic_context}


def _column_indices(columns: InputColumns) -> dict[tuple[str, int], int]:
    return {column: index for index, column in enumerate(
        (name, offset) for name, width in columns for offset in range(width)
    )}


def _transfer_columns(source: torch.Tensor, target: torch.Tensor,
                      old: InputColumns, new: InputColumns) -> torch.Tensor:
    previous = _column_indices(old)
    current = _column_indices(new)
    common = [key for key in previous if key in current]
    target[..., [current[key] for key in common]] = source[..., [previous[key] for key in common]]
    return target


def _transfer_network(module: torch.nn.Module, optimizer: torch.optim.Optimizer, state: dict,
                      optimizer_state: dict, old: InputColumns, new: InputColumns,
                      normalizer: str) -> None:
    if old != new:
        target = module.state_dict()
        weight = 'net.0.weight'
        state[weight] = _transfer_columns(state[weight], torch.zeros_like(target[weight], device='cpu'), old, new)
        # Wrenches and terrain have their own normalizers in the dynamics model.
        old_norm = tuple(group for group in old if group[0] not in ('wrenches', 'terrain'))
        new_norm = tuple(group for group in new if group[0] not in ('wrenches', 'terrain'))
        for suffix in ('mean', 'var'):
            key = f'{normalizer}.{suffix}'
            state[key] = _transfer_columns(state[key], target[key].to(device='cpu', copy=True), old_norm, new_norm)
        for key in target.keys() - state.keys():
            state[key] = target[key]
        for key in state.keys() - target.keys():
            del state[key]
        # Adam's moments follow the same input columns as the weight they track.
        position = list(dict(module.named_parameters())).index(weight)
        parameter_id = optimizer_state['param_groups'][0]['params'][position]
        moments = optimizer_state['state'].get(parameter_id, {})
        for name, value in moments.items():
            if value.ndim == state[weight].ndim:
                moments[name] = _transfer_columns(value, torch.zeros_like(state[weight]), old, new)
    module.load_state_dict(state)
    optimizer.load_state_dict(optimizer_state)


def continue_from_course(trainer: "Trainer", checkpoint: Path, previous: InputLayouts) -> None:
    """Restore learned state, keeping new model or critic input columns initially neutral."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    trainer.actor.load_state_dict(saved['actor'])
    trainer.actor_optimizer.load_state_dict(saved['actor_optimizer'])
    current = input_layouts(trainer)
    _transfer_network(trainer.model, trainer.model_optimizer, saved['world_model'], saved['model_optimizer'],
                      previous['world_model'], current['world_model'], 'input_norm')
    _transfer_network(trainer.critic, trainer.critic_optimizer, saved['critic'], saved['critic_optimizer'],
                      previous['critic'], current['critic'], 'norm')
