"""Recompute a recorded SuperMimic update without sampling or optimizer steps."""

from contextlib import closing, nullcontext
from dataclasses import replace
import json
import math
import os
from pathlib import Path

import torch

from rl_go2.config import config_at_iteration, resolve_run_asset
from rl_go2.env import QuadrupedEnv
from rl_go2.trainers.numerical_diagnostics import gradient_report, restore_rng
from rl_go2.trainers.supermimic import TrainConfig, Trainer, cuda_matmul_tf32, frozen


def largest_model_error(trainer: Trainer, batch: dict[str, torch.Tensor]) -> dict:
    """Locate the largest valid normalized error in the recorded prediction window."""
    trace = []
    with torch.no_grad():
        trainer.model.loss(batch, error_trace=trace)
    raw, normalized = (torch.stack(parts, dim=1) for parts in zip(*trace, strict=True))
    valid = batch['valid'].bool().unsqueeze(-1)
    finite = normalized.isfinite()
    eligible = valid & finite
    count = int((valid & ~finite).sum().item())
    if not eligible.any():
        return {'nonfinite_errors': count}
    index = normalized.abs().masked_fill(~eligible, -torch.inf).argmax().item()
    sample_step, channel = divmod(index, normalized.shape[-1])
    sample, step = divmod(sample_step, normalized.shape[1])
    joints = trainer.env.config['robot']['joint_names']
    channels = (
        *(f'displacement.{axis}' for axis in 'xyz'),
        *(f'quaternion.{axis}' for axis in 'wxyz'),
        *(f'linear_velocity.{axis}' for axis in 'xyz'),
        *(f'angular_velocity.{axis}' for axis in 'xyz'),
        *(f'joint_position.{joint}' for joint in joints),
        *(f'joint_velocity.{joint}' for joint in joints),
        *(f'torque.{joint}' for joint in joints),
    )
    return {
        'nonfinite_errors': count,
        'batch_index': sample, 'prediction_step': step + 1,
        'output_channel': channel, 'output_term': channels[channel],
        'raw_error': raw[sample, step, channel].item(),
        'normalized_error': normalized[sample, step, channel].item(),
    }


def replay_update(trainer: Trainer, snapshot: dict, batch: dict[str, torch.Tensor],
                  *, anomaly: bool = False) -> dict:
    """Restore pre-forward RNG and compute the loss and unclipped gradients."""
    kind = snapshot['kind']
    if kind == 'model':
        module, loss_fn = trainer.model, lambda: trainer.model_loss(batch)
    elif kind == 'actor':
        module, loss_fn = trainer.actor, lambda: trainer.actor_loss(batch)[0]
    elif kind == 'critic':
        module, loss_fn = trainer.critic, lambda: trainer.critic_loss(**batch)
    else:
        raise ValueError(f'Unknown diagnostic update: {kind}')
    module.zero_grad(set_to_none=True)
    restore_rng(snapshot['rng'], trainer.env.device)
    loss_value = None
    failure = None
    # Match the training actor path, where weights of the differentiable simulator are fixed.
    frozen_modules = frozen(trainer.model, trainer.critic) if kind == 'actor' else nullcontext()
    try:
        with frozen_modules, (torch.autograd.detect_anomaly(check_nan=True) if anomaly else nullcontext()):
            loss = loss_fn()
            value = loss.detach().item()
            loss_value = value if math.isfinite(value) else None
            loss.backward()
    except RuntimeError as error:
        failure = str(error)
    gradients = gradient_report({kind: module})
    return {
        'loss': loss_value, 'error': failure,
        'nonfinite_gradient_elements': sum(row['nonfinite_count'] for row in gradients.values()),
        'finite_gradient_norm_float64': math.hypot(*(row['finite_norm_float64'] for row in gradients.values())),
        'gradients': gradients,
    }


def replay_diagnostic(path: Path, *, device: str | None = None) -> None:
    """Compare eager and CUDA compile from one snapshot and its saved run config."""
    path = path.resolve()
    snapshot = torch.load(path, map_location='cpu', weights_only=True)
    config = json.loads((path.parents[2] / 'config.json').read_text())
    config = config_at_iteration(config, snapshot['iteration'])
    if device is not None:
        config['device'] = device
    if 'inductor_compile_threads' in config:
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = str(config['inductor_compile_threads'])
    for entity in config['scene']['entities']:
        if 'file' in entity:
            entity['file'] = resolve_run_asset(entity['file'])
    cfg = TrainConfig(**config['trainer'])
    if cfg.driver_policy is not None:
        cfg = replace(cfg, driver_policy=resolve_run_asset(cfg.driver_policy))
    # The recorded batch supplies the states and commands; a small env constructs the task and terrain.
    config['num_envs'] = 2 if cfg.driver_policy else 1
    results = {
        'iteration': snapshot['iteration'], 'kind': snapshot['kind'], 'trigger': snapshot['trigger'],
        'recorded_loss': snapshot['loss'], 'recorded_cuda_graph': snapshot['cuda_graph'],
    }
    with closing(QuadrupedEnv(config, record_transitions=True)) as env, cuda_matmul_tf32(
        cfg.tf32 and env.device.type == 'cuda'
    ):
        results['rng_device_type_matches'] = snapshot['rng']['device_type'] == env.device.type
        modes = ('eager', 'compile') if env.device.type == 'cuda' else ('eager',)
        batch = {key: value.to(env.device) for key, value in snapshot['batch'].items()}
        for mode in modes:
            trainer = Trainer(env, replace(cfg, compile_training=mode == 'compile', cuda_graph=False))
            modules = {'actor': trainer.actor, 'world_model': trainer.model, 'critic': trainer.critic}
            for name, state in snapshot['modules'].items():
                modules[name].load_state_dict(state)
            result = replay_update(trainer, snapshot, batch)
            if mode == 'eager' and (result['error'] or result['nonfinite_gradient_elements']):
                result['anomaly'] = replay_update(trainer, snapshot, batch, anomaly=True)['error']
            if mode == 'eager' and snapshot['kind'] == 'model':
                result['largest_model_error'] = largest_model_error(trainer, batch)
            results[mode] = result
            del modules, trainer
    print(json.dumps(results, indent=2, allow_nan=False))
