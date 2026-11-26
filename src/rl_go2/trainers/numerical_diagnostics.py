"""Save the exact batches and parameters involved in abnormal SuperMimic updates."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn


def gradient_report(modules: dict[str, nn.Module]) -> dict:
    """Summarize existing gradients on CPU, using float64 for finite values."""
    report = {}
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            values = parameter.grad.detach().cpu()
            finite = torch.isfinite(values)
            report[f"{module_name}.{name}"] = {
                "nonfinite_count": int((~finite).sum().item()),
                "finite_norm_float64": values[finite].double().norm().item(),
            }
    return report


def snapshot_rng(device: torch.device) -> dict:
    """Read RNG states after sampling the batch and before running its forward pass."""
    accelerator = None
    if device.type == 'cuda':
        accelerator = torch.cuda.get_rng_state(device)
    elif device.type == 'mps':
        accelerator = torch.mps.get_rng_state()
    return {'device_type': device.type, 'cpu': torch.get_rng_state(), 'accelerator': accelerator}


def restore_rng(state: dict, device: torch.device) -> None:
    """Restore recorded generators on a device of the same type."""
    torch.set_rng_state(state['cpu'])
    if state['device_type'] != device.type:
        return
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['accelerator'], device)
    elif device.type == 'mps':
        torch.mps.set_rng_state(state['accelerator'])


class NumericalDiagnostics:
    """Collect at most one finite model spike per run and save failed updates."""

    def __init__(self, output: Path, device: torch.device, threshold: float, warmup_iterations: int):
        self.output = output / 'diagnostics'
        self.device = device
        self.threshold = threshold
        self.warmup_iterations = warmup_iterations
        self.iteration = 0
        self.model_spike_saved = False

    def begin(self, kind: str, batch: dict[str, torch.Tensor], modules: dict[str, nn.Module],
              *, cuda_graph: bool) -> 'UpdateSnapshot':
        """Associate the current minibatch with its RNG state."""
        return UpdateSnapshot(self, kind, batch, modules, snapshot_rng(self.device), cuda_graph)


@dataclass
class UpdateSnapshot:
    """Write one update's inputs, weights, and available unclipped gradients."""

    diagnostics: NumericalDiagnostics
    kind: str
    batch: dict[str, torch.Tensor]
    modules: dict[str, nn.Module]
    rng: dict
    cuda_graph: bool

    def check_loss(self, loss: torch.Tensor) -> None:
        """Capture the first large finite model minibatch loss after warmup."""
        run = self.diagnostics
        if self.kind != 'model' or run.model_spike_saved or run.iteration <= run.warmup_iterations:
            return
        value = loss.detach().item()
        if value > run.threshold:
            run.model_spike_saved = self._save('loss-spike', 'forward', value)

    def failed(self, phase: str, error: RuntimeError, loss: torch.Tensor) -> None:
        """Save available gradients before the original exception is raised again."""
        self._save('failure', phase, loss, str(error), include_gradients=phase != 'loss')

    def _save(self, trigger: str, phase: str, loss: float | torch.Tensor, error: str | None = None,
              *, include_gradients: bool = False) -> bool:
        run = self.diagnostics
        destination = run.output / f'iteration-{run.iteration}-{self.kind}-{trigger}'
        try:
            value = loss.detach().item() if isinstance(loss, torch.Tensor) else loss
            updated = 'world_model' if self.kind == 'model' else self.kind
            destination.mkdir(parents=True, exist_ok=True)
            report = {
                'iteration': run.iteration, 'kind': self.kind, 'trigger': trigger, 'phase': phase,
                'loss': value if math.isfinite(value) else None,
                'error': error, 'cuda_graph': self.cuda_graph,
                'gradients': gradient_report({updated: self.modules[updated]}) if include_gradients else {},
            }
            snapshot = {
                **report,
                'modules': {name: module.state_dict() for name, module in self.modules.items()},
                'batch': self.batch,
                'rng': self.rng,
            }
            temporary = destination / 'snapshot.pt.tmp'
            torch.save(snapshot, temporary)
            temporary.replace(destination / 'snapshot.pt')
            (destination / 'snapshot.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        except (OSError, RuntimeError, ValueError) as save_error:
            print(f'Numerical diagnostic snapshot failed: {save_error}', flush=True)
            return False
        print(f'Numerical diagnostic snapshot: {destination}', flush=True)
        return True
