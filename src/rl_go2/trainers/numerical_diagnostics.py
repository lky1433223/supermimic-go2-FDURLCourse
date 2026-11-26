"""Capture eager autograd failures without allowing invalid optimizer updates."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn


def gradient_report(modules: dict[str, nn.Module]) -> dict:
    """Summarize parameter gradients, using float64 for finite-gradient norms."""
    report = {}
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = gradient.detach()
            finite = torch.isfinite(values)
            report[f"{module_name}.{name}"] = {
                "nonfinite_count": int((~finite).sum().item()),
                "finite_norm_float64": values[finite].double().norm().item(),
            }
    return report


@contextmanager
def diagnose_update(
    output: Path, iteration: int, modules: dict[str, nn.Module],
    batches: dict[str, dict[str, torch.Tensor]],
) -> Iterator[None]:
    """Trace forward operations and preserve failed-update evidence before reraising."""
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            yield
    except RuntimeError as error:
        destination = output / "diagnostics" / f"iteration-{iteration}"
        destination.mkdir(parents=True, exist_ok=False)
        report = {"iteration": iteration, "error": str(error), "gradients": gradient_report(modules)}
        (destination / "failure.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # A failure snapshot is evidence, not a resumable training checkpoint.
        snapshot = {
            "iteration": iteration,
            "modules": {name: module.state_dict() for name, module in modules.items()},
            "gradients": {
                f"{module_name}.{name}": parameter.grad
                for module_name, module in modules.items()
                for name, parameter in module.named_parameters() if parameter.grad is not None
            },
            "batches": batches,
            "rng_cpu_before_update": rng_cpu,
            "rng_cuda_before_update": rng_cuda,
        }
        torch.save(snapshot, destination / "failure.pt")
        print(f"Numerical diagnostic evidence: {destination}", flush=True)
        raise
