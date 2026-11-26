"""Verify diagnostic evidence for backward failures and gradient norm overflow."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from rl_go2.trainers.numerical_diagnostics import diagnose_update, gradient_report
from rl_go2.trainers.supermimic import TrainConfig


class NumericalDiagnosticsTests(unittest.TestCase):
    def test_nan_backward_preserves_batch_and_network(self) -> None:
        actor = nn.Linear(1, 1, bias=False)
        batch = {"state": torch.ones(1, 1)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "returned nan"):
                with diagnose_update(output, 7, {"actor": actor}, {"actor": batch}):
                    (actor(batch["state"]) * 0).sqrt().sum().backward()
            evidence = output / "diagnostics/iteration-7"
            self.assertEqual(json.loads((evidence / "failure.json").read_text())["iteration"], 7)
            saved = torch.load(evidence / "failure.pt", weights_only=True)
            torch.testing.assert_close(saved["batches"]["actor"]["state"], batch["state"])
            torch.testing.assert_close(saved["modules"]["actor"]["weight"], actor.weight)

    def test_float64_report_distinguishes_norm_overflow(self) -> None:
        actor = nn.Linear(2, 1, bias=False)
        actor.weight.grad = torch.full_like(actor.weight, 1e30)
        self.assertFalse(torch.isfinite(actor.weight.grad.norm()).item())
        report = gradient_report({"actor": actor})["actor.weight"]
        self.assertEqual(report["nonfinite_count"], 0)
        self.assertTrue(1e30 < report["finite_norm_float64"] < 2e30)

    def test_diagnostics_requires_eager_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "compile_training=False"):
            TrainConfig(numerical_diagnostics=True).validate()
        TrainConfig(numerical_diagnostics=True, compile_training=False, cuda_graph=False).validate()
