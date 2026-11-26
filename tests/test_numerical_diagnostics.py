"""Check abnormal-update snapshots and replay diagnostics on CPU."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from rl_go2.trainers.numerical_diagnostics import NumericalDiagnostics, gradient_report, restore_rng
from rl_go2.trainers.supermimic import TrainConfig, optimize


class NumericalDiagnosticsTests(unittest.TestCase):
    def test_actor_failure_saves_unclipped_gradients_and_pre_forward_rng(self) -> None:
        actor = nn.Linear(1, 1, bias=False)
        original_weight = actor.weight.detach().clone()
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
        batch = {"state": torch.ones(1, 1)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            capture = NumericalDiagnostics(output, torch.device('cpu'), 1.0, 0)
            capture.iteration = 7
            diagnostic = capture.begin('actor', batch, {'actor': actor}, cuda_graph=False)
            forward_draw = torch.rand(2)
            loss = (actor(batch['state']) * 0).sqrt().sum()
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                optimize(loss, optimizer, actor.parameters(), 1.0, diagnostic=diagnostic)
            evidence = output / 'diagnostics/iteration-7-actor-failure'
            report = json.loads((evidence / 'snapshot.json').read_text())
            self.assertEqual(report['phase'], 'gradients')
            self.assertEqual(report['gradients']['actor.weight']['nonfinite_count'], 1)
            saved = torch.load(evidence / 'snapshot.pt', weights_only=True)
            torch.testing.assert_close(saved['batch']['state'], batch['state'])
            torch.testing.assert_close(saved['modules']['actor']['weight'], original_weight)
            torch.testing.assert_close(actor.weight, original_weight)
            with torch.random.fork_rng(devices=[]):
                restore_rng(saved['rng'], torch.device('cpu'))
                torch.testing.assert_close(torch.rand(2), forward_draw)

    def test_model_spike_saves_once_after_warmup_before_optimizer_step(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = {'input': torch.ones(1, 1)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            capture = NumericalDiagnostics(output, torch.device('cpu'), 1.0, 2)
            for iteration in (2, 3, 4):
                capture.iteration = iteration
                diagnostic = capture.begin('model', batch, {'world_model': model}, cuda_graph=False)
                optimize(model(batch['input']).square().sum(), optimizer, model.parameters(), 1.0,
                         diagnostic=diagnostic)
            paths = list((output / 'diagnostics').glob('*/snapshot.pt'))
            self.assertEqual([path.parent.name for path in paths], ['iteration-3-model-loss-spike'])
            saved = torch.load(paths[0], weights_only=True)
            torch.testing.assert_close(saved['modules']['world_model']['weight'], torch.tensor([[1.9]]))
            self.assertEqual(saved['gradients'], {})
            self.assertLess(model.weight.item(), 1.9)

    def test_float64_report_distinguishes_norm_overflow(self) -> None:
        actor = nn.Linear(2, 1, bias=False)
        actor.weight.grad = torch.full_like(actor.weight, 1e30)
        self.assertFalse(torch.isfinite(actor.weight.grad.norm()).item())
        report = gradient_report({"actor": actor})["actor.weight"]
        self.assertEqual(report["nonfinite_count"], 0)
        self.assertTrue(1e30 < report["finite_norm_float64"] < 2e30)

    def test_diagnostics_accepts_the_configured_acceleration_modes(self) -> None:
        TrainConfig(numerical_diagnostics=True, compile_training=True, cuda_graph=True).validate()
        with self.assertRaisesRegex(ValueError, "diagnostic_model_loss_threshold"):
            TrainConfig(diagnostic_model_loss_threshold=float('inf')).validate()

    def test_model_snapshot_replays_from_saved_run_config(self) -> None:
        from rl_go2.config import load_experiment
        from rl_go2.env import QuadrupedEnv
        from rl_go2.trainers import save_run_spec
        from rl_go2.trainers.diagnostic_replay import replay_diagnostic
        from rl_go2.trainers.supermimic import Trainer

        repo = Path(__file__).resolve().parents[1]
        config = load_experiment(repo / 'configs/experiments/flat_supermimic.yaml')
        config.update(backend='mujoco', device='cpu', num_envs=1)
        config['trainer'].update(model_hidden=[16], actor_hidden=[16], critic_hidden=[16],
                                 model_window=2, model_batch=2, model_updates=1,
                                 compile_training=False, cuda_graph=False)
        env = QuadrupedEnv(config, record_transitions=True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'training'
                save_run_spec(config, output)
                capture = NumericalDiagnostics(output, env.device, 1e-12, 0)
                capture.iteration = 1
                trainer = Trainer(env, TrainConfig(**config['trainer']), capture)
                trainer.collect()
                trainer.collect()
                loss = trainer.fit_model()
                snapshot = output / 'diagnostics/iteration-1-model-loss-spike/snapshot.pt'
                with redirect_stdout(io.StringIO()) as report:
                    replay_diagnostic(snapshot)
                result = json.loads(report.getvalue())
                self.assertAlmostEqual(result['eager']['loss'], loss)
                self.assertIn('output_channel', result['eager']['largest_model_error'])
        finally:
            env.close()
