"""Replay boundaries, model fitting, analytic policy gradients and export contract."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rl_go2.config import load_experiment
from rl_go2.env import QuadrupedEnv
from rl_go2.trainers.model_learning import DynamicsModel, ReplayBuffer, pack, unpack, quat_mul, conjugate
from rl_go2.state import rotate_inverse
from rl_go2.trainers.supermimic import Trainer, TrainConfig, imagined_rollout, frozen, lambda_returns, train
from test_framework import FakeBackend

ROOT = Path(__file__).resolve().parents[1]


class SuperMimicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.config = load_experiment(ROOT / 'configs/experiments/flat_supermimic.yaml')
        self.cfg = TrainConfig(model_hidden=(32, 32), actor_hidden=(32,), critic_hidden=(32,),
                               collect_steps=4, model_window=3, model_batch=16, model_updates=1,
                               imagination_batch=8, horizon=4, critic_epochs=1, critic_minibatches=2)

    def make_trainer(self):
        env = QuadrupedEnv(self.config, backend=FakeBackend(self.config), record_transitions=True)
        return Trainer(env, self.cfg)

    def test_cuda_acceleration_defaults(self):
        config = TrainConfig()
        self.assertTrue(all((config.tf32, config.fused_adam, config.compile_training, config.cuda_graph)))

    def test_replay_wrap_and_terminal_transition(self):
        b = ReplayBuffer(4, 2, 'cpu')
        for t in range(7):
            b.add({'state': torch.full((2, 1), float(t)), 'next_state': torch.full((2, 1), t + 0.5),
                   'terminated': torch.full((2,), t == 5), 'truncated': torch.zeros(2, dtype=torch.bool)})
        w = b.sample_windows(100, 4)
        torch.testing.assert_close(w['state'][0, :, 0], torch.tensor([3., 4., 5., 6.]))
        torch.testing.assert_close(w['next_state'][0, :, 0], torch.tensor([3.5, 4.5, 5.5, 6.5]))
        torch.testing.assert_close(w['valid'], torch.tensor([1., 1., 1., 0.]).expand(100, -1))
        b.data['terminated'].zero_()
        b.data['truncated'][b._slots(torch.tensor(1))] = True
        torch.testing.assert_close(b.sample_windows(8, 4)['valid'], torch.tensor([1., 1., 0., 0.]).expand(8, -1))

    def test_official_windows_exclude_resets_and_score_one_fewer_step(self):
        replay = ReplayBuffer(4, 2, 'cpu')
        for t in range(7):
            state = torch.tensor([[float(t)], [float(t + 100)]])
            record = dict.fromkeys(('state', 'action', 'previous_action', 'torque', 'features'), state)
            record.update(terminated=torch.tensor([t == 4, t == 6]),
                          truncated=torch.zeros(2, dtype=torch.bool))
            replay.add(record)
        window = replay.sample_windows(8, 4, complete_frames=True)
        torch.testing.assert_close(window['state'][..., 0], torch.tensor([103., 104., 105.]).expand(8, -1))
        torch.testing.assert_close(window['next_state'], window['state'] + 1)
        self.assertTrue(window['valid'].all())

    def test_official_schedule_counts_collection_and_reuses_critic_shuffle(self):
        self.cfg.collect_steps = 1
        self.cfg.model_window = 8
        self.cfg.model_batch = 1
        self.cfg.model_updates = 10
        self.cfg.warmup_iterations = 0
        self.cfg.iterations = 11
        self.cfg.critic_epochs = 2
        self.cfg.bootstrap_weight = 1.0
        self.config['trainer'] = asdict(self.cfg)
        with tempfile.TemporaryDirectory() as directory, \
                patch('rl_go2.env.make_backend', return_value=FakeBackend(self.config)), \
                patch('torch.randperm', wraps=torch.randperm) as shuffle:
            output = Path(directory) / 'train'
            train(self.config, output)
            metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        self.assertEqual([row['environment_steps'] for row in metrics], list(range(2, 23, 2)))
        self.assertEqual([row['iteration'] for row in metrics if 'model_loss' in row], [9, 11])
        self.assertEqual(metrics[-1]['policy_updates'], 2)
        self.assertEqual(shuffle.call_count, 2)

    def test_context_matches_real_resample_and_reward(self):
        trainer = self.make_trainer()
        env = trainer.env
        env.task.resample_steps = 2
        env.task.command_steps_left.fill_(2)
        context = {k: v.clone() for k, v in env.task.context().items()}
        actual = env.observe()
        torch.testing.assert_close(actual, env.task.observe(env.state, env.previous_action, context))
        advanced = env.task.advance_context(context, env.state)
        torch.testing.assert_close(advanced['commands'], context['commands'])
        advanced = env.task.advance_context(advanced, env.state)
        self.assertTrue((advanced['command_steps_left'] == 2).all())
        self.assertTrue((context['command_steps_left'] == 2).all())
        env.task.commands.fill_(99)
        action = torch.randn(2, 12)
        expected = env.task.reward(env.state, action, env.previous_action, torch.zeros_like(action), context)
        env.task.commands.copy_(context['commands'])
        torch.testing.assert_close(expected, env.task.reward(env.state, action, env.previous_action, torch.zeros_like(action)))
        env.config['task']['episode_seconds'] = env.dt
        env.max_episode_length = 1
        _, _, _, extras = env.step(action)
        tr = extras['transition']
        torch.testing.assert_close(tr.previous_action, torch.zeros_like(action))
        self.assertTrue(tr.truncated.all())
        self.assertTrue((tr.next_state.position[:, 0] > env.state.position[:, 0]).all())

    def test_dynamics_heading_equivariance_and_learning(self):
        trainer = self.make_trainer()
        state = pack(trainer.env.state)
        model = DynamicsModel(0, (32,))
        torch.nn.init.normal_(model.net[-1].weight, std=0.01)
        action = torch.randn(2, 12) * .1
        features = torch.empty(2, 0)
        predicted, torque = model(state, action, action, features)
        yaw = .7
        rotation = torch.tensor([[torch.cos(torch.tensor(yaw/2)), 0., 0., torch.sin(torch.tensor(yaw/2))]]).expand(2, -1)
        def transform(x):
            s = unpack(x)
            s.position = rotate_inverse(conjugate(rotation), s.position) + torch.tensor([2., -3., 0.])
            s.quaternion = quat_mul(rotation, s.quaternion)
            s.linear_velocity = rotate_inverse(conjugate(rotation), s.linear_velocity)
            s.angular_velocity = rotate_inverse(conjugate(rotation), s.angular_velocity)
            return pack(s)
        other, other_torque = model(transform(state), action, action, features)
        torch.testing.assert_close(other, transform(predicted), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torque, other_torque, atol=1e-5, rtol=1e-5)
        # Recorded dynamics over multiple steps: q advances under the logged action.
        states, targets, actions, torques = [], [], [], []
        x = state[:1].repeat(16, 1)
        for _ in range(3):
            a = torch.randn(16, 12) * .2
            y = x.clone()
            y[:, 13:25] += .02 * a
            y[:, 25:] = a
            states.append(x); targets.append(y); actions.append(a); torques.append(a * 2)
            x = y
        w = {'state': torch.stack(states, 1), 'next_state': torch.stack(targets, 1),
             'action': torch.stack(actions, 1), 'previous_action': torch.zeros(16, 3, 12),
             'features': torch.empty(16, 3, 0), 'torque': torch.stack(torques, 1), 'valid': torch.ones(16, 3)}
        model = DynamicsModel(0, (64, 64))
        record = {k: v.reshape(v.shape[0] * v.shape[1], v.shape[-1]) for k, v in w.items() if k != 'valid'}
        model.update_stats(record)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        initial = model.loss(w).item()
        for _ in range(80):
            loss = model.loss(w)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertLess(model.loss(w).item(), initial * .5)

    def test_imagination_and_replay_use_applied_joint_limited_actions(self):
        self.cfg.action_noise = 0
        trainer = self.make_trainer()
        class ExtremeActor(torch.nn.Module):
            def forward(self, observation):
                return observation.new_full((observation.shape[0], 12), 5.)
        trainer.actor.forward = ExtremeActor().forward
        trainer.collect()
        expected = trainer.env.controller.action_upper
        torch.testing.assert_close(trainer.replay.data['action'][:4], expected.expand(4, 2, -1))
        seed = trainer.replay.sample_starts(8)
        seen = []
        def check_model_action(module, args):
            torch.testing.assert_close(args[1], expected.expand(8, -1))
            if seen:
                torch.testing.assert_close(args[2], expected.expand(8, -1))
            seen.append(True)
        handle = trainer.model.register_forward_pre_hook(check_model_action)
        try:
            imagined_rollout(trainer.env, trainer.model, ExtremeActor(), trainer.critic, seed, self.cfg)
            self.assertEqual(len(seen), self.cfg.horizon)
        finally:
            handle.remove()

    def test_actor_gradient_frozen_model_critic_and_export(self):
        self.cfg.bootstrap_weight = 1.0
        trainer = self.make_trainer()
        trainer.collect()
        trainer.fit_model()
        seed = trainer.replay.sample_starts(8)
        trainer.model.zero_grad(set_to_none=True)
        trainer.critic.zero_grad(set_to_none=True)
        before_model = deepcopy(trainer.model.state_dict())
        before_critic = deepcopy(trainer.critic.state_dict())
        with frozen(trainer.model, trainer.critic):
            objective, data, _ = imagined_rollout(trainer.env, trainer.model, trainer.actor, trainer.critic, seed, self.cfg)
            (-objective).backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.actor.parameters()))
            self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
            self.assertTrue(all(p.grad is None for p in trainer.critic.parameters()))
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before_model[name])
        for name, value in trainer.critic.state_dict().items():
            torch.testing.assert_close(value, before_critic[name])
        targets = lambda_returns(trainer.critic, data, .99, .95)
        self.assertFalse(targets.requires_grad)
        before = deepcopy(trainer.actor.state_dict())
        metrics = trainer.improve_policy()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(value, before[key]) for key, value in trainer.actor.state_dict().items()))
        with tempfile.TemporaryDirectory() as d:
            trainer.save(Path(d), 1)
            policy = torch.jit.load(str(Path(d) / 'policy.pt'))
            torch.testing.assert_close(policy(trainer.env.observe()), trainer.actor(trainer.env.observe()))

    def test_lambda_return_uses_termination_not_time_limit(self):
        class Value(torch.nn.Module):
            def forward(self, x):
                return x[..., 0]
        data = {'inputs': torch.tensor([[[1.], [2.], [3.], [4.]]]),
                'rewards': torch.tensor([[1., 2., 3.]]), 'continuations': torch.tensor([[1., 0., 1.]])}
        torch.testing.assert_close(lambda_returns(Value(), data, .5, 1.), torch.tensor([[2., 2., 5.]]))

    def test_physics_features_and_context_support_arbitrary_batches(self):
        trainer = self.make_trainer()
        env = trainer.env
        original = env.task.context
        env.task.context = lambda: {**original(), 'friction': torch.full((2, 1), .8)}
        # Small differentiable terrain probe: no backend-specific tensor indexing.
        env.task.model_features = lambda state, context: torch.cat((
            state.position[:, 2:3] - .05 * state.position[:, :1].sin(), context['friction']), -1)
        trainer = Trainer(env, self.cfg)
        trainer.collect()
        self.assertEqual(trainer.replay.data['features'].shape[-1], 2)
        seed = trainer.replay.sample_starts(5)
        self.assertEqual(seed['context/friction'].shape, (5, 1))
        state = seed['state'].detach().requires_grad_(True)
        features = env.task.model_features(unpack(state), {'friction': seed['context/friction']})
        features.sum().backward()
        self.assertTrue(state.grad[:, :3].abs().sum() > 0)
        trainer.fit_model()
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in trainer.improve_policy().values()))

    def test_dynamics_do_not_depend_on_previous_action(self):
        trainer = self.make_trainer()
        state = pack(trainer.env.state)
        action = torch.randn(2, 12, requires_grad=True)
        previous = torch.randn(2, 12, requires_grad=True)
        features = torch.empty(2, 0)
        expected = trainer.model(state, action, previous, features)
        actual = trainer.model(state, action, previous + 3, features)
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y)
        gradients = torch.autograd.grad(expected[0].sum(), (action, previous), allow_unused=True)
        self.assertIsNone(gradients[1])

    def test_per_foot_friction_reaches_contacts_and_model_replay(self):
        config = self.config
        config.update(backend='mujoco', device='cpu', num_envs=1)
        feet = ('FR', 'FL', 'RR', 'RL')
        base = torch.tensor([[.4, .01, .002], [.5, .02, .003], [.6, .03, .004], [.7, .04, .005]])
        config['domain_randomization'] = {'foot_ground_friction': {
            'enabled': True, 'env_multiplier_range': [.75, .75],
            'base_ranges': {
                foot: {term: [value, value] for term, value in zip(
                    ('sliding', 'torsional', 'rolling'), row.tolist())}
                for foot, row in zip(feet, base)}}}
        env = QuadrupedEnv(config, record_transitions=True)
        self.addCleanup(env.close)
        expected = base * .75
        trainer = Trainer(env, self.cfg)
        self.assertEqual(trainer.model.input_norm.mean.numel(), 59)
        self.assertEqual(env.observe().shape[-1], 45)
        trainer.collect()
        seed = trainer.replay.sample_starts(3)
        torch.testing.assert_close(seed['features'], expected.flatten().expand(3, -1))
        torch.testing.assert_close(seed['context/foot_ground_friction'], seed['features'])
        trainer.fit_model()
        trainer.improve_policy()

        env.reset()
        backend = env.backend
        ground = backend.model.geom('ground').id
        foot_rows = dict(zip(backend.foot_ids, expected))
        seen = set()
        for _ in range(20):
            env.step(torch.zeros(1, 12))
            for contact in backend.data.contact:
                if ground not in contact.geom:
                    continue
                foot = next((geom for geom in contact.geom if geom in foot_rows), None)
                if foot is None:
                    continue
                self.assertEqual(contact.dim, 6)
                torch.testing.assert_close(torch.tensor(contact.friction[[0, 2, 3]], dtype=base.dtype), foot_rows[foot])
                seen.add(foot)
        self.assertEqual(seen, set(foot_rows))

    def test_data_driver_never_changes_or_exports_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = torch.nn.Linear(45, 12)
            torch.nn.init.zeros_(driver.weight)
            torch.nn.init.ones_(driver.bias)
            path = Path(directory) / 'driver.pt'
            torch.jit.script(driver).save(str(path))
            self.cfg.driver_policy = str(path)
            self.cfg.action_noise = 0
            env = QuadrupedEnv(self.config, backend=FakeBackend(self.config), record_transitions=True)
            trainer = Trainer(env, self.cfg)
            before = deepcopy(trainer.actor.net.state_dict())
            trainer.collect()
            torch.testing.assert_close(trainer.replay.data['action'][:4, 1], torch.ones(4, 12))
            self.assertTrue(trainer.replay.data['is_driver'][:4, 1].all())
            self.assertFalse(trainer.replay.data['is_driver'][:4, 0].any())
            for key, value in trainer.actor.net.state_dict().items():
                torch.testing.assert_close(value, before[key])
            trainer.save(Path(directory), 1)
            exported = torch.jit.load(str(Path(directory) / 'policy.pt'))
            torch.testing.assert_close(exported(env.observe()), trainer.actor(env.observe()))
            self.assertFalse(torch.allclose(exported(env.observe()), driver(env.observe())))

    def test_guidance_learns_and_restores_policy_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = torch.nn.Linear(45, 12)
            torch.nn.init.zeros_(driver.weight)
            torch.nn.init.ones_(driver.bias)
            path = Path(directory) / 'driver.pt'
            torch.jit.script(driver).save(str(path))
            self.cfg.driver_policy = str(path)
            self.cfg.behavior_updates = 4
            self.cfg.actor_learning_rate = .00005
            env = QuadrupedEnv(self.config, backend=FakeBackend(self.config), record_transitions=True)
            trainer = Trainer(env, self.cfg)
            trainer.collect()
            seed = trainer.replay.sample_starts(8)
            initial = trainer.behavior_loss(seed).item()
            for _ in range(30):
                trainer.warmup_actor()
            self.assertLess(trainer.behavior_loss(seed).item(), initial * .1)
            self.assertEqual(trainer.actor_optimizer.param_groups[0]['lr'], self.cfg.actor_learning_rate)
            self.assertTrue(all(parameter.grad is None for parameter in trainer.driver.parameters()))
            self.cfg.behavior_weight = .5
            self.assertTrue(torch.isfinite(torch.tensor(trainer.improve_policy()['behavior_loss'])))

    def test_disabled_bootstrap_never_queries_critic(self):
        class Poison(torch.nn.Module):
            def forward(self, x):
                raise AssertionError('Disabled critic must not affect the actor')
        trainer = self.make_trainer()
        trainer.collect()
        seed = trainer.replay.sample_starts(8)
        objective, data, _ = imagined_rollout(trainer.env, trainer.model, trainer.actor, Poison(), seed, self.cfg)
        self.assertTrue(torch.isfinite(objective))
        self.assertEqual(data, {})

    def test_config_validation(self):
        for change in ({'model_window': 300}, {'horizon': 0}, {'gamma': 1.}, {'height_softness': 0.}, {'behavior_updates': 1}, {'behavior_weight': -1.}, {'bootstrap_weight': 2.}):
            with self.assertRaises(ValueError):
                TrainConfig(**change).validate()


if __name__ == '__main__':
    unittest.main()
