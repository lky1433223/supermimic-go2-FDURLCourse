"""Replay boundaries, model fitting, analytic policy gradients and export contract."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rl_go2.backends import LinkPointForces
from rl_go2.config import FOOT_NAMES, load_experiment, load_scene
from rl_go2.env import QuadrupedEnv
from rl_go2.point_push import Go2PushGeometry, PointPushConfig, PointPushScheduler, PushState
from rl_go2.trainers.model_learning import DynamicsModel, ReplayBuffer, pack, unpack, quat_mul, conjugate
from rl_go2.state import rotate_inverse
from rl_go2.tasks.flat_walk import FlatWalkTask
from rl_go2.trainers.supermimic import Trainer, TrainConfig, imagined_rollout, frozen, lambda_returns, train
from rl_go2.trainers.supermimic_init import initialize_weights
from rl_go2.trainers.terrain_features import TerrainHeightScan
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

    def test_sensor_noise_history_continues_in_imagination(self):
        self.config['domain_randomization'] = {'observation_noise': {
            'attitude': [{'type': 'uniform', 'low': [-0.02, 0, 0], 'high': [0.02, 0, 0]}],
            'joint_position': [{'type': 'bias', 'offset': 0.1}, {'type': 'delay', 'steps': 1}],
            'joint_velocity': [{'type': 'gaussian', 'std': 0.2}, {'type': 'dropout', 'probability': 1}],
        }}
        trainer = self.make_trainer()
        env = trainer.env
        initial = env.observe()
        torch.testing.assert_close(env.observe(), initial)
        torch.testing.assert_close(initial[:, 3:6].norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(initial[:, 9:21] - env.observe(clean=True)[:, 9:21], torch.full((2, 12), 0.1))
        env.backend.default = env.backend.default + 0.2
        env.step(torch.zeros(2, 12))
        torch.testing.assert_close(env.observe()[:, 9:33], initial[:, 9:33])
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.observe()[0, 9:21], torch.full((12,), 0.3))
        torch.testing.assert_close(env.observe()[1, 9:33], initial[1, 9:33])

        trainer.collect()
        seed = trainer.replay.sample_starts(2)
        seen = []

        def record_input(module, args):
            seen.append(args[0].detach())

        handle = trainer.actor.register_forward_pre_hook(record_input)
        try:
            objective, _, _ = imagined_rollout(env, trainer.model, trainer.actor, trainer.critic, seed, self.cfg)
            objective.backward()
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], seed['actor_observation'])
        torch.testing.assert_close(seen[1][:, 9:21], seed['sensor_history/joint_position/1'][:, 0] - env.task.default)
        torch.testing.assert_close(seen[-1][:, 21:33], seen[0][:, 21:33])

    def test_frame_delay_is_shared_by_sensors_and_fixed_until_reset(self):
        self.config['domain_randomization'] = {'observation_noise': {
            'frame_delay_steps': [0, 1],
            'projected_gravity': [{'type': 'bias', 'offset': [0.05, 0.0, -0.05]}],
        }}
        with patch('rl_go2.observation_noise.torch.randint', return_value=torch.tensor([0, 1])) as sample_delay:
            trainer = self.make_trainer()
            env = trainer.env
            env.backend.default += 0.2
            observed, _, _, _ = env.step(torch.zeros(2, 12))
            sample_delay.assert_called_once_with(0, 2, (2,), device=env.device)
        torch.testing.assert_close(observed[:, 3:6], torch.tensor([[0.05, 0.0, -1.05]]).expand(2, -1))
        torch.testing.assert_close(observed[:, 9:21], torch.tensor([[0.2], [0.0]]).expand(-1, 12))

        with patch('rl_go2.observation_noise.torch.randint', return_value=torch.tensor([0])) as sample_delay:
            env.reset(torch.tensor([1]))
            sample_delay.assert_called_once_with(0, 2, (1,), device=env.device)
        torch.testing.assert_close(env.observe()[:, 9:21], torch.full((2, 12), 0.2))
        self.assertEqual(env.noise_history['frame_delay/steps'].tolist(), [0, 0])

    def test_action_delay_applies_queued_commands_and_preserves_imagination_gradients(self):
        self.config['domain_randomization'] = {
            'action_delay_steps': [0, 2],
            'observation_noise': {'frame_delay_steps': [1, 1]},
        }
        with patch('rl_go2.step_delay.torch.randint', return_value=torch.tensor([0, 2])) as sample_delay:
            trainer = self.make_trainer()
            env = trainer.env
            for issued, delayed in ((0.1, 0.0), (0.2, 0.0), (0.3, 0.1)):
                command = torch.full((2, 12), issued)
                observation, _, _, extras = env.step(command)
                expected = torch.tensor([[issued], [delayed]]).expand(-1, 12)
                torch.testing.assert_close(extras['transition'].action, expected)
                torch.testing.assert_close(observation[:, -12:], command)
            sample_delay.assert_called_once_with(0, 3, (2,), device=env.device)

        with patch('rl_go2.step_delay.torch.randint', return_value=torch.tensor([1])):
            env.reset(torch.tensor([0]))
        self.assertEqual(env.action_delay_state.steps.tolist(), [1, 2])
        torch.testing.assert_close(env.action_delay_state.history[0], torch.zeros(2, 12))
        trainer.collect()
        seed = {key: frames[0] for key, frames in trainer.replay.data.items()}
        torch.testing.assert_close(seed['action'], torch.tensor([[0.0], [0.2]]).expand(-1, 12))

        issued_actions, model_actions = [], []

        def record_command(module, args, output):
            issued_actions.append(output)

        def record_applied_action(module, args):
            model_actions.append(args[1])

        actor_hook = trainer.actor.register_forward_hook(record_command)
        model_hook = trainer.model.register_forward_pre_hook(record_applied_action)
        try:
            imagined_rollout(env, trainer.model, trainer.actor, trainer.critic, seed, self.cfg)
        finally:
            actor_hook.remove()
            model_hook.remove()
        torch.testing.assert_close(model_actions[0], seed['action'])
        torch.testing.assert_close(model_actions[1][1], seed['action_queue'][1, 0])
        torch.testing.assert_close(model_actions[2][1], env.controller.clip_action(issued_actions[0])[1])
        gradient, = torch.autograd.grad(model_actions[2][1].sum(), issued_actions[0])
        self.assertGreater(gradient[1].abs().sum().item(), 0)

    def test_terrain_scan_reads_exact_tiles_in_the_yaw_frame(self):
        scene = load_scene(ROOT / 'configs/scenes/rough_ground.yaml')
        patch = scene['height_maps']['rough']
        scan = TerrainHeightScan(scene)
        positions = torch.tensor([[1.38, 0.02, 0.30], [1.38, 0.02, 0.30], [-1.0, 0.0, 0.30]])
        yaw_90 = math.sqrt(0.5)
        inverse_headings = torch.tensor([[1, 0, 0, 0], [yaw_90, 0, 0, -yaw_90], [1, 0, 0, 0]])
        expected = []
        for sample, (base_x, base_y, base_z) in enumerate(positions.tolist()):
            local = []
            for row in range(-3, 6):
                for column in range(-5, 6, 2):
                    x, y = row / 10, column / 20
                    if sample == 1:
                        x, y = -y, x
                    tile_x = math.floor((base_x + x - 0.8) / 0.30)
                    tile_y = math.floor((base_y + y + 0.6) / 0.30)
                    height = patch['height_map_m'][tile_x][tile_y] if 0 <= tile_x < 5 and 0 <= tile_y < 4 else 0
                    local.append(height - base_z)
            expected.append(local)
        torch.testing.assert_close(scan(positions, inverse_headings), torch.tensor(expected))

    def test_world_model_requeries_height_at_predicted_positions(self):
        flat = self.make_trainer()
        self.config['scene'] = load_scene(ROOT / 'configs/scenes/rough_ground.yaml')
        self.cfg.bootstrap_weight = 1.0
        trainer = self.make_trainer()
        self.assertEqual(trainer.actor.norm.mean.shape, flat.actor.norm.mean.shape)
        self.assertEqual(trainer.critic.norm.mean.shape, flat.critic.norm.mean.shape)
        self.assertEqual(trainer.model.net[0].in_features, flat.model.net[0].in_features + 54)
        trainer.env.backend.position[:, 0] = 0.76
        trainer.env.state = trainer.env.backend.read_state()
        trainer.collect()
        self.assertGreater(trainer.model.terrain_norm.count.item(), 1)
        window = trainer.replay.sample_windows(2, self.cfg.model_window)
        self.assertTrue(torch.isfinite(trainer.model.loss(window)))

        # Make the model advance by one tile per imagined step.
        trainer.model.output_norm.mean[0] = 0.30
        positions, scans = [], []

        def record_scan(module, args, output):
            positions.append(args[0][:, 0].detach().clone())
            scans.append(output.detach().clone())

        handle = trainer.model.terrain_scan.register_forward_hook(record_scan)
        try:
            seed = trainer.replay.sample_starts(2)
            objective, _, _ = imagined_rollout(trainer.env, trainer.model, trainer.actor, trainer.critic, seed, self.cfg)
            objective.backward()
        finally:
            handle.remove()
        self.assertEqual(len(positions), self.cfg.horizon)
        torch.testing.assert_close(positions[0], seed['state'][:, 0])
        torch.testing.assert_close(positions[1], positions[0] + 0.30)
        self.assertFalse(torch.equal(scans[1], scans[0]))

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

    def test_five_link_wrenches_reach_replay_model_and_imagination(self):
        self.config.update(backend='mujoco', device='cpu', num_envs=1)
        recipe = load_experiment(ROOT / 'configs/experiments/flat_supermimic_bootstrap.yaml')
        self.config['domain_randomization'] = recipe['domain_randomization']
        for section in self.config['domain_randomization'].values():
            section['enabled'] = True
        robot = next(entity for entity in self.config['scene']['entities'] if entity.get('robot'))
        robot.update(position=[0, 0, 2], rotation=[2**-0.5, 0, 0, 2**-0.5])
        self.cfg.condition_on_link_wrenches = True
        self.cfg.warmup_iterations = 0
        self.cfg.bootstrap_weight = 1.0
        env = QuadrupedEnv(self.config, record_transitions=True)
        self.addCleanup(env.close)
        trainer = Trainer(env, self.cfg)
        self.assertEqual(trainer.model.net[0].in_features, 93)
        self.assertEqual(trainer.model.net[-1].out_features, 49)
        self.assertEqual(trainer.actor.net[0].in_features, 45)
        self.assertEqual(trainer.critic.net[0].in_features, 73)

        link_names = env.backend.link_names
        loads = LinkPointForces(torch.zeros(1, len(link_names), 3), torch.zeros(1, len(link_names), 3))
        base = link_names.index('base')
        calf = link_names.index('FR_calf')
        loads.point_local_m[0, base, 1] = 0.2
        loads.force_world_n[0, base, 1] = 10
        loads.force_world_n[0, calf, 2] = 5
        trainer.collect(point_force_plan=[loads, None, None, None])

        replay = trainer.replay.data
        recorded = replay['link_wrenches_world'][0]
        self.assertEqual(recorded.shape, (1, 5, 6))
        torch.testing.assert_close(recorded[0, 0], torch.tensor([0., 10., 0., 0., 0., -2.]))
        torch.testing.assert_close(recorded[0, 1], torch.tensor([0., 0., 5., 0., 0., 0.]))
        self.assertFalse(replay['link_wrenches_world'][1:4].any())
        record = {name: value[0] for name, value in replay.items()}
        inputs = trainer.model.inputs(record['state'], record['action'], record['previous_action'],
                                      record['features'], recorded)
        torch.testing.assert_close(inputs[0, -30:-24], torch.tensor([10., 0., 0., 0., 0., -2.]),
                                   atol=1e-5, rtol=1e-5)
        window = trainer.replay.sample_windows(4, self.cfg.model_window, complete_frames=True)
        self.assertEqual(window['link_wrenches_world'].shape, (4, 2, 5, 6))
        self.assertTrue(torch.isfinite(trainer.model.loss(window)))

        plan = recorded.new_zeros((1, self.cfg.horizon, 5, 6))
        plan[:, 1] = recorded
        seen = []
        with trainer.model.register_forward_pre_hook(lambda _, args: seen.append(args[4])):
            imagined_rollout(env, trainer.model, trainer.actor, trainer.critic, record, self.cfg,
                             wrench_plan_world=plan)
        torch.testing.assert_close(torch.stack(seen, dim=1), plan)

    def test_scheduled_pushes_continue_in_imagination_and_reset(self):
        self.check_scheduled_pushes('mujoco')

    def test_random_push_duration_is_preserved_in_replay(self):
        config = load_experiment(ROOT / 'configs/experiments/flat_supermimic_push.yaml')
        push = config['domain_randomization']['point_push']
        push.update(first_wait_seconds=[0, 0], gap_seconds=[1, 1], no_push_episode_probability=0)
        spec = PointPushConfig.from_config(config)
        geometry = Go2PushGeometry(config, FOOT_NAMES, torch.device('cpu'))
        scheduler = PointPushScheduler(spec, 0.02, geometry)
        state = scheduler.reset(torch.zeros(128, 3))
        link_quat = torch.zeros(128, 5, 4)
        link_quat[..., 0] = 1
        state = scheduler.prepare(state, link_quat)
        state = PushState.from_replay(state.replay_fields())
        expected = state.duration_steps.amax(dim=1)
        self.assertEqual(expected.min().item(), 5)
        self.assertEqual(expected.max().item(), 15)
        torch.testing.assert_close(state.wait, expected + 50)

        active_steps = torch.zeros_like(expected)
        for _ in range(16):
            active_steps += scheduler.loads(state).force_world_n.ne(0).any(dim=(1, 2))
            state = scheduler.prepare(scheduler.advance(state), link_quat)
        torch.testing.assert_close(active_steps, expected)

    def test_course_base_push_impulse_caps_duration_in_whole_steps(self):
        config = load_experiment(ROOT / 'configs/experiments/curriculum_supermimic.yaml')
        config.update(config['courses'][4]['settings'])
        push = config['domain_randomization']['point_push']
        push.update(first_wait_seconds=[0, 0], gap_seconds=[1, 1],
                    no_push_episode_probability=0, duration_seconds=0.3)
        push['targets']['base']['probability'] = 1
        push['targets']['foot']['probability'] = 0
        geometry = Go2PushGeometry(config, FOOT_NAMES, torch.device('cpu'))
        link_quat = torch.zeros(1, 5, 4)
        link_quat[..., 0] = 1
        dt = 0.02
        for force, expected_steps in ((200, 5), (160, 6), (50, 15)):
            with self.subTest(force=force):
                push['targets']['base']['force_range_n'] = [force, force]
                scheduler = PointPushScheduler(PointPushConfig.from_config(config), dt, geometry)
                state = scheduler.prepare(scheduler.reset(torch.zeros(1, 3)), link_quat)
                self.assertEqual(state.wait.item(), 50 + expected_steps)
                state = PushState.from_replay(state.replay_fields())
                impulse = 0.0
                for _ in range(16):
                    impulse += scheduler.loads(state).force_world_n[0, 0].norm().item() * dt
                    state = scheduler.prepare(scheduler.advance(state), link_quat)
                self.assertAlmostEqual(impulse, force * expected_steps * dt, places=5)

    @unittest.skipUnless(os.environ.get('RL_GO2_TEST_GENESIS') == '1', 'opt-in Genesis CPU kernel compilation')
    def test_genesis_scheduled_pushes_continue_in_imagination_and_reset(self):
        self.check_scheduled_pushes('genesis')

    @unittest.skipUnless(os.environ.get('RL_GO2_TEST_CUDA') == '1' and torch.cuda.is_available(),
                         'opt-in Genesis CUDA graph regression')
    def test_cuda_graph_actor_update_with_scheduled_pushes(self):
        self.check_scheduled_pushes('genesis', device='cuda')

    def check_scheduled_pushes(self, backend: str, *, device: str = 'cpu') -> None:
        config = load_experiment(ROOT / 'configs/experiments/flat_supermimic_push.yaml')
        config.update(backend=backend, device=device, num_envs=1)
        push = config['domain_randomization']['point_push']
        push.update(no_push_episode_probability=0.0, first_wait_seconds=[0.02, 0.02],
                    gap_seconds=[0.2, 0.2], duration_seconds=0.06)
        push['targets']['base']['probability'] = 1.0
        push['targets']['foot']['probability'] = 0.0
        push['overlap'] = dict(probability=1.0, second_start_delay_steps=[1, 1])
        self.cfg.collect_steps = 6
        self.cfg.horizon = 6
        self.cfg.condition_on_link_wrenches = True
        env = QuadrupedEnv(config, record_transitions=True)
        self.addCleanup(env.close)
        trainer = Trainer(env, self.cfg)
        scheduler = trainer.push_scheduler
        predicted = scheduler.geometry.link_quaternions(env.state)
        native = env.backend.read_link_quaternions()[:, trainer.wrench_link_indices]
        aligned = torch.where((predicted * native).sum(dim=-1, keepdim=True) < 0, -predicted, predicted)
        torch.testing.assert_close(aligned, native, atol=1e-6, rtol=0)

        stats = trainer.collect()
        self.assertAlmostEqual(stats['push_overlap_fraction'], 2 / 6)
        replay = trainer.replay.data
        forces = replay['link_wrenches_world'][:6, 0, :, :3]
        self.assertEqual(forces.norm(dim=-1).gt(0).sum(dim=-1).tolist(), [0, 1, 2, 2, 1, 0])
        torch.testing.assert_close(forces[1:4, 0], forces[1, 0].expand(3, -1))
        seed = {name: value[2] for name, value in replay.items()}
        seen = []
        with trainer.model.register_forward_pre_hook(lambda _, args: seen.append(args[4])):
            imagined_rollout(env, trainer.model, trainer.actor, trainer.critic, seed, self.cfg,
                             push_scheduler=scheduler)
        resumed = torch.stack(seen)[:, 0, :, :3]
        torch.testing.assert_close(resumed[:4], forces[2:6])
        if device == 'cuda':
            stats = trainer.improve_policy()
            self.assertIsNotNone(trainer.graphed_actor_backward)
            self.assertGreater(stats['actor_grad_norm'], 0.0)
        else:
            loss, _ = trainer.actor_loss(seed)
            loss.backward()
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in trainer.actor.parameters()))

        env.max_episode_length = 1
        self.cfg.collect_steps = 2
        trainer.collect()
        latest = {name: value[7] for name, value in replay.items()}
        self.assertEqual(PushState.from_replay(latest).wait.item(), 1)
        self.assertFalse(latest['link_wrenches_world'].any())

    def test_checkpoint_initialization_appends_zero_wrench_weights(self):
        source = self.make_trainer()
        source.collect()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'config.json').write_text(json.dumps(self.config))
            source.save(output, 40)
            backend = FakeBackend(self.config)
            backend.link_names = ('base', 'FR_calf', 'FL_calf', 'RR_calf', 'RL_calf')
            self.cfg.condition_on_link_wrenches = True
            target = Trainer(QuadrupedEnv(self.config, backend=backend, record_transitions=True), self.cfg)
            initialize_weights(target, output / 'model_40.pt')
        old_weight = source.model.net[0].weight
        torch.testing.assert_close(target.model.net[0].weight[:, :old_weight.shape[1]], old_weight)
        self.assertFalse(target.model.net[0].weight[:, -30:].any())
        torch.testing.assert_close(target.model.input_norm.mean, source.model.input_norm.mean)
        torch.testing.assert_close(target.actor.norm.mean, source.actor.norm.mean)
        self.assertFalse(target.actor_optimizer.state)

    def test_flat_checkpoint_initializes_extra_terrain_inputs(self):
        source = self.make_trainer()
        source.collect()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'config.json').write_text(json.dumps(self.config))
            source.save(output, 40)
            self.config['scene'] = load_scene(ROOT / 'configs/scenes/rough_ground.yaml')
            target = self.make_trainer()
            initialize_weights(target, output / 'model_40.pt')
        old_weight = source.model.net[0].weight
        torch.testing.assert_close(target.model.net[0].weight[:, :old_weight.shape[1]], old_weight)
        self.assertFalse(target.model.net[0].weight[:, -54:].any())
        state = pack(source.env.state)
        action = source.env.previous_action
        features = source.env.task.model_features(source.env.state, source.env.model_context())
        torch.testing.assert_close(target.model(state, action, action, features),
                                   source.model(state, action, action, features))

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

    def test_curriculum_preserves_optimizer_progress_across_six_courses(self):
        config = load_experiment(ROOT / 'configs/experiments/curriculum_supermimic.yaml')
        config.update(backend='mujoco', device='cpu', num_envs=1)
        self.cfg.iterations = 12
        self.cfg.warmup_iterations = 0
        self.cfg.bootstrap_weight = 1.0
        self.cfg.condition_on_link_wrenches = True
        config['trainer'] = asdict(self.cfg)
        for index, course in enumerate(config['courses'], start=1):
            course.update(iterations=2, end_iteration=2 * index)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'train'
            train(config, output)
            metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            checkpoints = [torch.load(output / f'model_{iteration}.pt', weights_only=True)
                           for iteration in range(2, 13, 2)]
        self.assertEqual([row['course_stage'] for row in metrics], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        self.assertEqual([row['environment_steps'] for row in metrics], list(range(4, 49, 4)))
        for name in ('actor', 'model', 'critic'):
            steps = [saved[f'{name}_optimizer']['state'][0]['step'].item() for saved in checkpoints]
            self.assertTrue(all(left < right for left, right in zip(steps, steps[1:])), (name, steps))
        widths = [saved['world_model']['net.0.weight'].shape[1] for saved in checkpoints]
        self.assertEqual(widths, [77, 77, 77, 93, 93, 147])

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

    def test_course_command_sampling_matches_between_reset_and_imagination(self):
        config = load_experiment(ROOT / 'configs/experiments/curriculum_supermimic.yaml')
        for course in config['courses']:
            self.assertEqual(course['settings']['task']['command_sampling'],
                             {'stand_probability': 0.05, 'turn_in_place_probability': 0.05})
        task = FlatWalkTask(config['task'], torch.zeros(12), 4, torch.device('cpu'), 0.02, seed=1)
        draws = torch.tensor([
            [.75, .75, .75, .049],  # stand
            [.75, .75, .25, .051],  # turn
            [.75, .75, .50, .099],  # turn, even when the yaw draw lands exactly at zero
            [.75, .75, .75, .101],  # regular command
        ])
        expected = torch.tensor([[0, 0, 0], [0, 0, -.25], [0, 0, .5], [.625, .25, .25]])
        with patch('torch.rand', return_value=draws):
            task.reset(torch.arange(4), None, {})
            torch.testing.assert_close(task.commands, expected)
            context = {**task.context(), 'commands': torch.zeros_like(task.commands),
                       'command_steps_left': torch.ones_like(task.command_steps_left)}
            torch.testing.assert_close(task.advance_context(context, None)['commands'], expected)

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

    def test_base_mass_reaches_physics_and_model_replay(self):
        config = self.config
        config.update(backend='mujoco', device='cpu', num_envs=1)
        config['domain_randomization'] = {'base_mass': {'enabled': True, 'scale_range': [1.1, 1.1]}}
        env = QuadrupedEnv(config, record_transitions=True)
        self.addCleanup(env.close)
        backend = env.backend
        env.reset()
        mass = backend.nominal_base_mass * 1.1
        self.assertAlmostEqual(backend.model.body_mass[backend.base], mass, places=6)
        torch.testing.assert_close(torch.from_numpy(backend.model.body_inertia[backend.base]),
                                   torch.from_numpy(backend.nominal_base_inertia) * 1.1)
        trainer = Trainer(env, self.cfg)
        self.assertEqual(trainer.model.input_norm.mean.numel(), 48)
        self.assertEqual(env.observe().shape[-1], 45)
        trainer.collect()
        seed = trainer.replay.sample_starts(3)
        torch.testing.assert_close(seed['features'], torch.full((3, 1), mass))
        torch.testing.assert_close(seed['context/base_mass_kg'], seed['features'])
        trainer.fit_model()
        trainer.improve_policy()

    def test_base_com_offset_survives_reset_and_reaches_replay(self):
        config = self.config
        config.update(backend='mujoco', device='cpu', num_envs=1)
        offset = torch.tensor([.01, -.02, .03])
        config['domain_randomization'] = {'base_com': {
            'enabled': True, 'offset_ranges_m': {
                axis: [value, value] for axis, value in zip('xyz', offset.tolist())}}}
        env = QuadrupedEnv(config, record_transitions=True)
        self.addCleanup(env.close)
        env.reset()
        backend = env.backend
        displacement = backend.model.body_ipos[backend.base] - backend.nominal_base_com
        torch.testing.assert_close(torch.from_numpy(displacement).float(), offset)
        trainer = Trainer(env, self.cfg)
        self.assertEqual(trainer.model.input_norm.mean.numel(), 50)
        trainer.collect()
        seed = trainer.replay.sample_starts(3)
        torch.testing.assert_close(seed['features'], offset.expand(3, -1))
        torch.testing.assert_close(seed['context/base_com_offset_m'], seed['features'])

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

    def test_timed_friction_switch_preserves_transition_and_advances_imagination(self):
        config = load_experiment(ROOT / 'configs/experiments/flat_supermimic_bootstrap.yaml')
        config.update(backend='mujoco', device='cpu', num_envs=1)
        config['domain_randomization']['foot_ground_friction'].update(
            enabled=True, env_multiplier_range=[.75, 1.25], switch_interval_seconds=[.04, .04])
        env = QuadrupedEnv(config, record_transitions=True)
        self.addCleanup(env.close)
        initial = env.foot_friction.clone()
        action = torch.zeros(1, 12)
        env.step(action)
        torch.testing.assert_close(env.foot_friction, initial)
        _, _, _, extras = env.step(action)
        context = extras['transition'].context
        torch.testing.assert_close(context['foot_ground_friction'], initial)
        self.assertEqual(context['friction_steps_left'].item(), 1)
        self.assertEqual(env.friction_steps_left.item(), 2)
        self.assertFalse(torch.equal(env.foot_friction, initial))
        actual = env.backend.model.geom_friction[env.backend.foot_ids]
        torch.testing.assert_close(torch.tensor(actual).float(), env.foot_friction.reshape(4, 3))
        imagined = env.advance_context(context, env.state)
        self.assertEqual(imagined['friction_steps_left'].item(), 2)
        self.assertFalse(torch.equal(imagined['foot_ground_friction'], initial))
        trainer = Trainer(env, self.cfg)
        trainer.collect()
        self.assertTrue(torch.isfinite(torch.tensor(trainer.fit_model())))
        trainer.improve_policy()

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
