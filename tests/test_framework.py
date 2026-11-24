"""Control, reset boundaries and policy export checks; no training."""
from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path
import os
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from rl_go2.backends import LinkPointForces, make_backend
from rl_go2.config import load_experiment, load_scene
from rl_go2.control import JointPositionControl
from rl_go2.env import QuadrupedEnv
from rl_go2.evaluation.conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from rl_go2.state import RobotState, rotate_inverse
from rl_go2.tasks import TASK_TYPES

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    device = torch.device("cpu")
    num_envs = 2

    def __init__(self, config):
        self.default = torch.tensor(config["robot"]["default_joint_positions"])
        self.position = torch.zeros(2, 3)
        self.fall = False
        self.forces = []

    def reset(self, ids):
        self.position[ids] = torch.tensor([0., 0., 0.42])

    def set_base_pose(self, env_ids, position, quaternion):
        self.position[env_ids] = position

    def read_state(self):
        return RobotState(self.position.clone(), torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
                          torch.zeros(2, 3), torch.zeros(2, 3), self.default.repeat(2, 1), torch.zeros(2, 12))

    def step(self, torque, *, base_force=None):
        self.forces.append(None if base_force is None else base_force.clone())
        self.position[:, 0] += 0.01
        if self.fall:
            self.position[0, 2] = 0.01

    def close(self):
        pass


class ProbeTask:
    """Test-only task: different observation size, no velocity-command fields."""

    observation_terms = [("position_x", 1), ("phase", 1)]

    @staticmethod
    def validate_config(config, robot):
        pass

    def __init__(self, config, default_positions, num_envs, device, dt, seed):
        self.phase = torch.zeros(num_envs, device=device)

    def reset(self, env_ids, state, updates):
        self.phase[env_ids] = 0

    def observe(self, state, previous_action):
        return torch.stack((state.position[:, 0], self.phase), dim=-1)

    def reward(self, state, action, previous_action, torque):
        return torch.ones_like(self.phase)

    def terminated(self, state):
        return torch.zeros_like(self.phase, dtype=torch.bool)

    def on_step_end(self, state, episode_steps):
        self.phase.copy_(episode_steps)

    def context(self):
        return {"phase": self.phase}

    def external_force(self, state, episode_steps):
        return None


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment(ROOT / "configs/experiments/flat_ppo.yaml")

    def test_training_terrain_weights_and_partial_reset(self):
        config = load_experiment(ROOT / 'configs/experiments/terrain_supermimic_push.yaml')
        limits = config['robot']['joint_position_limits']
        self.assertEqual(limits['RR_hip_joint'], [-0.837758, 0.837758])
        self.assertEqual(limits['RL_hip_joint'], limits['RR_hip_joint'])
        self.assertEqual(limits['FR_hip_joint'], self.config['robot']['joint_position_limits']['FR_hip_joint'])
        patches = config['scene']['terrain_patches']
        weights = {}
        for patch in patches:
            weights[patch['type']] = weights.get(patch['type'], 0) + patch['weight']
        self.assertEqual(weights, dict.fromkeys(('flat', 'step', 'ramp', 'rough_ground'), 1.0))
        config['domain_randomization'] = {}
        env = QuadrupedEnv(config, backend=FakeBackend(config))
        env.reset()
        before = env.state.position[1].clone()
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.state.position[1], before)
        expected = env.terrain_positions[env.terrain_ids]
        torch.testing.assert_close(env.state.position, expected)
        env.reset(torch.tensor([], dtype=torch.long))

    def test_terrain_scan_matches_mujoco_ramp_and_step_surfaces(self):
        import mujoco
        import numpy as np
        from rl_go2.trainers.terrain_features import TerrainHeightScan

        for filename, x in (('ramp.yaml', 1.53), ('single_step.yaml', 1.33)):
            scene = load_scene(ROOT / 'configs/scenes' / filename)
            config = {**self.config, 'scene': scene, 'backend': 'mujoco', 'device': 'cpu', 'num_envs': 1}
            env = QuadrupedEnv(config)
            try:
                scan = TerrainHeightScan(scene)
                position = torch.tensor([[x, 0.0, 0.4]], requires_grad=True)
                values = scan(position, torch.tensor([[1., 0., 0., 0.]]))
                for index, offset in enumerate(scan.offsets):
                    origin = np.array([x + offset[0].item(), offset[1].item(), 1.0])
                    distance = mujoco.mj_ray(env.backend.model, env.backend.data, origin,
                                             np.array([0., 0., -1.]), None, True, -1,
                                             np.zeros(1, dtype=np.int32))
                    self.assertAlmostEqual(values[0, index].item(), 1 - distance - 0.4, places=5)
                values.sum().backward()
                self.assertTrue(torch.isfinite(position.grad).all())
            finally:
                env.close()

    def step_contact_config(self, backend):
        scene = load_scene(ROOT / 'configs/scenes/single_step.yaml')
        robot = next(entity for entity in scene['entities'] if entity.get('robot'))
        robot['position'] = [1.3, 0.0, 0.46]
        config = {**self.config, 'scene': scene, 'backend': backend, 'device': 'cpu', 'num_envs': 1}
        return NOMINAL_SAMPLE.condition('step-contact', NOMINAL_SLIDING, friction_multiplier=0.5).apply(config)

    def test_step_height_override_rejects_invalid_values(self):
        path = ROOT / 'configs/scenes/single_step.yaml'
        for height in (0, -0.01, float('inf'), True):
            with self.subTest(height=height), self.assertRaisesRegex(ValueError, 'Step step'):
                load_scene(path, entity_overrides={'step': {'height_m': height}})
        with self.assertRaisesRegex(ValueError, 'named terrain entities'):
            load_scene(path, entity_overrides={'ground': {'height_m': 0.05}})

    def test_rough_ground_seed_reproduces_geometry_and_bounded_height_changes(self):
        path = ROOT / 'configs/scenes/rough_ground.yaml'
        first = load_scene(path)
        second = load_scene(path, entity_overrides={'rough': {'seed': 102}})
        self.assertEqual(first, load_scene(path, entity_overrides={'rough': {'seed': 101}}))
        self.assertNotEqual(first['height_maps']['rough']['height_map_m'], second['height_maps']['rough']['height_map_m'])
        for scene in (first, second):
            metadata = scene['height_maps']['rough']
            heights = metadata['height_map_m']
            tiles = {entity['id']: entity for entity in scene['entities'] if entity['type'] == 'box'}
            self.assertEqual(len(heights), 5)
            for row, row_heights in enumerate(heights):
                self.assertEqual(len(row_heights), 4)
                for column, height in enumerate(row_heights):
                    self.assertIn(height, (0, 0.01, 0.02))
                    previous_height = heights[row - 1][column] if row else 0
                    self.assertLessEqual(abs(height - previous_height), 0.01)
                    if row == len(heights) - 1:
                        self.assertLessEqual(height, 0.01)
                    if column:
                        self.assertLessEqual(abs(height - row_heights[column - 1]), 0.01)
                    tile = tiles.pop(f'rough-{row}-{column}', None)
                    if height == 0:
                        self.assertIsNone(tile)
                    else:
                        self.assertEqual(tile['size'], [0.30, 0.30, height])
                        self.assertAlmostEqual(tile['position'][2] + tile['size'][2] / 2, height)
            self.assertFalse(tiles)

    def test_mujoco_rough_ground_matches_height_map(self):
        import mujoco
        import numpy as np

        scene = load_scene(ROOT / 'configs/scenes/rough_ground.yaml')
        config = {**self.config, 'scene': scene, 'backend': 'mujoco', 'device': 'cpu', 'num_envs': 1}
        env = QuadrupedEnv(NOMINAL_SAMPLE.condition('rough-contact', NOMINAL_SLIDING).apply(config))
        try:
            heights = scene['height_maps']['rough']['height_map_m']
            for row, row_heights in enumerate(heights):
                for column, height in enumerate(row_heights):
                    origin = np.array([0.8 + (row + 0.5) * 0.30, (column - 1.5) * 0.30, 1.0])
                    distance = mujoco.mj_ray(env.backend.model, env.backend.data, origin,
                                             np.array([0.0, 0.0, -1.0]), None, True, -1,
                                             np.zeros(1, dtype=np.int32))
                    self.assertAlmostEqual(1 - distance, height, places=7)
        finally:
            env.close()

    def test_ramp_slope_override_rejects_invalid_values(self):
        for slope in (0, -5, 90, float('nan')):
            with self.subTest(slope=slope), self.assertRaisesRegex(ValueError, 'Ramp ramp'):
                load_scene(ROOT / 'configs/scenes/ramp.yaml', entity_overrides={'ramp': {'slope_deg': slope}})

    def test_mujoco_ramp_surface_meets_ground_and_platform(self):
        import mujoco
        import numpy as np

        for slope in (5, 10):
            scene = load_scene(ROOT / 'configs/scenes/ramp.yaml', entity_overrides={'ramp': {'slope_deg': slope}})
            config = {**self.config, 'scene': scene, 'backend': 'mujoco', 'device': 'cpu', 'num_envs': 1}
            config = NOMINAL_SAMPLE.condition('ramp-contact', NOMINAL_SLIDING).apply(config)
            env = QuadrupedEnv(config)
            try:
                rise = 0.8 * math.tan(math.radians(slope))
                surface_samples = (
                    (0.99, 0), (1.01, rise / 80), (1.4, rise / 2), (1.799, rise * 0.799 / 0.8),
                    (1.801, rise), (2.399, rise), (2.401, rise * 0.799 / 0.8),
                    (2.8, rise / 2), (3.19, rise / 80), (3.21, 0),
                )
                for x, height in surface_samples:
                    with self.subTest(slope=slope, x=x):
                        distance = mujoco.mj_ray(
                            env.backend.model, env.backend.data, np.array([x, 0.0, 1.0]),
                            np.array([0.0, 0.0, -1.0]), None, True, -1, np.zeros(1, dtype=np.int32),
                        )
                        self.assertAlmostEqual(1 - distance, height, places=7)
            finally:
                env.close()

    @unittest.skipUnless(os.environ.get('RL_GO2_TEST_GENESIS') == '1', 'opt-in Genesis CPU kernel compilation')
    def test_genesis_foot_contacts_the_ramp_incline(self):
        scene = load_scene(ROOT / 'configs/scenes/ramp.yaml')
        robot = next(entity for entity in scene['entities'] if entity.get('robot'))
        robot['position'] = [1.4, 0, 0.42 + 0.4 * math.tan(math.radians(5))]
        config = {**self.config, 'scene': scene, 'backend': 'genesis', 'device': 'cpu', 'num_envs': 1}
        config = NOMINAL_SAMPLE.condition('ramp-contact', NOMINAL_SLIDING).apply(config)
        env = QuadrupedEnv(config)
        try:
            backend = env.backend
            *foot_ids, _ground_id, uphill_id, _deck_id, _downhill_id = backend.friction_geom_ids
            feet = torch.tensor(foot_ids, device=env.device)
            for _ in range(12):
                env.step(torch.zeros(1, 12))
            contacts = backend.robot.get_contacts(exclude_self_contact=True)
            a, b = contacts['geom_a'], contacts['geom_b']
            touching_ramp = ((a == uphill_id) & torch.isin(b, feet)) | ((b == uphill_id) & torch.isin(a, feet))
            self.assertTrue((touching_ramp & contacts['valid_mask']).any())
        finally:
            env.close()

    def test_mujoco_step_uses_randomized_foot_contact_parameters(self):
        env = QuadrupedEnv(self.step_contact_config('mujoco'))
        try:
            backend = env.backend
            step_id = backend.model.geom('step').id
            for _ in range(12):
                env.step(torch.zeros(1, 12))
            contacts = [contact for contact in backend.data.contact
                        if step_id in contact.geom and any(foot in contact.geom for foot in backend.foot_ids)]
            self.assertTrue(contacts)
            for contact in contacts:
                self.assertEqual(contact.dim, 6)
                torch.testing.assert_close(torch.tensor(contact.friction.tolist()),
                                           torch.tensor([0.4, 0.4, 0.01, 0.005, 0.005]))
        finally:
            env.close()

    def test_mujoco_viewer_retries_while_macos_releases_previous_window(self):
        from rl_go2.backends.mujoco import _launch_passive_viewer

        handle = object()
        busy = RuntimeError('another MuJoCo viewer is already open')
        with (
            patch('rl_go2.backends.mujoco.sys.platform', 'darwin'),
            patch('rl_go2.backends.mujoco.time.sleep') as sleep,
            patch('mujoco.viewer.launch_passive', side_effect=[busy, handle]) as launch,
        ):
            self.assertIs(_launch_passive_viewer(None, None), handle)
        self.assertEqual(launch.call_count, 2)
        sleep.assert_called_once()

    def test_mujoco_viewer_starts_lit_and_tracks_the_robot(self):
        import mujoco
        import numpy as np

        handle = Mock(cam=mujoco.MjvCamera())
        handle.lock.return_value = nullcontext()
        config = {**self.config, 'backend': 'mujoco', 'device': 'cpu', 'num_envs': 1}
        with patch('rl_go2.backends.mujoco._launch_passive_viewer', return_value=handle):
            env = QuadrupedEnv(config, viewer=True)
        try:
            backend = env.backend
            self.assertEqual(handle.cam.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
            self.assertEqual(handle.cam.trackbodyid, backend.base)
            np.testing.assert_allclose(handle.cam.lookat, backend.data.subtree_com[backend.base])
            np.testing.assert_allclose(backend.model.vis.headlight.ambient, [0.6, 0.6, 0.6])
            np.testing.assert_allclose(backend.model.geom('ground').rgba, [0.75, 0.78, 0.8, 1])
            handle.lock.assert_called_once()
        finally:
            env.close()

    @unittest.skipUnless(os.environ.get('RL_GO2_TEST_GENESIS') == '1', 'opt-in Genesis CPU kernel compilation')
    def test_genesis_step_is_in_randomized_foot_contact_surfaces(self):
        env = QuadrupedEnv(self.step_contact_config('genesis'))
        try:
            backend = env.backend
            *foot_ids, ground_id, step_id = backend.friction_geom_ids
            ratios = backend.scene.rigid_solver.get_geoms_friction_ratio([ground_id, step_id])
            torch.testing.assert_close(ratios, torch.full_like(ratios, 0.5))
            for _ in range(12):
                env.step(torch.zeros(1, 12))
            contacts = backend.robot.get_contacts(exclude_self_contact=True)
            foot_ids = torch.tensor(foot_ids, device=env.device)
            a, b = contacts['geom_a'], contacts['geom_b']
            touching_step = ((a == step_id) & torch.isin(b, foot_ids)) | ((b == step_id) & torch.isin(a, foot_ids))
            self.assertTrue((touching_step & contacts['valid_mask']).any())
        finally:
            env.close()

    def test_coordinate_rotation_and_torque_clipping(self):
        quat = torch.tensor([[2**-0.5, 0., 0., 2**-0.5]])
        actual = rotate_inverse(quat, torch.tensor([[1., 0., 0.]]))
        torch.testing.assert_close(actual, torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)
        controller = JointPositionControl(self.config["robot"], "cpu")
        torque = controller.torque(torch.full((2, 12), 100.), FakeBackend(self.config).read_state())
        torch.testing.assert_close(torque, controller.limits.expand(2, -1))

    def test_force_applies_each_substep_and_reset_after_inference(self):
        backend = FakeBackend(self.config)
        env = QuadrupedEnv(self.config, backend=backend, record_transitions=True)
        force = torch.tensor([[10., 0., 0.], [0., -10., 0.]])
        with torch.inference_mode():
            _, _, _, info = env.step(torch.zeros(2, 12), base_force=force)
        env.reset()
        self.assertEqual(len(backend.forces), 4)
        for actual in backend.forces:
            torch.testing.assert_close(actual, force)
        torch.testing.assert_close(info['transition'].context['external_force_world'], force)
        env.step(torch.zeros(2, 12))
        self.assertTrue(all(f is None for f in backend.forces[-4:]))
        with self.assertRaises(ValueError):
            env.step(torch.zeros(2, 12), base_force=torch.ones(3))

    def test_push_scheduler_duration_partial_reset_and_curriculum(self):
        from rl_go2.curriculum import make_curriculum
        config = load_experiment(ROOT / 'configs/experiments/push_ppo.yaml')
        config['curriculum'] = {'name': 'none'}
        config['task']['push'].update(interval_seconds=[.1, .1], duration_seconds=.04, force_range_n=[20, 20])
        env = QuadrupedEnv(config, backend=FakeBackend(config), record_transitions=True)
        task = env.task
        first = task.external_force(env.state, torch.tensor([5, 5])).clone()
        torch.testing.assert_close(torch.linalg.vector_norm(first, dim=-1), torch.full((2,), 20.))
        task.reset(torch.tensor([0]), env.state, {})
        second = task.external_force(env.state, torch.tensor([0, 6])).clone()
        torch.testing.assert_close(second[0], torch.zeros(3))
        torch.testing.assert_close(second[1], first[1])
        third = task.external_force(env.state, torch.tensor([1, 7]))
        torch.testing.assert_close(third, torch.zeros(2, 3))
        curriculum = make_curriculum({'name': 'push_force', 'stages': [
            {'after_steps': 0, 'force_range_n': [5, 10]}, {'after_steps': 100, 'force_range_n': [10, 20]}]})
        self.assertEqual(curriculum.on_reset(torch.tensor([0]), environment_steps=99, metrics={}), {'push_force_range': [5, 10]})
        self.assertEqual(curriculum.on_reset(torch.tensor([0]), environment_steps=100, metrics={}), {'push_force_range': [10, 20]})
        task.reset(torch.tensor([0]), env.state, {'push_force_range': [10, 20]})
        torch.testing.assert_close(task.force_ranges, torch.tensor([[10., 20.], [20., 20.]]))
        self.assertEqual(task.observe(env.state, env.previous_action).shape, (2, 45))

    def test_mujoco_external_force_impulse_and_clearing(self):
        import mujoco
        config = deepcopy(self.config)
        config.update(backend='mujoco', device='cpu', num_envs=1)
        config['scene']['sim_options']['gravity'] = [0, 0, 0]
        next(e for e in config['scene']['entities'] if e.get('robot'))['position'] = [0, 0, 2]
        backend = make_backend(config)
        try:
            backend.step(torch.zeros(1, 12), base_force=torch.tensor([[10., 0., 0.]]))
            mujoco.mj_subtreeVel(backend.model, backend.data)
            expected = 10 * config['scene']['sim_options']['dt'] / backend.model.body_subtreemass[backend.base]
            velocity = float(backend.data.subtree_linvel[backend.base, 0])
            self.assertAlmostEqual(velocity, expected, places=5)
            backend.step(torch.zeros(1, 12))
            self.assertFalse(backend.data.xfrc_applied.any())
            mujoco.mj_subtreeVel(backend.model, backend.data)
            self.assertAlmostEqual(float(backend.data.subtree_linvel[backend.base, 0]), velocity, places=5)
        finally:
            backend.close()

    def test_mujoco_point_force_matches_native_point_application(self):
        import mujoco
        import numpy as np

        config = deepcopy(self.config)
        config.update(backend="mujoco", device="cpu", num_envs=1)
        robot = next(entity for entity in config["scene"]["entities"] if entity.get("robot"))
        robot.update(position=[0, 0, 2], rotation=[2**-0.5, 0, 0, 2**-0.5])
        backend = make_backend(config)
        try:
            self.assertEqual(len(backend.link_names), 13)
            shape = (1, len(backend.link_names), 3)
            point_forces = LinkPointForces(torch.zeros(shape), torch.zeros(shape))
            calf = backend.link_names.index("FR_calf")
            point_forces.point_local_m[0, calf, 2] = -0.2
            point_forces.force_world_n[0, calf, 1] = 20
            body = backend.model.body("FR_calf").id
            reference = mujoco.MjData(backend.model)
            reference.qpos[:] = backend.data.qpos
            mujoco.mj_forward(backend.model, reference)
            point_world = reference.xpos[body] + (
                reference.xmat[body].reshape(3, 3) @ point_forces.point_local_m[0, calf].numpy()
            )
            mujoco.mj_applyFT(backend.model, reference, np.array([0., 20., 0.]), np.zeros(3),
                              point_world, body, reference.qfrc_applied)
            mujoco.mj_step(backend.model, reference)
            backend.step(torch.zeros(1, 12), link_point_forces=point_forces)
            np.testing.assert_allclose(backend.data.qvel, reference.qvel, atol=1e-10)
            self.assertTrue(backend.data.xfrc_applied[body, 3:].any())
            backend.step(torch.zeros(1, 12))
            self.assertFalse(backend.data.xfrc_applied.any())
            with self.assertRaisesRegex(ValueError, "point_local_m"):
                invalid = LinkPointForces(torch.zeros(1, 3), point_forces.force_world_n)
                backend.step(torch.zeros(1, 12), link_point_forces=invalid)
        finally:
            backend.close()

    @unittest.skipUnless(os.environ.get("RL_GO2_TEST_GENESIS") == "1", "opt-in Genesis CPU kernel compilation")
    def test_genesis_point_force_targets_each_environment_link(self):
        config = deepcopy(self.config)
        config.update(backend="genesis", device="cpu", num_envs=3)
        config["scene"]["sim_options"]["gravity"] = [0, 0, 0]
        robot = next(entity for entity in config["scene"]["entities"] if entity.get("robot"))
        robot.update(position=[0, 0, 2], rotation=[2**-0.5, 0, 0, 2**-0.5])
        backend = make_backend(config)
        try:
            self.assertEqual(len(backend.link_names), 13)
            shape = (3, len(backend.link_names), 3)
            point_forces = LinkPointForces(torch.zeros(shape), torch.zeros(shape))
            calf = backend.link_names.index("FR_calf")
            base = backend.link_names.index("base")
            point_forces.point_local_m[0, calf, 2] = -0.2
            point_forces.point_local_m[1, base, 1] = 0.2
            point_forces.force_world_n[0, calf, 1] = 20
            point_forces.force_world_n[1, base, 1] = 20
            backend.step(torch.zeros(3, 12), link_point_forces=point_forces)
            state = backend.read_state()
            torch.testing.assert_close(state.joint_velocity[2], torch.zeros(12), atol=1e-5, rtol=0)
            self.assertGreater(float(state.joint_velocity[0, 1:3].abs().max()), 0.01)
            self.assertFalse(torch.allclose(state.joint_velocity[0], state.joint_velocity[1]))
            # In compatibility mode Genesis exposes Cartesian velocity on the next step.
            backend.step(torch.zeros(3, 12))
            state = backend.read_state()
            self.assertLess(float(state.angular_velocity[1, 2]), -0.01)
            self.assertGreater(float(state.linear_velocity[1, 1]), 0.001)
            torch.testing.assert_close(state.angular_velocity[2], torch.zeros(3), atol=1e-5, rtol=0)
        finally:
            backend.close()

    def test_warm_start_keeps_fresh_optimizer_and_iteration(self):
        from rl_go2.trainers.ppo import make_runner, save_run_spec, initialize_weights
        config = deepcopy(self.config)
        config.update(backend='mujoco', device='cpu', num_envs=1)
        env = QuadrupedEnv(config)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'source'
                save_run_spec(config, output)
                source = make_runner(env)
                source.current_learning_iteration = 999
                parameter = next(source.alg.get_policy().parameters())
                source.alg.optimizer.state[parameter] = {
                    'step': torch.tensor(1.), 'exp_avg': torch.zeros_like(parameter),
                    'exp_avg_sq': torch.zeros_like(parameter)}
                source.save(str(output / 'model_999.pt'))
                target = make_runner(env)
                initialize_weights(target, output / 'model_999.pt', config)
                self.assertEqual(target.current_learning_iteration, 0)
                self.assertFalse(target.alg.optimizer.state)
                for key, value in source.alg.get_policy().state_dict().items():
                    torch.testing.assert_close(value, target.alg.get_policy().state_dict()[key])
        finally:
            env.close()

    def test_config_selects_a_different_task_without_walk_assumptions(self):
        from rl_go2.trainers.ppo import make_runner, save_run_spec

        with patch.dict(TASK_TYPES, {"probe": f"{__name__}:ProbeTask"}):
            config = deepcopy(self.config)
            config["task"] = {"name": "probe", "episode_seconds": 20}
            env = QuadrupedEnv(config, backend=FakeBackend(config), record_transitions=True)
            self.assertIsInstance(env.task, ProbeTask)
            observation, reward, done, info = env.step(torch.zeros(2, 12))
            self.assertEqual(observation.shape, (2, 2))
            torch.testing.assert_close(reward, torch.ones(2))
            self.assertFalse(done.any())
            torch.testing.assert_close(info["transition"].context["phase"], torch.zeros(2))
            env.step(torch.zeros(2, 12))
            torch.testing.assert_close(info["transition"].context["phase"], torch.zeros(2))
            runner = make_runner(env)
            with torch.inference_mode():
                self.assertEqual(runner.get_inference_policy()(runner.env.get_observations()).shape, (2, 12))
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "probe"
                save_run_spec(config, output)
                metadata = json.loads((output / "policy_spec.json").read_text())
                self.assertEqual(metadata["task"], "probe")
                self.assertEqual(metadata["observation_terms"], [["position_x", 1], ["phase", 1]])

    def test_task_validation_runs_before_backend_creation(self):
        config = deepcopy(self.config)
        config["task"]["name"] = "not_implemented"
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            QuadrupedEnv(config)
        config["task"]["name"] = "flat_walk"
        config["task"]["tracking_sigma"] = 0
        with self.assertRaisesRegex(ValueError, "tracking_sigma"):
            QuadrupedEnv(config)

    def test_timeouts_keep_pre_reset_transition(self):
        self.config["task"]["episode_seconds"] = 0.02
        env = QuadrupedEnv(self.config, backend=FakeBackend(self.config), record_transitions=True)
        before = env.observe().clone()
        observation, _, done, info = env.step(torch.ones(2, 12))
        self.assertTrue(done.all())
        self.assertTrue(info["truncated"].all())
        self.assertFalse(info["terminated"].any())
        transition = info["transition"]
        torch.testing.assert_close(transition.next_state.position[:, 0], torch.full((2,), 0.04))
        torch.testing.assert_close(env.state.position[:, 0], torch.zeros(2))
        torch.testing.assert_close(info["final_observation"][:, -12:], torch.ones(2, 12))
        torch.testing.assert_close(observation[:, -12:], torch.zeros(2, 12))
        self.assertEqual(before.shape, (2, 45))
        env.step(torch.zeros(2, 12))
        torch.testing.assert_close(transition.next_state.position[:, 0], torch.full((2,), 0.04))

    def test_fall_resets_only_failed_environment(self):
        backend = FakeBackend(self.config)
        env = QuadrupedEnv(self.config, backend=backend)
        backend.fall = True
        _, _, done, info = env.step(torch.zeros(2, 12))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(info["terminated"].tolist(), [True, False])
        self.assertFalse(info["truncated"].any())
        self.assertEqual(env.episode_length_buf.tolist(), [0, 1])
        torch.testing.assert_close(env.state.position[:, 0], torch.tensor([0., 0.04]))

    def test_reserved_features_fail_explicitly(self):
        config = deepcopy(self.config)
        config["backend"] = "isaac"
        with self.assertRaises(NotImplementedError):
            make_backend(config)
        config["curriculum"] = {"name": "adaptive"}
        with self.assertRaises(NotImplementedError):
            QuadrupedEnv(config, backend=FakeBackend(config))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("schema: rl-go2.scene/v1\nscene:\n  sim_options: {dt: 0.005, gravity: [0,0,-9.81]}\n  entities:\n    - {id: unsupported, type: mesh}\n")
            with self.assertRaises(NotImplementedError):
                load_scene(path)

    def test_joint_target_limits_and_legacy_config(self):
        config = deepcopy(self.config)
        config.update(backend="mujoco", device="cpu", num_envs=1)
        config['robot']['joint_position_limits']['RR_hip_joint'] = [-0.837758, 0.837758]
        env = QuadrupedEnv(config)
        try:
            controller = env.controller
            for sign in (-1, 1):
                target, applied = controller.target(torch.full((1, 12), sign * 100.))
                torch.testing.assert_close(target, controller.default + config['robot']['action_scale'] * applied)
                for i, name in enumerate(config['robot']['joint_names']):
                    physical = env.backend.model.jnt_range[env.backend.model.joint(name).id]
                    self.assertEqual(list(physical), config['robot']['joint_position_limits'][name])
                    self.assertGreaterEqual(float(target[0, i]), physical[0] + .05 - 1e-6)
                    self.assertLessEqual(float(target[0, i]), physical[1] - .05 + 1e-6)
            action = torch.zeros(1, 12, requires_grad=True)
            controller.target(action)[0].sum().backward()
            torch.testing.assert_close(action.grad, torch.full_like(action, .25))
            legacy = deepcopy(config['robot'])
            legacy.pop('joint_position_limits')
            legacy.pop('joint_limit_margin')
            _, applied = JointPositionControl(legacy, 'cpu').target(torch.full((1, 12), 100.))
            torch.testing.assert_close(applied, torch.full((1, 12), 5.))
            bad = deepcopy(config['robot'])
            bad['joint_position_limits']['FR_calf_joint'] = [0., 1.]
            with self.assertRaisesRegex(ValueError, 'Default joint positions'):
                JointPositionControl(bad, 'cpu')
            bad = deepcopy(config['robot'])
            bad['reset_joint_positions'][6] = -0.9
            with self.assertRaisesRegex(ValueError, 'Reset joint positions.*RR_hip_joint'):
                JointPositionControl(bad, 'cpu')
        finally:
            env.close()

    def test_mujoco_joint_mapping_and_ppo_export(self):
        from rl_go2.trainers.ppo import make_runner, save_run_spec
        config = deepcopy(self.config)
        config.update(backend="mujoco", device="cpu", num_envs=1)
        env = QuadrupedEnv(config)
        try:
            torch.testing.assert_close(env.state.position[0], torch.tensor([0., 0., 0.42]))
            torch.testing.assert_close(env.state.joint_position[0],
                                       torch.tensor(config["robot"]["reset_joint_positions"]))
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "run"
                save_run_spec(config, output)
                runner = make_runner(env)
                with torch.inference_mode():
                    expected = runner.get_inference_policy()(runner.env.get_observations())
                    runner.export_policy_to_jit(str(output))
                    policy = torch.jit.load(str(output / "policy.pt")).eval()
                    torch.testing.assert_close(policy(env.observe()), expected)
                    for _ in range(8):
                        obs, reward, _, _ = env.step(policy(env.observe()))
                        self.assertTrue(torch.isfinite(obs).all() and torch.isfinite(reward).all())
                self.assertTrue((output / "config.json").is_file())
                self.assertTrue((output / "policy_spec.json").is_file())
        finally:
            env.close()

    @unittest.skipUnless(os.environ.get("RL_GO2_TEST_GENESIS") == "1", "opt-in Genesis CPU kernel compilation")
    def test_genesis_absolute_pose_partial_reset_and_timeout(self):
        config = deepcopy(self.config)
        config.update(backend="genesis", device="cpu", num_envs=2)
        config['robot']['joint_position_limits']['RR_hip_joint'] = [-0.837758, 0.837758]
        config["task"]["episode_seconds"] = 0.04
        config["domain_randomization"] = {"foot_ground_friction": {
            "enabled": True, "env_multiplier_range": [0.75, 1.25], "switch_interval_seconds": [0.04, 0.04],
            "base_ranges": {foot: {"sliding": [0.8, 0.8], "torsional": [0.02, 0.02], "rolling": [0.01, 0.01]}
                            for foot in ("FR", "FL", "RR", "RL")}}}
        env = QuadrupedEnv(config, record_transitions=True)
        try:
            backend = env.backend
            lower, upper = backend.robot.get_dofs_limit(backend.joints)
            limits = torch.tensor([config['robot']['joint_position_limits'][name]
                                   for name in config['robot']['joint_names']])
            torch.testing.assert_close(lower, limits[:, 0])
            torch.testing.assert_close(upper, limits[:, 1])
            torch.testing.assert_close(env.state.position, torch.tensor([[0., 0., 0.42]]).expand(2, -1))
            torch.testing.assert_close(env.state.quaternion, torch.tensor([[1., 0., 0., 0.]]).expand(2, -1))
            expected = torch.tensor(env.config["robot"]["reset_joint_positions"], device=env.device)
            torch.testing.assert_close(env.state.joint_position, expected.expand(2, -1))
            force = torch.tensor([[10., 0., 0.], [0., 0., 0.]])
            env.friction_steps_left[0] = 1
            friction = env.foot_friction.clone()
            env.step(torch.zeros(2, 12), base_force=force)
            self.assertFalse(torch.equal(env.foot_friction[0], friction[0]))
            torch.testing.assert_close(env.foot_friction[1], friction[1])
            ratios = backend.scene.rigid_solver.get_geoms_friction_ratio(backend.friction_geom_ids)
            torch.testing.assert_close(ratios, (env.foot_friction[:, :1] / 0.8).expand_as(ratios))
            self.assertGreater(float(env.state.linear_velocity[0, 0]), float(env.state.linear_velocity[1, 0]))
            other = env.state.clone()
            env.reset(torch.tensor([0]))
            torch.testing.assert_close(env.state.position[1], other.position[1])
            torch.testing.assert_close(env.state.joint_velocity[1], other.joint_velocity[1])
            torch.testing.assert_close(env.state.position[0], torch.tensor([0., 0., 0.42]))
            env.reset()
            env.step(torch.zeros(2, 12))
            _, _, done, extras = env.step(torch.zeros(2, 12))
            self.assertTrue(done.all() and extras["truncated"].all())
            torch.testing.assert_close(env.state.position, torch.tensor([[0., 0., 0.42]]).expand(2, -1))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
