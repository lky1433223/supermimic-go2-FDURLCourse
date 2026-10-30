"""Control, reset boundaries and policy export checks; no training."""
from copy import deepcopy
from pathlib import Path
import os
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from rl_go2.backends import make_backend
from rl_go2.config import load_experiment, load_scene
from rl_go2.control import JointPositionControl
from rl_go2.env import QuadrupedEnv
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

    def test_mujoco_joint_mapping_and_ppo_export(self):
        from rl_go2.trainers.ppo import make_runner, save_run_spec
        config = deepcopy(self.config)
        config.update(backend="mujoco", device="cpu", num_envs=1)
        env = QuadrupedEnv(config)
        try:
            torch.testing.assert_close(env.state.position[0], torch.tensor([0., 0., 0.42]))
            torch.testing.assert_close(env.state.joint_position[0], env.controller.default)
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
        config["task"]["episode_seconds"] = 0.04
        env = QuadrupedEnv(config, record_transitions=True)
        try:
            torch.testing.assert_close(env.state.position, torch.tensor([[0., 0., 0.42]]).expand(2, -1))
            torch.testing.assert_close(env.state.quaternion, torch.tensor([[1., 0., 0., 0.]]).expand(2, -1))
            torch.testing.assert_close(env.state.joint_position, env.controller.default.expand(2, -1))
            force = torch.tensor([[10., 0., 0.], [0., 0., 0.]])
            env.step(torch.zeros(2, 12), base_force=force)
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
