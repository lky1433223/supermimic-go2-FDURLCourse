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

    def reset(self, ids):
        self.position[ids] = torch.tensor([0., 0., 0.42])

    def read_state(self):
        return RobotState(self.position.clone(), torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
                          torch.zeros(2, 3), torch.zeros(2, 3), self.default.repeat(2, 1), torch.zeros(2, 12))

    def step(self, torque):
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
        from rl_go2.trainers.supermimic import train
        with self.assertRaises(NotImplementedError):
            train(config, "/unused")
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
            env.step(torch.zeros(2, 12))
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
