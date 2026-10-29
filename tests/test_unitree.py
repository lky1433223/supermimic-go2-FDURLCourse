"""Offline SDK protocol checks. No DDS participant or robot connection is created."""

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl_go2.config import load_experiment
from rl_go2.hardware.unitree import (
    POSITION_STOP, VELOCITY_STOP, SDK_JOINT_NAMES, UnitreeStateReader,
    decode_low_state, position_command, torque_command,
)
from rl_go2.state import RobotState
from rl_go2.tasks import FlatWalkTask

ROOT = Path(__file__).resolve().parents[1]
HAS_SDK = find_spec("unitree_sdk2py") is not None


class UnitreeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment(ROOT / "configs/experiments/flat_ppo.yaml")

    def test_motor_order_and_body_frame_observation_match_sim(self):
        # Reverse policy joint order so identity motor mapping cannot accidentally pass.
        robot = deepcopy(self.config["robot"])
        robot["joint_names"] = list(reversed(SDK_JOINT_NAMES))
        robot["default_joint_positions"] = list(reversed(robot["default_joint_positions"]))
        message = SimpleNamespace(
            imu_state=SimpleNamespace(quaternion=[2**-0.5, 0., 0., 2**-0.5], gyroscope=[1., 2., 3.]),
            motor_state=[SimpleNamespace(q=i/10, dq=i/20) for i in range(20)], tick=42,
        )
        telemetry = decode_low_state(message, robot["joint_names"])
        np.testing.assert_allclose(telemetry.joint_position, np.arange(11, -1, -1)/10)
        np.testing.assert_allclose(telemetry.joint_velocity, np.arange(11, -1, -1)/20)
        default = torch.tensor(robot["default_joint_positions"])
        task = FlatWalkTask(self.config["task"], default, 1, "cpu", 0.02, 1)
        task.commands[:] = torch.tensor([0.5, 0.1, 0.2])
        action = torch.full((1, 12), 0.25)
        # A +90 degree yaw rotates body [1,2,3] angular velocity into world [-2,1,3].
        state = RobotState(torch.zeros(1, 3), torch.from_numpy(telemetry.quaternion).unsqueeze(0),
                           torch.zeros(1, 3), torch.tensor([[-2., 1., 3.]]),
                           torch.from_numpy(telemetry.joint_position).unsqueeze(0),
                           torch.from_numpy(telemetry.joint_velocity).unsqueeze(0))
        expected = task.observe(state, action)
        actual = telemetry.observation(task.commands[0].numpy(), action[0].numpy(), robot, self.config["task"])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (1, 45))

    def test_bad_telemetry_and_missing_interface_fail(self):
        message = SimpleNamespace(imu_state=SimpleNamespace(quaternion=[0., 0., 0., 0.], gyroscope=[0]*3),
                                  motor_state=[SimpleNamespace(q=0., dq=0.) for _ in range(20)], tick=0)
        with self.assertRaises(ValueError):
            decode_low_state(message, SDK_JOINT_NAMES)
        with self.assertRaises(ValueError):
            decode_low_state(message, [SDK_JOINT_NAMES[0]]*12)
        with self.assertRaises(ValueError):
            UnitreeStateReader({"interface": None}, SDK_JOINT_NAMES)

    @unittest.skipUnless(HAS_SDK, "install the optional unitree extra for real SDK message checks")
    def test_official_commands_mapping_crc_and_serialization(self):
        from unitree_sdk2py.utils.crc import CRC
        robot = deepcopy(self.config["robot"])
        robot["joint_names"] = list(reversed(SDK_JOINT_NAMES))
        targets = np.arange(12, dtype=float)/10
        command = position_command(targets, robot)
        self.assertEqual(command.head, [0xFE, 0xEF])
        self.assertEqual(command.level_flag, 0xFF)
        for i in range(12):
            self.assertAlmostEqual(command.motor_cmd[11-i].q, targets[i], places=6)
        for motor in command.motor_cmd[12:]:
            self.assertEqual((motor.q, motor.dq), (POSITION_STOP, VELOCITY_STOP))
            self.assertEqual((motor.kp, motor.kd, motor.tau), (0, 0, 0))
        self.assertEqual(command.crc, CRC().Crc(command))
        self.assertTrue(command.serialize())
        command = torque_command(np.full(12, -1000.), robot)
        for i, limit in enumerate(robot["torque_limits"]):
            motor = command.motor_cmd[11-i]
            self.assertEqual((motor.kp, motor.kd), (0, 0))
            self.assertAlmostEqual(motor.tau, -limit, places=5)
        self.assertEqual(command.crc, CRC().Crc(command))
        self.assertTrue(command.serialize())
        with self.assertRaises(ValueError):
            position_command([float("nan")]*12, robot)

    @unittest.skipUnless(HAS_SDK, "install unitree extra for SDK transport checks")
    def test_read_only_transport_timeout_crc_and_close_without_network(self):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
        from unitree_sdk2py.utils.crc import CRC
        message = unitree_go_msg_dds__LowState_()
        message.head = [0xFE, 0xEF]
        message.imu_state.quaternion = [1., 0., 0., 0.]
        message.crc = CRC().Crc(message)
        config = {"interface": "test0", "domain_id": 1, "read_timeout_seconds": 0.1, "lowstate_topic": "rt/lowstate"}
        with patch("unitree_sdk2py.core.channel.ChannelFactoryInitialize") as initialize, \
                patch("unitree_sdk2py.core.channel.ChannelSubscriber") as subscriber:
            reader = UnitreeStateReader(config, SDK_JOINT_NAMES)
            initialize.assert_called_once_with(1, "test0")
            subscriber.return_value.Read.return_value = message
            self.assertEqual(reader.read().tick, 0)
            message.crc ^= 1
            with self.assertRaises(ValueError):
                reader.read()
            subscriber.return_value.Read.return_value = None
            with self.assertRaises(TimeoutError):
                reader.read()
            reader.close()
            subscriber.return_value.Close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
