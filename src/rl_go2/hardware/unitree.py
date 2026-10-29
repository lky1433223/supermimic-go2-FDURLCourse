"""Go2 SDK2 message adapters and explicit read-only DDS connection.

LowState supplies joint state and IMU, NOT simulated base position/linear velocity.
The actor uses only these measured fields. Do not fabricate a full RobotState.
"""

from dataclasses import dataclass
import math

import numpy as np

# Official SDK example/go2/low_level/unitree_legged_const.py: motors 0..11.
SDK_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                        for joint in ("hip", "thigh", "calf"))
POSITION_STOP = 2.146e9
VELOCITY_STOP = 16000.0


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


def _sdk_indices(joint_names):
    if len(joint_names) != 12 or set(joint_names) != set(SDK_JOINT_NAMES):
        raise ValueError("Expected exactly the twelve Go2 joint names")
    return [SDK_JOINT_NAMES.index(name) for name in joint_names]


@dataclass
class Go2Telemetry:
    quaternion: np.ndarray  # wxyz, body-to-world IMU orientation
    angular_velocity_body: np.ndarray  # gyroscope is already in the body frame
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    tick: int

    def observation(self, commands, previous_action, robot_config, task_config):
        """CPU batch of size one; uses the exact same observation code as training."""
        import torch
        from rl_go2.observations import policy_observation

        def batch(value, size, name):
            return torch.from_numpy(_vector(value, size, name)).unsqueeze(0)

        return policy_observation(
            batch(self.quaternion, 4, "quaternion"), batch(self.angular_velocity_body, 3, "gyroscope"),
            batch(self.joint_position, 12, "joint position"), batch(self.joint_velocity, 12, "joint velocity"),
            batch(commands, 3, "commands"), batch(previous_action, 12, "previous action"),
            batch(robot_config["default_joint_positions"], 12, "default position"),
            task_config["observation_scales"],
        )


def decode_low_state(message, joint_names):
    indices = _sdk_indices(joint_names)
    if len(message.motor_state) < 12:
        raise ValueError("LowState has fewer than twelve motor states")
    quaternion = _vector(message.imu_state.quaternion, 4, "IMU quaternion")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, abs_tol=0.01):
        raise ValueError("IMU quaternion must be a unit wxyz quaternion")
    return Go2Telemetry(
        quaternion / norm, _vector(message.imu_state.gyroscope, 3, "gyroscope"),
        _vector([message.motor_state[i].q for i in indices], 12, "motor position"),
        _vector([message.motor_state[i].dq for i in indices], 12, "motor velocity"), int(message.tick),
    )


def position_command(targets, robot_config):
    """Build an official SDK LowCmd with CRC; does not initialize DDS or publish.

    This encodes onboard PD position targets, not an external torque-control loop.
    Deployment timing, handover and stop handling must be implemented before sending.
    """
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    indices = _sdk_indices(robot_config["joint_names"])
    targets = _vector(targets, 12, "target position")
    kp, kd = float(robot_config["kp"]), float(robot_config["kd"])
    if not all(math.isfinite(gain) and gain >= 0 for gain in (kp, kd)):
        raise ValueError("PD gains must be finite and nonnegative")
    command = unitree_go_msg_dds__LowCmd_()
    command.head = [0xFE, 0xEF]
    command.level_flag = 0xFF
    command.gpio = 0
    for motor in command.motor_cmd:
        motor.mode = 0x01
        motor.q, motor.dq = POSITION_STOP, VELOCITY_STOP
        motor.kp, motor.kd, motor.tau = 0.0, 0.0, 0.0
    for index, target in zip(indices, targets):
        motor = command.motor_cmd[index]
        motor.q, motor.dq = float(target), 0.0
        motor.kp, motor.kd = kp, kd
    command.crc = CRC().Crc(command)
    return command


def torque_command(torques, robot_config):
    """Encode clipped external-PD torques with onboard PD disabled; never publishes."""
    from unitree_sdk2py.utils.crc import CRC

    indices = _sdk_indices(robot_config["joint_names"])
    torques = _vector(torques, 12, "torque")
    limits = _vector(robot_config["torque_limits"], 12, "torque limits")
    if (limits <= 0).any():
        raise ValueError("Torque limits must be positive")
    command = position_command(robot_config["default_joint_positions"], robot_config)
    for index, torque in zip(indices, np.clip(torques, -limits, limits)):
        motor = command.motor_cmd[index]
        motor.q, motor.dq = POSITION_STOP, VELOCITY_STOP
        motor.kp, motor.kd, motor.tau = 0.0, 0.0, float(torque)
    command.crc = CRC().Crc(command)
    return command


class UnitreeStateReader:
    """Explicit connection only. This reader never creates a command publisher."""

    def __init__(self, config, joint_names):
        _sdk_indices(joint_names)
        interface = config["interface"]
        if not isinstance(interface, str) or not interface.strip():
            raise ValueError("Set hardware.interface to the actual robot-facing Ethernet interface")
        self.timeout = float(config["read_timeout_seconds"])
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("read_timeout_seconds must be finite and positive")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

        ChannelFactoryInitialize(config["domain_id"], interface)
        self.subscriber = ChannelSubscriber(config["lowstate_topic"], LowState_)
        self.subscriber.Init()
        self.joint_names = tuple(joint_names)

    def read(self):
        message = self.subscriber.Read(self.timeout)
        if message is None:
            raise TimeoutError("No Go2 LowState received within the configured timeout")
        from unitree_sdk2py.utils.crc import CRC
        if list(message.head) != [0xFE, 0xEF] or message.crc != CRC().Crc(message):
            raise ValueError("Invalid Go2 LowState header or CRC")
        return decode_low_state(message, self.joint_names)

    def close(self):
        self.subscriber.Close()
