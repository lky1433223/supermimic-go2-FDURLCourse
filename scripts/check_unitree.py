"""Offline by default. --interface explicitly opens a read-only state subscription."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from rl_go2.config import load_experiment, read_yaml
from rl_go2.hardware.unitree import UnitreeStateReader, decode_low_state, position_command

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", help="Open a read-only DDS subscription on this interface")
    args = parser.parse_args()
    config = load_experiment(ROOT / "configs/experiments/flat_ppo.yaml")
    if args.interface:
        hardware = read_yaml(ROOT / "configs/hardware/go2.yaml")
        hardware["interface"] = args.interface
        reader = UnitreeStateReader(hardware, config["robot"]["joint_names"])
        try:
            telemetry = reader.read()
            print(json.dumps({"tick": telemetry.tick, "joint_position": telemetry.joint_position.tolist(),
                              "quaternion_wxyz": telemetry.quaternion.tolist()}, indent=2))
        finally:
            reader.close()
    else:
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
        from unitree_sdk2py.utils.crc import CRC

        message = unitree_go_msg_dds__LowState_()
        message.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        telemetry = decode_low_state(message, config["robot"]["joint_names"])
        observation = telemetry.observation([0, 0, 0], [0]*12, config["robot"], config["task"])
        command = position_command(config["robot"]["default_joint_positions"], config["robot"])
        assert command.crc == CRC().Crc(command)
        assert observation.shape == (1, 45)
        assert command.serialize()
        print(json.dumps({"sdk": version("unitree-sdk2py"), "cyclonedds": version("cyclonedds"),
                          "observation_shape": list(observation.shape), "lowcmd_crc": command.crc,
                          "dds_initialized": False, "commands_sent": 0}, indent=2))


if __name__ == "__main__":
    main()
