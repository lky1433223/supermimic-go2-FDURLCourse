from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from rl_go2.state import RobotState


class MujocoBackend:
    """Single-environment CPU evaluation; engine indices never escape this adapter."""

    def __init__(self, config, viewer=False):
        if config["num_envs"] != 1 or config["device"] != "cpu":
            raise ValueError("MuJoCo validation currently requires num_envs=1 and device=cpu")
        self.num_envs, self.device = 1, torch.device("cpu")
        self.config = config
        robot = next(e for e in config["scene"]["entities"] if e.get("robot"))
        path = Path(robot["file"])
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str(path.parent / compiler.get("meshdir", "")))
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        sim = config["scene"]["sim_options"]
        option.set("timestep", str(sim["dt"]))
        option.set("gravity", " ".join(map(str, sim["gravity"])))
        world = root.find("worldbody")
        for entity in config["scene"]["entities"]:
            if entity.get("robot"):
                continue
            kind = entity["type"]
            size = [0, 0, 0.1] if kind == "plane" else [s/2 for s in entity["size"]]
            ET.SubElement(world, "geom", name=entity["id"], type=kind,
                          pos=" ".join(map(str, entity["position"])),
                          quat=" ".join(map(str, entity["rotation"])),
                          size=" ".join(map(str, size)), friction=f"{entity.get('friction', 0.8)} 0.005 0.0001")
        # Keep model-authored collision masks/self-collision rules and motor parameters.
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        names = config["robot"]["joint_names"]
        joints = [self.model.joint(name).id for name in names]
        self.q_indices = self.model.jnt_qposadr[joints]
        self.v_indices = self.model.jnt_dofadr[joints]
        self.base = self.model.body(config["robot"]["base_link"]).id
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Robot model must start with a floating-base joint")
        self.motors = []
        for joint in joints:
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError("Expected one direct torque motor for each canonical joint")
            self.motors.append(int(matches[0]))
        if not np.allclose(self.model.actuator_gear[self.motors, 0], 1):
            raise ValueError("Only unit-gear direct torque motors are supported")
        self.viewer = None
        self.reset(torch.tensor([0]))
        if viewer:
            from mujoco import viewer as viewer_module
            self.viewer = viewer_module.launch_passive(self.model, self.data)

    def read_state(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
        # mj_objectVelocity is at the body's COM. Shift to the authored body origin.
        linear = velocity[3:] + np.cross(velocity[:3], self.data.xpos[self.base] - self.data.xipos[self.base])
        def tensor(array):
            return torch.tensor(np.array(array, copy=True), dtype=torch.float32).unsqueeze(0)
        return RobotState(tensor(self.data.xpos[self.base]), tensor(self.data.xquat[self.base]),
                          tensor(linear), tensor(velocity[:3]), tensor(self.data.qpos[self.q_indices]),
                          tensor(self.data.qvel[self.v_indices]))

    def reset(self, env_ids):
        if env_ids.numel() == 0:
            return
        mujoco.mj_resetData(self.model, self.data)
        robot = next(e for e in self.config["scene"]["entities"] if e.get("robot"))
        self.data.qpos[:3] = robot["position"]
        self.data.qpos[3:7] = robot["rotation"]
        self.data.qpos[self.q_indices] = self.config["robot"]["default_joint_positions"]
        mujoco.mj_forward(self.model, self.data)

    def step(self, torque, *, base_force=None):
        self.data.xfrc_applied[:] = 0
        if base_force is not None:
            if base_force.shape != (1, 3) or not torch.isfinite(base_force).all():
                raise ValueError("base_force must have shape (1, 3) and be finite")
            self.data.xfrc_applied[self.base, :3] = base_force[0].detach().cpu().numpy()
        self.data.ctrl[self.motors] = torque[0].detach().cpu().numpy()
        mujoco.mj_step(self.model, self.data)
        # mj_step advances qpos after computing derived kinematics.
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
