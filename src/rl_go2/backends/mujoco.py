from collections.abc import Callable
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from rl_go2.backends import LinkPointForces
from rl_go2.config import FOOT_NAMES, foot_friction_ranges
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
        if foot_friction_ranges(config) is not None:
            self.foot_ids = [self.model.geom(name).id for name in FOOT_NAMES]
            surfaces = [self.model.geom(entity["id"]).id for entity in config["scene"]["entities"]
                        if not entity.get("robot")]
            has_explicit_pair = any(
                (a in surfaces and b in self.foot_ids) or (b in surfaces and a in self.foot_ids)
                for a, b in zip(self.model.pair_geom1, self.model.pair_geom2)
            )
            if (not surfaces or np.any(self.model.geom_priority[self.foot_ids, None] <= self.model.geom_priority[surfaces])
                    or np.any(self.model.geom_condim[self.foot_ids] != 6)
                    or has_explicit_pair):
                raise ValueError("Foot friction requires implicit surface contacts with higher-priority condim=6 feet")
        names = config["robot"]["joint_names"]
        joints = [self.model.joint(name).id for name in names]
        self.q_indices = self.model.jnt_qposadr[joints]
        self.v_indices = self.model.jnt_dofadr[joints]
        self.base = self.model.body(config["robot"]["base_link"]).id
        links = sorted((body.get("name"), self.model.body(body.get("name")).id)
                       for body in world.iter("body") if body.get("name") is not None)
        self.link_names = tuple(name for name, _ in links)
        self._link_indices = np.array([index for _, index in links], dtype=np.intp)
        self.nominal_base_com = self.model.body_ipos[self.base].copy()
        self.nominal_base_mass = self.model.body_mass[self.base]
        self.nominal_base_inertia = self.model.body_inertia[self.base].copy()
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
        self._interactive_force_callback: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None
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

    def set_foot_ground_friction(self, base, multipliers):
        self.foot_friction_base = base
        self.model.geom_friction[self.foot_ids] = (base * multipliers[0]).numpy()

    def set_foot_friction_multipliers(self, multipliers, env_ids):
        self.model.geom_friction[self.foot_ids] = (self.foot_friction_base * multipliers[0]).numpy()

    def set_base_com_offset(self, offsets):
        self.model.body_ipos[self.base] = self.nominal_base_com + offsets[0].numpy()
        mujoco.mj_setConst(self.model, self.data)

    def set_base_mass_scale(self, scales):
        scale = scales.item()
        mass = self.nominal_base_mass * scale
        if not np.isfinite(mass) or not 0 < mass <= np.finfo(np.float32).max:
            raise ValueError("Sampled base mass must be finite and positive in float32")
        self.model.body_mass[self.base] = mass
        self.model.body_inertia[self.base] = self.nominal_base_inertia * scale
        mujoco.mj_setConst(self.model, self.data)
        return scales.new_full((1, 1), mass)

    def reset(self, env_ids):
        if env_ids.numel() == 0:
            return
        mujoco.mj_resetData(self.model, self.data)
        robot = next(e for e in self.config["scene"]["entities"] if e.get("robot"))
        self.data.qpos[:3] = robot["position"]
        self.data.qpos[3:7] = robot["rotation"]
        spec = self.config["robot"]
        self.data.qpos[self.q_indices] = spec.get("reset_joint_positions", spec["default_joint_positions"])
        mujoco.mj_forward(self.model, self.data)

    def _apply_link_point_forces(self, point_forces: LinkPointForces) -> None:
        links = self._link_indices
        rotations = self.data.xmat[links].reshape(-1, 3, 3)
        local_points = point_forces.point_local_m[0].detach().to(device="cpu", dtype=torch.float64).numpy()
        forces = point_forces.force_world_n[0].detach().to(device="cpu", dtype=torch.float64).numpy()
        point_world = self.data.xpos[links] + np.einsum("lij,lj->li", rotations, local_points)
        self.data.xfrc_applied[links, :3] += forces
        # MuJoCo applies each Cartesian wrench at the body's center of mass.
        self.data.xfrc_applied[links, 3:] += np.cross(point_world - self.data.xipos[links], forces)

    def read_link_quaternions(self) -> torch.Tensor:
        """Return current authored link orientations in link_names order (world wxyz)."""
        return torch.from_numpy(self.data.xquat[self._link_indices]).float().unsqueeze(0)

    def set_interactive_force_callback(
        self, callback: Callable[[mujoco.MjModel, mujoco.MjData], None] | None,
    ) -> None:
        """Register a force writer called after clearing forces on each physics step.

        The callback writes world-frame forces and COM torques into xfrc_applied.
        Task forces and link-point forces are added afterwards. Pass None to detach.
        """
        self._interactive_force_callback = callback

    def step(
        self,
        torque: torch.Tensor,
        *,
        base_force: torch.Tensor | None = None,
        link_point_forces: LinkPointForces | None = None,
    ) -> None:
        """Advance one physics step, applying the supplied forces for that step."""
        if link_point_forces is not None:
            link_point_forces.validate(self.num_envs, len(self.link_names))
        self.data.xfrc_applied[:] = 0
        if self._interactive_force_callback is not None:
            self._interactive_force_callback(self.model, self.data)
        if base_force is not None:
            if base_force.shape != (1, 3) or not torch.isfinite(base_force).all():
                raise ValueError("base_force must have shape (1, 3) and be finite")
            self.data.xfrc_applied[self.base, :3] += base_force[0].detach().cpu().numpy()
        if link_point_forces is not None:
            self._apply_link_point_forces(link_point_forces)
        self.data.ctrl[self.motors] = torque[0].detach().cpu().numpy()
        mujoco.mj_step(self.model, self.data)
        # mj_step advances qpos after computing derived kinematics.
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
