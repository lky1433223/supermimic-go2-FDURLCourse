import torch

from rl_go2.state import RobotState


class GenesisBackend:
    def __init__(self, config, viewer=False):
        import genesis as gs

        device = config["device"]
        choices = {"cpu": gs.cpu, "cuda": gs.cuda, "mps": gs.metal}
        if device not in choices:
            raise ValueError("Genesis device must be cpu, cuda or mps")
        gs.init(backend=choices[device], seed=config["seed"], precision="32", logging_level="warning")
        self.gs = gs
        self.device = torch.device(gs.device)
        self.num_envs = config["num_envs"]
        self.config = config
        sim = config["scene"]["sim_options"]
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim["dt"], substeps=1, gravity=tuple(sim["gravity"])),
            rigid_options=gs.options.RigidOptions(
                enable_mujoco_compatibility=True, enable_self_collision=True,
                friction_cone=gs.friction_cone.elliptic,
                enable_torsional_friction=True, enable_rolling_friction=True,
            ),
            show_viewer=viewer,
        )
        for entity in config["scene"]["entities"]:
            kwargs = dict(pos=tuple(entity["position"]), quat=tuple(entity["rotation"]))
            if entity.get("robot"):
                self.robot = self.scene.add_entity(gs.morphs.MJCF(file=entity["file"], **kwargs))
            else:
                if entity["type"] == "plane":
                    morph = gs.morphs.Plane(**kwargs)
                else:
                    morph = gs.morphs.Box(size=tuple(entity["size"]), fixed=True, **kwargs)
                self.scene.add_entity(morph, material=gs.materials.Rigid(friction=entity.get("friction", 0.8)))
        self.scene.build(n_envs=self.num_envs)
        self.joints = [self.robot.get_joint(name).dofs_idx_local[0] for name in config["robot"]["joint_names"]]
        self.reset(torch.arange(self.num_envs, device=self.device))

    def read_state(self):
        # Getter tensors can alias engine buffers, so snapshots must own their storage.
        return RobotState(self.robot.get_pos(), self.robot.get_quat(), self.robot.get_vel(),
                          self.robot.get_ang(), self.robot.get_dofs_position(self.joints),
                          self.robot.get_dofs_velocity(self.joints)).clone()

    def reset(self, env_ids):
        if env_ids.numel() == 0:
            return
        self.scene.reset(envs_idx=env_ids)
        entity = next(e for e in self.config["scene"]["entities"] if e.get("robot"))
        # MJCF morph.pos composes with the XML root pose; canonical reset is absolute.
        position = torch.tensor(entity["position"], device=self.device)
        quaternion = torch.tensor(entity["rotation"], device=self.device, dtype=torch.float32)
        self.robot.set_pos(position.expand(len(env_ids), -1), envs_idx=env_ids)
        self.robot.set_quat(quaternion.expand(len(env_ids), -1), envs_idx=env_ids)
        positions = torch.tensor(self.config["robot"]["default_joint_positions"], device=self.device)
        self.robot.set_dofs_position(positions.expand(len(env_ids), -1), self.joints, envs_idx=env_ids)

    def step(self, torque):
        self.robot.control_dofs_force(torque, self.joints)
        self.scene.step()

    def close(self):
        self.scene.destroy()
        self.gs.destroy()
