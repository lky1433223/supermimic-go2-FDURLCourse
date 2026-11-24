import torch

from rl_go2.backends import LinkPointForces
from rl_go2.config import (
    FOOT_NAMES,
    base_com_offset_ranges,
    base_mass_scale_range,
    foot_friction_ranges,
    joint_position_bounds,
)
from rl_go2.state import RobotState


class GenesisBackend:
    def __init__(self, config, viewer=False):
        import genesis as gs

        bounds = joint_position_bounds(config["robot"])
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
        com_enabled = base_com_offset_ranges(config) is not None
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim["dt"], substeps=1, gravity=tuple(sim["gravity"])),
            rigid_options=gs.options.RigidOptions(
                enable_mujoco_compatibility=True, enable_self_collision=True,
                friction_cone=gs.friction_cone.elliptic,
                enable_torsional_friction=True, enable_rolling_friction=True,
                batch_links_info=com_enabled or base_mass_scale_range(config) is not None,
            ),
            show_viewer=viewer,
            viewer_options=gs.options.ViewerOptions(**config.get('viewer_options', {})),
            vis_options=gs.options.VisOptions(**config.get('vis_options', {})),
        )
        friction_enabled = foot_friction_ranges(config) is not None
        surfaces = []
        for entity in config["scene"]["entities"]:
            kwargs = dict(pos=tuple(entity["position"]), quat=tuple(entity["rotation"]))
            if entity.get("robot"):
                if com_enabled:
                    kwargs["align"] = False
                self.robot = self.scene.add_entity(gs.morphs.MJCF(file=entity["file"], **kwargs))
            else:
                if entity["type"] == "plane":
                    morph = gs.morphs.Plane(**kwargs)
                else:
                    morph = gs.morphs.Box(size=tuple(entity["size"]), fixed=True, **kwargs)
                surface = self.scene.add_entity(morph, material=gs.materials.Rigid(friction=entity.get("friction", 0.8)))
                if friction_enabled:
                    surfaces.append(surface)
        self.scene.build(n_envs=self.num_envs)
        self.friction_ratios = None
        if friction_enabled:
            # Genesis can reuse metadata from identical sphere meshes. Identify
            # each foot by its calf link and the unique sphere collision geom.
            feet = [[geom.idx for geom in self.robot.get_link(f"{name}_calf").geoms
                     if geom.type == gs.GEOM_TYPE.SPHERE] for name in FOOT_NAMES]
            if not surfaces or any(len(surface.geoms) != 1 for surface in surfaces) or any(
                len(ids) != 1 for ids in feet
            ):
                raise ValueError("Foot friction requires one geom per surface and one collision sphere per Go2 calf")
            self.friction_geom_ids = [ids[0] for ids in feet] + [surface.geoms[0].idx for surface in surfaces]
        self.joints = [self.robot.get_joint(name).dofs_idx_local[0] for name in config["robot"]["joint_names"]]
        if bounds is not None:
            lower, upper = zip(*bounds)
            self.robot.set_dofs_limit(lower, upper, dofs_idx_local=self.joints)
        self.base = self.robot.get_link(config["robot"]["base_link"])
        links = sorted(self.robot.links, key=lambda link: link.name)
        self.link_names = tuple(link.name for link in links)
        self._link_indices = [link.idx_local for link in links]
        self.reset(torch.arange(self.num_envs, device=self.device))

    def set_foot_ground_friction(self, base, multipliers):
        # Genesis takes the per-component max of the two scaled geoms.
        surface_count = len(self.friction_geom_ids) - base.shape[0]
        ground = base.amin(dim=0, keepdim=True).expand(surface_count, -1)
        sliding, torsional, rolling = torch.cat((base, ground)).to(self.device).unbind(dim=1)
        solver = self.scene.rigid_solver
        solver.set_geoms_friction(sliding.contiguous(), self.friction_geom_ids)
        solver.set_geoms_friction_torsional(torsional.contiguous(), self.friction_geom_ids)
        solver.set_geoms_friction_rolling(rolling.contiguous(), self.friction_geom_ids)
        if self.friction_ratios is None:
            self.friction_ratios = multipliers.to(self.device)[:, None].expand(-1, len(self.friction_geom_ids)).contiguous()
            solver.set_geoms_friction_ratio(self.friction_ratios, self.friction_geom_ids)

    def set_foot_friction_multipliers(self, multipliers, env_ids):
        ratios = multipliers[:, None].expand(-1, len(self.friction_geom_ids)).contiguous()
        self.scene.rigid_solver.set_geoms_friction_ratio(ratios, self.friction_geom_ids, env_ids)
        self.friction_ratios[env_ids] = ratios

    def set_base_com_offset(self, offsets):
        nominal = offsets.new_tensor(self.base.desc.inertial_pos)
        com = (nominal + offsets).to(self.device)
        self.scene.rigid_solver.set_links_COM(com[:, None, :], [self.base.idx])

    def set_base_mass_scale(self, scales):
        mass = scales.to(self.device) * self.base.desc.mass
        if not torch.isfinite(mass).all() or (mass <= 0).any():
            raise ValueError("Sampled base mass must be finite and positive")
        self.scene.rigid_solver.set_links_mass(mass, [self.base.idx], scale_inertia=True)
        return mass

    def read_state(self):
        # Getter tensors can alias engine buffers, so snapshots must own their storage.
        return RobotState(self.robot.get_pos(), self.robot.get_quat(), self.robot.get_vel(),
                          self.robot.get_ang(), self.robot.get_dofs_position(self.joints),
                          self.robot.get_dofs_velocity(self.joints)).clone()

    def reset(self, env_ids):
        if env_ids.numel() == 0:
            return
        self.scene.reset(envs_idx=env_ids)
        if self.friction_ratios is not None:
            self.scene.rigid_solver.set_geoms_friction_ratio(
                self.friction_ratios[env_ids], self.friction_geom_ids, env_ids)
        entity = next(e for e in self.config["scene"]["entities"] if e.get("robot"))
        # MJCF morph.pos composes with the XML root pose; canonical reset is absolute.
        position = torch.tensor(entity["position"], device=self.device)
        quaternion = torch.tensor(entity["rotation"], device=self.device, dtype=torch.float32)
        self.robot.set_pos(position.expand(len(env_ids), -1), envs_idx=env_ids)
        self.robot.set_quat(quaternion.expand(len(env_ids), -1), envs_idx=env_ids)
        robot = self.config["robot"]
        positions = torch.tensor(robot.get("reset_joint_positions", robot["default_joint_positions"]),
                                 device=self.device)
        self.robot.set_dofs_position(positions.expand(len(env_ids), -1), self.joints, envs_idx=env_ids)

    def set_base_pose(self, env_ids: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor) -> None:
        """Place reset environments at the selected terrain spawn poses."""
        self.robot.set_pos(position, envs_idx=env_ids)
        self.robot.set_quat(quaternion, envs_idx=env_ids)

    def _apply_link_point_forces(self, point_forces: LinkPointForces) -> None:
        link_quaternion = self.read_link_quaternions()
        link_position = self.robot.get_links_pos(self._link_indices)
        offsets = point_forces.point_local_m.detach().to(link_position)
        forces = point_forces.force_world_n.detach().to(link_position)
        point_world = link_position + self.gs.utils.geom.transform_by_quat(offsets, link_quaternion)
        self.robot.apply_links_external_wrench(
            force=forces, links_idx_local=self._link_indices, pos=point_world,
            ref=self.gs.link_ref_frame.link_origin, local=False,
        )

    def read_link_quaternions(self) -> torch.Tensor:
        """Return current authored link orientations in link_names order (world wxyz)."""
        # MuJoCo compatibility defers the post-integration Cartesian pose update.
        self.scene.rigid_solver.update_forward_pos()
        return self.robot.get_links_quat(self._link_indices)

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
        if base_force is not None:
            if base_force.shape != (self.num_envs, 3) or not torch.isfinite(base_force).all():
                raise ValueError(f"base_force must have shape ({self.num_envs}, 3) and be finite")
            self.base.apply_external_wrench(force=base_force, ref=self.gs.link_ref_frame.link_COM, local=False)
        if link_point_forces is not None:
            self._apply_link_point_forces(link_point_forces)
        self.robot.control_dofs_force(torque, self.joints)
        self.scene.step()

    def close(self):
        self.scene.destroy()
        self.gs.destroy()
