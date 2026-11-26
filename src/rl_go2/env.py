from dataclasses import dataclass
import math

import torch

from rl_go2.backends import LinkPointForces, make_backend
from rl_go2.config import base_com_offset_ranges, base_mass_scale_range, foot_friction_ranges, foot_friction_switch_seconds
from rl_go2.control import JointPositionControl
from rl_go2.curriculum import make_curriculum
from rl_go2.observation_noise import NoiseHistory, ObservationNoise
from rl_go2.observations import SensorReadings
from rl_go2.state import RobotState
from rl_go2.step_delay import StepDelayState, action_delay_from_config
from rl_go2.tasks import get_task_type
from rl_go2.tasks.walk_rewards import WalkRewards
from rl_go2.terrain_curriculum import TerrainCurriculum
from rl_go2.terrain_features import TerrainHeightScan


@dataclass
class Transition:
    """Record the applied action and pre-reset next state, with command history for the actor."""

    state: RobotState
    action: torch.Tensor
    next_state: RobotState
    context: dict[str, torch.Tensor]
    terminated: torch.Tensor
    truncated: torch.Tensor
    previous_action: torch.Tensor
    torque: torch.Tensor
    # (envs, links, 6), world force and moment about each authored link origin at step start.
    link_wrenches_world: torch.Tensor | None = None
    action_delay_state: StepDelayState | None = None  # Before receiving this step's command.
    contact_labels: torch.Tensor | None = None  # FR, FL, RR, RL, prohibited contact.
    prohibited_contact_force_n: torch.Tensor | None = None


class QuadrupedEnv:
    def __init__(self, config, *, viewer=False, record_transitions=False, backend=None):
        self.config = config
        task_type = get_task_type(config["task"]["name"])
        task_type.validate_config(config["task"], config["robot"])
        self.action_delay = action_delay_from_config(config)
        self.action_delay_state: StepDelayState | None = None
        self.noise_history: NoiseHistory = {}
        self._sensor_readings: SensorReadings | None = None
        self.curriculum = make_curriculum(config["curriculum"])
        self.backend = backend if backend is not None else make_backend(config, viewer)
        self.device = self.backend.device
        self.num_envs = self.backend.num_envs
        self.num_actions = len(config["robot"]["joint_names"])
        self.dt = config["scene"]["sim_options"]["dt"] * config["robot"]["control_decimation"]
        self.max_episode_length = math.ceil(config["task"]["episode_seconds"] / self.dt)
        self.controller = JointPositionControl(config["robot"], self.device)
        self.observation_noise = ObservationNoise.from_config(
            config, device=self.device, dtype=self.controller.default.dtype
        )
        self.task = task_type(config["task"], self.controller.default, self.num_envs,
                              self.device, self.dt, config["seed"])
        self.walk_rewards = None
        if config["task"]["name"] in ("flat_walk", "push_walk"):
            self.walk_rewards = WalkRewards(config["task"], config["robot"], self.controller.default, self.dt)
        if self.walk_rewards is not None and self.walk_rewards.requires_contacts:
            self.backend.configure_ground_contacts(self.walk_rewards.exempt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.environment_steps = 0
        self.record_transitions = record_transitions
        scene = config["scene"]
        self.ground_height_m = next((e["position"][2] for e in scene["entities"] if e["type"] == "plane"), 0.0)
        patches = scene.get('terrain_patches', [])
        self.terrain_scan = None
        if scene.get("height_maps") or patches or any(e["type"] == "box" for e in scene["entities"]):
            self.terrain_scan = TerrainHeightScan(scene).to(self.device)
        self.terrain_patches = patches
        self._fixed_terrain_index: int | None = None
        self.terrain_curriculum: TerrainCurriculum | None = None
        if patches:
            self.terrain_generator = torch.Generator().manual_seed(config['seed'] ^ 0x53504157)
            self.terrain_weights = torch.tensor([patch['weight'] for patch in patches])
            self.terrain_positions = torch.tensor([patch['position'] for patch in patches], device=self.device)
            self.terrain_rotations = torch.tensor([patch['rotation'] for patch in patches], device=self.device)
            self.terrain_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            if 'terrain_curriculum' in scene:
                self.terrain_curriculum = TerrainCurriculum(
                    patches, scene['terrain_curriculum'], self.num_envs, self.device,
                    self.terrain_generator, self.dt, self.max_episode_length,
                )
        self.foot_friction = None
        self.friction_bounds = None
        self.friction_switch_steps = None
        ranges = foot_friction_ranges(config)
        if ranges is not None:
            base, ratio = ranges
            self.friction_generator = torch.Generator().manual_seed(config["seed"])
            self.friction_bounds = torch.tensor(base, dtype=torch.float32)
            self.friction_multipliers = ratio[0] + (ratio[1] - ratio[0]) * torch.rand(
                self.num_envs, generator=self.friction_generator)
            self.resample_foot_friction()
            interval = foot_friction_switch_seconds(config)
            if interval is not None:
                self.friction_switch_steps = tuple(max(1, round(seconds / self.dt)) for seconds in interval)
                self.friction_ratio_range = ratio
                self.friction_switch_base = self.friction_bounds[..., 0].flatten().to(self.device)
                self.friction_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.base_mass_kg = None
        mass_range = base_mass_scale_range(config)
        if mass_range is not None:
            low, high = mass_range
            generator = torch.Generator().manual_seed(config["seed"] ^ 0x4D415353)
            scales = low + (high - low) * torch.rand(self.num_envs, 1, generator=generator)
            self.base_mass_kg = self.backend.set_base_mass_scale(scales)
        self.base_com_offset_m = None
        com_ranges = base_com_offset_ranges(config)
        if com_ranges is not None:
            low, high = torch.tensor(com_ranges, dtype=torch.float32).unbind(-1)
            generator = torch.Generator().manual_seed(config["seed"] ^ 0x434F4D)
            offsets = low + (high - low) * torch.rand(self.num_envs, 3, generator=generator)
            self.backend.set_base_com_offset(offsets)
            self.base_com_offset_m = offsets.to(self.device)
        self.reset()

    def resample_foot_friction(self):
        if self.friction_bounds is None or self.friction_switch_steps is not None:
            return
        low, high = self.friction_bounds.unbind(-1)
        base = low + (high - low) * torch.rand(low.shape, generator=self.friction_generator)
        self.backend.set_foot_ground_friction(base, self.friction_multipliers)
        self.foot_friction = (self.friction_multipliers[:, None, None] * base).flatten(1).to(self.device)

    def sample_friction_segment(self, count, device, generator=None):
        low, high = self.friction_ratio_range
        ratio = low + (high - low) * torch.rand(count, device=device, generator=generator)
        first, last = self.friction_switch_steps
        steps = torch.randint(first, last + 1, (count,), device=device, generator=generator)
        return ratio, steps

    def reset_friction_segment(self, env_ids):
        if not env_ids.numel():
            return
        ratio, steps = self.sample_friction_segment(env_ids.numel(), "cpu", self.friction_generator)
        ratio = ratio.to(self.device)
        self.backend.set_foot_friction_multipliers(ratio, env_ids)
        self.foot_friction[env_ids] = ratio[:, None] * self.friction_switch_base
        self.friction_steps_left[env_ids] = steps.to(self.device)

    def set_fixed_friction_multiplier(self, multiplier: float) -> None:
        """Apply an absolute multiplier to the fixed per-foot friction without resetting."""
        if self.friction_bounds is None or self.friction_switch_steps is not None:
            raise ValueError("Fixed friction adjustment requires fixed per-foot friction")
        low, high = self.friction_bounds.unbind(-1)
        if not torch.equal(low, high) or not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Fixed friction adjustment requires constant bounds and a positive multiplier")
        self.friction_multipliers.fill_(multiplier)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.backend.set_foot_friction_multipliers(self.friction_multipliers.to(self.device), env_ids)
        self.foot_friction.copy_((multiplier * low).flatten().to(self.device).expand_as(self.foot_friction))

    def reset_with_episode_parameters(
        self, *, mass_scale: float, com_offset_m: tuple[float, float, float],
        observation_noise: dict, action_delay_steps: int,
    ) -> None:
        """Set fixed body and sensor parameters and reset on the currently selected terrain."""
        if self.base_mass_kg is None or self.base_com_offset_m is None:
            raise ValueError("Episode adjustment requires enabled base mass and CoM")
        config = {'domain_randomization': {
            'observation_noise': observation_noise,
            'action_delay_steps': [action_delay_steps, action_delay_steps],
        }}
        action_delay = action_delay_from_config(config)
        noise = ObservationNoise.from_config(config, device=self.device, dtype=self.controller.default.dtype)
        self.base_mass_kg = self.backend.set_base_mass_scale(torch.full((self.num_envs, 1), mass_scale))
        offsets = torch.tensor(com_offset_m).expand(self.num_envs, -1).contiguous()
        self.backend.set_base_com_offset(offsets)
        self.base_com_offset_m = offsets.to(self.device)
        self.action_delay = action_delay
        self.action_delay_state = None
        self.observation_noise = noise
        self._sensor_readings = None
        self.noise_history = {}
        self.reset()

    def advance_context(self, context, state):
        context = self.task.advance_context(context, state)
        if "episode_step" in context:
            context["episode_step"] = context["episode_step"] + 1
        if self.friction_switch_steps is None:
            return context
        left = context["friction_steps_left"] - 1
        ratio, steps = self.sample_friction_segment(left.shape[0], left.device)
        changed = left <= 0
        return {
            **context,
            "friction_steps_left": torch.where(changed, steps, left),
            "foot_ground_friction": torch.where(changed[:, None], ratio[:, None] * self.friction_switch_base,
                                                context["foot_ground_friction"]),
        }

    def model_context(self):
        context = self.task.context()
        if self.walk_rewards is not None and self.walk_rewards.gait_weight:
            context["episode_step"] = self.episode_length_buf
        if self.foot_friction is not None:
            context["foot_ground_friction"] = self.foot_friction
        if self.friction_switch_steps is not None:
            context["friction_steps_left"] = self.friction_steps_left
        if self.base_mass_kg is not None:
            context["base_mass_kg"] = self.base_mass_kg
        if self.base_com_offset_m is not None:
            context["base_com_offset_m"] = self.base_com_offset_m
        return context

    def walking_height_m(self, state: RobotState) -> torch.Tensor | None:
        """Return terrain-relative walking height, or use the task's default world Z."""
        if self.walk_rewards is None:
            return None
        if self.terrain_scan is not None:
            return self.terrain_scan.base_clearance_m(state.position, state.quaternion)
        if self.ground_height_m:
            return state.position[:, 2] - self.ground_height_m
        return None

    def select_terrain(self, index: int) -> None:
        """Reset on the selected patch and keep subsequent resets on that patch."""
        if not 0 <= index < len(self.terrain_patches):
            raise ValueError(f'Terrain patch index out of range: {index}')
        self._fixed_terrain_index = index
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None, metrics: dict | None = None) -> torch.Tensor:
        """Reset the selected environments and initialize their delays for the new episode."""
        reset_all = env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        updates = self.curriculum.on_reset(env_ids, environment_steps=self.environment_steps, metrics=metrics or {})
        self.backend.reset(env_ids)
        if self.terrain_patches and len(env_ids):
            if self._fixed_terrain_index is not None:
                selected = torch.full_like(env_ids, self._fixed_terrain_index)
            elif self.terrain_curriculum is not None:
                selected = self.terrain_curriculum.sample_patches(env_ids)
            else:
                selected = torch.multinomial(self.terrain_weights, len(env_ids), replacement=True,
                                             generator=self.terrain_generator).to(self.device)
            self.terrain_ids[env_ids] = selected
            self.backend.set_base_pose(env_ids, self.terrain_positions[selected], self.terrain_rotations[selected])
        if self.friction_switch_steps is not None:
            self.reset_friction_segment(env_ids)
        self.state = self.backend.read_state()
        if self.terrain_curriculum is not None and self._fixed_terrain_index is None:
            self.terrain_curriculum.start_episodes(
                env_ids, self.terrain_ids[env_ids], self.state.position, self.state.quaternion,
            )
        self.task.reset(env_ids, self.state, updates)
        self.episode_length_buf[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.previous_action[env_ids] = 0
        self._reset_action_delay(env_ids, reset_all)
        if self.observation_noise is not None:
            readings = SensorReadings.from_state(self.state)
            if reset_all:
                self._sensor_readings, self.noise_history = self.observation_noise.apply(readings)
            elif env_ids.numel():
                self._reset_sensor_readings(readings, env_ids)
        return self.observe()

    def _reset_action_delay(self, env_ids: torch.Tensor, reset_all: bool) -> None:
        if self.action_delay is None or not env_ids.numel():
            return
        reset_state = self.action_delay.reset(self.previous_action[env_ids])
        if reset_all:
            self.action_delay_state = reset_state
            return
        old = self.action_delay_state
        assert old is not None
        self.action_delay_state = StepDelayState(old.steps.index_copy(0, env_ids, reset_state.steps),
                                                old.history.index_copy(0, env_ids, reset_state.history))

    def _reset_sensor_readings(self, readings: SensorReadings, env_ids: torch.Tensor) -> None:
        selected = SensorReadings(*(reading[env_ids] for reading in readings))
        reset_readings, reset_history = self.observation_noise.apply(selected)
        self._sensor_readings = SensorReadings(*(
            old.index_copy(0, env_ids, new)
            for old, new in zip(self._sensor_readings, reset_readings, strict=True)
        ))
        self.noise_history = {key: self.noise_history[key].index_copy(0, env_ids, value)
                              for key, value in reset_history.items()}

    def observe(self, *, clean: bool = False) -> torch.Tensor:
        """Return the current actor observation without advancing any sensor history."""
        if clean or self._sensor_readings is None:
            return self.task.observe(self.state, self.previous_action)
        return self.task.observe(self.state, self.previous_action, readings=self._sensor_readings)

    def step(
        self,
        action: torch.Tensor,
        *,
        base_force: torch.Tensor | None = None,
        link_point_forces: LinkPointForces | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance one control step, holding optional point loads over its physics substeps."""
        action = action.detach().to(self.device)
        if action.shape != self.previous_action.shape or not torch.isfinite(action).all():
            raise ValueError(f"Expected finite actions of shape {tuple(self.previous_action.shape)}")
        old_state = self.state
        if base_force is None:
            base_force = self.task.external_force(self.state, self.episode_length_buf)
        context = ({name: value.detach().clone() for name, value in self.model_context().items()}
                   if self.record_transitions else None)
        if base_force is not None:
            base_force = base_force.detach().to(self.device)
            if base_force.shape != (self.num_envs, 3) or not torch.isfinite(base_force).all():
                raise ValueError(f"base_force must be finite with shape ({self.num_envs}, 3)")
            if context is not None:
                context["external_force_world"] = base_force.clone()
        link_wrenches_world = None
        if link_point_forces is not None:
            link_point_forces.validate(self.num_envs, len(self.backend.link_names))
            link_point_forces = LinkPointForces(
                link_point_forces.point_local_m.detach().to(self.state.position),
                link_point_forces.force_world_n.detach().to(self.state.position),
            )
            if self.record_transitions:
                link_wrenches_world = link_point_forces.wrenches_world(self.backend.read_link_quaternions())
        issued_action = self.controller.clip_action(action)
        delay_state = self.action_delay_state
        next_delay_state = None
        applied_action = issued_action
        if self.action_delay is not None:
            assert delay_state is not None
            applied_action = self.action_delay.select(issued_action, delay_state)
            next_delay_state = self.action_delay.advance(issued_action, delay_state)
        target = self.controller.position_target(applied_action)
        walk = self.walk_rewards
        track_contacts = walk is not None and walk.requires_contacts
        peak_contact_force = torch.zeros_like(self.episode_return) if track_contacts else None
        contacts = None
        for _ in range(self.config["robot"]["control_decimation"]):
            torque = self.controller.torque(target, self.state)
            if link_point_forces is not None:
                self.backend.step(torque, base_force=base_force, link_point_forces=link_point_forces)
            elif base_force is None:
                self.backend.step(torque)
            else:
                self.backend.step(torque, base_force=base_force)
            self.state = self.backend.read_state()
            if peak_contact_force is not None:
                contacts = self.backend.read_ground_contacts()
                peak_contact_force = torch.maximum(peak_contact_force, contacts.prohibited_force_n)
        self.episode_length_buf += 1
        self.environment_steps += self.num_envs
        height = self.walking_height_m(self.state)
        height_arg = {} if height is None else {"base_height_m": height}
        reward = self.task.reward(self.state, issued_action, self.previous_action, torque, **height_arg)
        terminated = self.task.terminated(self.state, **height_arg)
        contact_labels = None
        contact_failure = None
        if walk is not None:
            feet = None
            if contacts is not None:
                assert peak_contact_force is not None
                contact_failure = peak_contact_force > walk.force_threshold_n
                feet = contacts.feet.to(reward.dtype)
                contact_labels = torch.cat((feet, contact_failure[:, None].to(reward.dtype)), dim=-1)
                if walk.contact_enabled:
                    terminated = terminated | contact_failure
            if walk.failure_weight or walk.posture_weight or walk.gait_weight:
                commands = self.task.context()["commands"] if walk.gait_weight else None
                reward = reward + walk.extra_reward(self.state.joint_position, terminated, foot_contact=feet,
                                                    episode_step=self.episode_length_buf, commands=commands)
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        done = terminated | truncated
        terrain_curriculum = self.terrain_curriculum if self._fixed_terrain_index is None else None
        if terrain_curriculum is not None:
            commands = context["commands"] if context is not None else self.task.context()["commands"]
            terrain_curriculum.step(self.state.position, commands, terminated, self.episode_length_buf)
        self.episode_return += reward
        old_action = self.previous_action.clone() if self.record_transitions else None
        # Keep a normal persistent buffer so reset also works outside inference_mode.
        self.previous_action.copy_(issued_action)
        self.action_delay_state = next_delay_state
        if self.observation_noise is not None:
            readings = SensorReadings.from_state(self.state)
            self._sensor_readings, self.noise_history = self.observation_noise.apply(readings, self.noise_history)
        extras = {"time_outs": truncated, "terminated": terminated, "truncated": truncated}
        if contact_failure is not None:
            extras["prohibited_contact"] = contact_failure
            extras["prohibited_contact_force_n"] = peak_contact_force
        # Allocate only when needed; buffers returned to the learner never alias future in-place updates.
        if self.record_transitions:
            extras["transition"] = Transition(old_state.clone(), applied_action.clone(), self.state.clone(),
                                               context, terminated.clone(), truncated.clone(),
                                               old_action, torque.detach().clone(), link_wrenches_world, delay_state,
                                               contact_labels, peak_contact_force)
        done_ids = done.nonzero(as_tuple=False).flatten()
        if self.friction_switch_steps is not None:
            self.friction_steps_left.sub_(1)
            switch_ids = ((self.friction_steps_left <= 0) & ~done).nonzero(as_tuple=False).flatten()
            self.reset_friction_segment(switch_ids)
        if done_ids.numel():
            extras["final_env_ids"] = done_ids
            extras["final_observation"] = self.observe()[done_ids].clone()
            extras["log"] = {
                "/episode/return": self.episode_return[done_ids].mean().item(),
                "/episode/seconds": (self.episode_length_buf[done_ids].float().mean() * self.dt).item(),
            }
            if terrain_curriculum is not None:
                extras["terrain_curriculum"] = terrain_curriculum.finish_episodes(done_ids, terminated)
            self.reset(done_ids, extras["log"])
            curriculum_state = self.curriculum.state_dict()
            if "stage" in curriculum_state:
                extras["log"]["/curriculum/stage"] = curriculum_state["stage"]
        self.task.on_step_end(self.state, self.episode_length_buf)
        return self.observe(), reward, done, extras

    def close(self):
        self.backend.close()
