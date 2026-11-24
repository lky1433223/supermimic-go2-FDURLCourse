from dataclasses import dataclass
import math

import torch

from rl_go2.backends import LinkPointForces, make_backend
from rl_go2.config import base_com_offset_ranges, base_mass_scale_range, foot_friction_ranges, foot_friction_switch_seconds
from rl_go2.control import JointPositionControl
from rl_go2.curriculum import make_curriculum
from rl_go2.state import RobotState
from rl_go2.tasks import get_task_type


@dataclass
class Transition:
    """Optional model-learning record. next_state is ALWAYS before automatic reset."""

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


class QuadrupedEnv:
    def __init__(self, config, *, viewer=False, record_transitions=False, backend=None):
        self.config = config
        task_type = get_task_type(config["task"]["name"])
        task_type.validate_config(config["task"], config["robot"])
        self.curriculum = make_curriculum(config["curriculum"])
        self.backend = backend if backend is not None else make_backend(config, viewer)
        self.device = self.backend.device
        self.num_envs = self.backend.num_envs
        self.num_actions = len(config["robot"]["joint_names"])
        self.dt = config["scene"]["sim_options"]["dt"] * config["robot"]["control_decimation"]
        self.max_episode_length = math.ceil(config["task"]["episode_seconds"] / self.dt)
        self.controller = JointPositionControl(config["robot"], self.device)
        self.task = task_type(config["task"], self.controller.default, self.num_envs,
                              self.device, self.dt, config["seed"])
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.environment_steps = 0
        self.record_transitions = record_transitions
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

    def advance_context(self, context, state):
        context = self.task.advance_context(context, state)
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
        if self.foot_friction is not None:
            context["foot_ground_friction"] = self.foot_friction
        if self.friction_switch_steps is not None:
            context["friction_steps_left"] = self.friction_steps_left
        if self.base_mass_kg is not None:
            context["base_mass_kg"] = self.base_mass_kg
        if self.base_com_offset_m is not None:
            context["base_com_offset_m"] = self.base_com_offset_m
        return context

    def reset(self, env_ids=None, metrics=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        updates = self.curriculum.on_reset(env_ids, environment_steps=self.environment_steps, metrics=metrics or {})
        self.backend.reset(env_ids)
        if self.friction_switch_steps is not None:
            self.reset_friction_segment(env_ids)
        self.state = self.backend.read_state()
        self.task.reset(env_ids, self.state, updates)
        self.episode_length_buf[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.previous_action[env_ids] = 0
        return self.observe()

    def observe(self):
        return self.task.observe(self.state, self.previous_action)

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
        target, clipped_action = self.controller.target(action)
        for _ in range(self.config["robot"]["control_decimation"]):
            torque = self.controller.torque(target, self.state)
            if link_point_forces is not None:
                self.backend.step(torque, base_force=base_force, link_point_forces=link_point_forces)
            elif base_force is None:
                self.backend.step(torque)
            else:
                self.backend.step(torque, base_force=base_force)
            self.state = self.backend.read_state()
        self.episode_length_buf += 1
        self.environment_steps += self.num_envs
        reward = self.task.reward(self.state, clipped_action, self.previous_action, torque)
        terminated = self.task.terminated(self.state)
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        done = terminated | truncated
        self.episode_return += reward
        old_action = self.previous_action.clone() if self.record_transitions else None
        # Keep a normal persistent buffer so reset also works outside inference_mode.
        self.previous_action.copy_(clipped_action)
        extras = {"time_outs": truncated, "terminated": terminated, "truncated": truncated}
        # Allocate only when needed; buffers returned to the learner never alias future in-place updates.
        if self.record_transitions:
            extras["transition"] = Transition(old_state.clone(), clipped_action.clone(), self.state.clone(),
                                               context, terminated.clone(), truncated.clone(),
                                               old_action, torque.detach().clone(), link_wrenches_world)
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
            self.reset(done_ids, extras["log"])
            curriculum_state = self.curriculum.state_dict()
            if "stage" in curriculum_state:
                extras["log"]["/curriculum/stage"] = curriculum_state["stage"]
        self.task.on_step_end(self.state, self.episode_length_buf)
        return self.observe(), reward, done, extras

    def close(self):
        self.backend.close()
