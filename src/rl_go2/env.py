from dataclasses import dataclass
import math

import torch

from rl_go2.backends import make_backend
from rl_go2.control import JointPositionControl
from rl_go2.curriculum import make_curriculum
from rl_go2.state import RobotState
from rl_go2.tasks import FlatWalkTask


@dataclass
class Transition:
    """Optional model-learning record. next_state is ALWAYS before automatic reset."""

    state: RobotState
    action: torch.Tensor
    next_state: RobotState
    commands: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class QuadrupedEnv:
    def __init__(self, config, *, viewer=False, record_transitions=False, backend=None):
        self.config = config
        self.curriculum = make_curriculum(config["curriculum"])
        self.backend = backend if backend is not None else make_backend(config, viewer)
        self.device = self.backend.device
        self.num_envs = self.backend.num_envs
        self.num_actions = len(config["robot"]["joint_names"])
        self.dt = config["scene"]["sim_options"]["dt"] * config["robot"]["control_decimation"]
        self.max_episode_length = math.ceil(config["task"]["episode_seconds"] / self.dt)
        self.controller = JointPositionControl(config["robot"], self.device)
        self.task = FlatWalkTask(config["task"], self.controller.default, self.num_envs,
                                 self.device, self.dt, config["seed"])
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.environment_steps = 0
        self.record_transitions = record_transitions
        self.reset()

    def reset(self, env_ids=None, metrics=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        updates = self.curriculum.on_reset(env_ids, environment_steps=self.environment_steps, metrics=metrics or {})
        self.backend.reset(env_ids)
        self.task.resample(env_ids, updates)
        self.episode_length_buf[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.previous_action[env_ids] = 0
        self.state = self.backend.read_state()
        return self.observe()

    def observe(self):
        return self.task.observe(self.state, self.previous_action)

    def step(self, action):
        action = action.detach().to(self.device)
        if action.shape != self.previous_action.shape or not torch.isfinite(action).all():
            raise ValueError(f"Expected finite actions of shape {tuple(self.previous_action.shape)}")
        old_state = self.state
        target, clipped_action = self.controller.target(action)
        for _ in range(self.config["robot"]["control_decimation"]):
            torque = self.controller.torque(target, self.state)
            self.backend.step(torque)
            self.state = self.backend.read_state()
        self.episode_length_buf += 1
        self.environment_steps += self.num_envs
        reward = self.task.reward(self.state, clipped_action, self.previous_action, torque)
        terminated = self.task.terminated(self.state)
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        done = terminated | truncated
        self.episode_return += reward
        self.previous_action = clipped_action.clone()
        extras = {"time_outs": truncated, "terminated": terminated, "truncated": truncated}
        # Allocate only when needed; buffers returned to the learner never alias future in-place updates.
        if self.record_transitions:
            extras["transition"] = Transition(old_state.clone(), clipped_action.clone(), self.state.clone(),
                                               self.task.commands.clone(), terminated.clone(), truncated.clone())
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            extras["final_env_ids"] = done_ids
            extras["final_observation"] = self.observe()[done_ids].clone()
            extras["log"] = {
                "/episode/return": self.episode_return[done_ids].mean().item(),
                "/episode/seconds": (self.episode_length_buf[done_ids].float().mean() * self.dt).item(),
            }
            self.reset(done_ids, extras["log"])
        periodic_ids = ((self.episode_length_buf > 0) &
                        (self.episode_length_buf % self.task.resample_steps == 0)).nonzero(as_tuple=False).flatten()
        self.task.resample(periodic_ids)
        return self.observe(), reward, done, extras

    def close(self):
        self.backend.close()
