import math

import torch

from rl_go2.state import rotate_inverse
from rl_go2.observations import OBSERVATION_TERMS, policy_observation


class FlatWalkTask:
    """Minimal velocity tracking task; reward coefficients are initial values, not tuned results."""

    observation_terms = OBSERVATION_TERMS

    @staticmethod
    def validate_config(config, robot):
        if len(robot["joint_names"]) != 12:
            raise ValueError("flat_walk expects a 12-joint quadruped")
        for key in ("command_resample_seconds", "tracking_sigma"):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        ranges = config["command_ranges"]
        if len(ranges) != 3 or any(len(pair) != 2 or pair[1] < pair[0]
                                   or not all(math.isfinite(x) for x in pair) for pair in ranges):
            raise ValueError("command_ranges must be three finite ordered pairs")

    def __init__(self, config, default_positions, num_envs, device, dt, seed):
        self.config = config
        self.default = default_positions
        self.device = device
        self.dt = dt
        self.commands = torch.zeros(num_envs, 3, device=device)
        # CPU RNG also works when the simulation uses Metal (MPS has no torch.Generator).
        self.generator = torch.Generator().manual_seed(seed)
        self.ranges = torch.tensor(config["command_ranges"], device=device).expand(num_envs, -1, -1).clone()
        self.resample_steps = max(1, round(config["command_resample_seconds"] / dt))

    def reset(self, env_ids, state, updates):
        self.resample(env_ids, updates)

    def on_step_end(self, state, episode_steps):
        ids = ((episode_steps > 0) & (episode_steps % self.resample_steps == 0)).nonzero(as_tuple=False).flatten()
        self.resample(ids)

    def context(self):
        return {"commands": self.commands}

    def resample(self, env_ids, updates=None):
        updates = updates or {}
        if updates.keys() - {"command_ranges"}:
            raise NotImplementedError("Curriculum currently supports command_ranges updates only")
        if "command_ranges" in updates:
            ranges = torch.tensor(updates["command_ranges"], device=self.device)
            if ranges.shape != (3, 2) or (ranges[:, 1] < ranges[:, 0]).any():
                raise ValueError("command_ranges must be three ordered [min, max] pairs")
            self.ranges[env_ids] = ranges
        sample = torch.rand(len(env_ids), 3, generator=self.generator).to(self.device)
        ranges = self.ranges[env_ids]
        self.commands[env_ids] = ranges[:, :, 0] + sample * (ranges[:, :, 1] - ranges[:, :, 0])

    def body_state(self, state):
        gravity = torch.zeros_like(state.linear_velocity)
        gravity[:, 2] = -1
        return (rotate_inverse(state.quaternion, state.linear_velocity),
                rotate_inverse(state.quaternion, state.angular_velocity),
                rotate_inverse(state.quaternion, gravity))

    def observe(self, state, previous_action):
        angular = rotate_inverse(state.quaternion, state.angular_velocity)
        return policy_observation(state.quaternion, angular, state.joint_position, state.joint_velocity,
                                  self.commands, previous_action, self.default, self.config["observation_scales"])

    def reward(self, state, action, previous_action, torque):
        linear, angular, gravity = self.body_state(state)
        sigma = self.config["tracking_sigma"]
        terms = {
            "tracking_linear": torch.exp(-((linear[:, :2] - self.commands[:, :2])**2).sum(-1) / sigma),
            "tracking_yaw": torch.exp(-(angular[:, 2] - self.commands[:, 2])**2 / sigma),
            "vertical_velocity": linear[:, 2]**2,
            "roll_pitch_velocity": (angular[:, :2]**2).sum(-1),
            "orientation": (gravity[:, :2]**2).sum(-1),
            "height": (state.position[:, 2] - self.config["target_height"])**2,
            "torque": (torque**2).sum(-1),
            "action_rate": ((action - previous_action)**2).sum(-1),
        }
        # Integrate reward rates over physical time, independent of policy frequency.
        return sum(terms[name] * weight for name, weight in self.config["reward_weights"].items()) * self.dt

    def terminated(self, state):
        _, _, gravity = self.body_state(state)
        return ((state.position[:, 2] < self.config["termination_min_height"])
                | (-gravity[:, 2] < self.config["termination_min_upright"]))
