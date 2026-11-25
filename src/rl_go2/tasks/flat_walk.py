import math

import torch

from rl_go2.config import MODEL_PARAMETER_KEYS

from rl_go2.state import rotate_inverse
from rl_go2.observations import OBSERVATION_TERMS, SensorReadings, policy_observation
from rl_go2.state import RobotState
from rl_go2.tasks.walk_rewards import walk_config


class FlatWalkTask:
    """Minimal velocity tracking task; reward coefficients are initial values, not tuned results."""

    observation_terms = OBSERVATION_TERMS

    @staticmethod
    def validate_config(config, robot):
        config = walk_config(config)
        if len(robot["joint_names"]) != 12:
            raise ValueError("flat_walk expects a 12-joint quadruped")
        for key in ("command_resample_seconds", "tracking_sigma"):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        ranges = config["command_ranges"]
        if len(ranges) != 3 or any(len(pair) != 2 or pair[1] < pair[0]
                                   or not all(math.isfinite(x) for x in pair) for pair in ranges):
            raise ValueError("command_ranges must be three finite ordered pairs")
        sampling = config.get("command_sampling", {})
        keys = {"stand_probability", "turn_in_place_probability"}
        if (not isinstance(sampling, dict) or sampling.keys() - keys
                or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in sampling.values())
                or sum(sampling.values()) > 1):
            raise ValueError("command_sampling probabilities must be in [0, 1] and sum to at most 1")
        if sampling.get("turn_in_place_probability", 0) and ranges[2][0] == ranges[2][1] == 0:
            raise ValueError("Turn-in-place sampling requires a nonzero yaw command in command_ranges")

    def __init__(self, config, default_positions, num_envs, device, dt, seed):
        self.config = walk_config(config)
        self.default = default_positions
        self.device = device
        self.dt = dt
        self.commands = torch.zeros(num_envs, 3, device=device)
        scales = config["observation_scales"]
        self.observation_scales = {**scales, "commands": self.commands.new_tensor(scales["commands"])}
        # CPU RNG also works when the simulation uses Metal (MPS has no torch.Generator).
        self.generator = torch.Generator().manual_seed(seed)
        self.ranges = torch.tensor(config["command_ranges"], device=device).expand(num_envs, -1, -1).clone()
        sampling = config.get("command_sampling", {})
        self.stand_probability = sampling.get("stand_probability", 0)
        self.turn_in_place_probability = sampling.get("turn_in_place_probability", 0)
        self.resample_steps = max(1, round(config["command_resample_seconds"] / dt))
        self.command_steps_left = torch.full((num_envs, 1), self.resample_steps, device=device, dtype=torch.long)

    def reset(self, env_ids, state, updates):
        self.resample(env_ids, updates)
        self.command_steps_left[env_ids] = self.resample_steps

    def on_step_end(self, state, episode_steps):
        ids = ((episode_steps > 0) & (episode_steps % self.resample_steps == 0)).nonzero(as_tuple=False).flatten()
        self.resample(ids)
        self.command_steps_left[:, 0] = self.resample_steps - episode_steps % self.resample_steps

    def context(self):
        return {"commands": self.commands, "command_steps_left": self.command_steps_left,
                "command_ranges": self.ranges}

    def external_force(self, state, episode_steps):
        return None

    def resample(self, env_ids, updates=None):
        updates = updates or {}
        if updates.keys() - {"command_ranges"}:
            raise NotImplementedError("Curriculum currently supports command_ranges updates only")
        if "command_ranges" in updates:
            ranges = self.ranges.new_tensor(updates["command_ranges"])
            if ranges.shape != (3, 2) or (ranges[:, 1] < ranges[:, 0]).any():
                raise ValueError("command_ranges must be three ordered [min, max] pairs")
            if self.turn_in_place_probability and not ranges[2].any():
                raise ValueError("Turn-in-place sampling requires a nonzero yaw command in command_ranges")
            self.ranges[env_ids] = ranges
        self.commands[env_ids] = self._sample_commands(self.ranges[env_ids], generator=self.generator)

    def _sample_commands(self, ranges: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample [vx, vy, yaw_rate] from per-environment ranges shaped [batch, 3, 2]."""
        special_probability = self.stand_probability + self.turn_in_place_probability
        width = 4 if special_probability else 3
        device = ranges.device if generator is None else "cpu"
        draws = torch.rand((ranges.shape[0], width), device=device, generator=generator).to(ranges.device)
        low, high = ranges.unbind(dim=-1)
        commands = low + draws[:, :3] * (high - low)
        if not special_probability:
            return commands
        mode = draws[:, 3:4]
        # A discrete RNG can hit exactly zero inside a nonzero yaw range.
        yaw = commands[:, 2:3]
        yaw_endpoint = torch.where(high[:, 2:3] != 0, high[:, 2:3], low[:, 2:3])
        turn = torch.cat((torch.zeros_like(commands[:, :2]), torch.where(yaw != 0, yaw, yaw_endpoint)), dim=-1)
        commands = torch.where(mode < special_probability, turn, commands)
        return torch.where(mode < self.stand_probability, 0.0, commands)

    def body_state(self, state):
        gravity = torch.zeros_like(state.linear_velocity)
        gravity[:, 2] = -1
        return (rotate_inverse(state.quaternion, state.linear_velocity),
                rotate_inverse(state.quaternion, state.angular_velocity),
                rotate_inverse(state.quaternion, gravity))

    def observe(self, state: RobotState, previous_action: torch.Tensor,
                context: dict[str, torch.Tensor] | None = None, *,
                readings: SensorReadings | None = None) -> torch.Tensor:
        """Build the actor input from optional simulated sensor readings."""
        commands = self.commands if context is None else context["commands"]
        if readings is None:
            readings = SensorReadings.from_state(state)
        return policy_observation(readings.attitude, readings.angular_velocity, readings.joint_position,
                                  readings.joint_velocity, commands, previous_action, self.default,
                                  self.observation_scales, projected_gravity=readings.projected_gravity)

    def reward(self, state, action, previous_action, torque, context=None, *,
               base_height_m: torch.Tensor | None = None):
        commands = self.commands if context is None else context["commands"]
        linear, angular, gravity = self.body_state(state)
        sigma = self.config["tracking_sigma"]
        height = state.position[:, 2] if base_height_m is None else base_height_m
        terms = {
            "tracking_linear": torch.exp(-((linear[:, :2] - commands[:, :2])**2).sum(-1) / sigma),
            "tracking_yaw": torch.exp(-(angular[:, 2] - commands[:, 2])**2 / sigma),
            "vertical_velocity": linear[:, 2]**2,
            "roll_pitch_velocity": (angular[:, :2]**2).sum(-1),
            "orientation": (gravity[:, :2]**2).sum(-1),
            "height": (height - self.config["target_height"])**2,
            "torque": (torque**2).sum(-1),
            "action_rate": ((action - previous_action)**2).sum(-1),
        }
        # Integrate reward rates over physical time, independent of policy frequency.
        return sum(terms[name] * weight for name, weight in self.config["reward_weights"].items()) * self.dt

    def terminated(self, state, *, base_height_m: torch.Tensor | None = None):
        _, _, gravity = self.body_state(state)
        height = state.position[:, 2] if base_height_m is None else base_height_m
        return ((height < self.config["termination_min_height"])
                | (-gravity[:, 2] < self.config["termination_min_upright"]))

    def model_features(self, state, context):
        """Dynamics-only features; terrain tasks can provide local scans/physics parameters."""
        parameters = [context[key] for key in MODEL_PARAMETER_KEYS if key in context]
        if parameters:
            return torch.cat(parameters, dim=-1)
        return state.position.new_zeros((state.position.shape[0], 0))

    def advance_context(self, context, next_state):
        """Pure batched goal update for imagination; never mutates the simulator task."""
        left = context["command_steps_left"] - 1
        commands = self._sample_commands(context["command_ranges"])
        return {**context, "commands": torch.where(left <= 0, commands, context["commands"]),
                "command_steps_left": torch.where(left <= 0, self.resample_steps, left)}

    def termination_probability(self, state, context, height_softness, upright_softness, *,
                                base_height_m: torch.Tensor | None = None):
        _, _, gravity = self.body_state(state)
        height = state.position[:, 2] if base_height_m is None else base_height_m
        low = torch.sigmoid((self.config["termination_min_height"] - height) / height_softness)
        tilted = torch.sigmoid((self.config["termination_min_upright"] + gravity[:, 2]) / upright_softness)
        return 1 - (1 - low) * (1 - tilted)

    def metrics(self, state, context):
        linear, angular, _ = self.body_state(state)
        error = torch.cat((linear[:, :2], angular[:, 2:3]), -1) - context["commands"]
        return {name: error[:, i].square() for i, name in enumerate(
            ("vx_error_squared", "vy_error_squared", "yaw_error_squared"))}
