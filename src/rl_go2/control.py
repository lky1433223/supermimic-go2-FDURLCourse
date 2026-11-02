import math

import torch


class JointPositionControl:
    """Identical external PD controller in each physics substep, on both engines."""

    def __init__(self, config, device):
        self.config = config
        self.default = torch.tensor(config["default_joint_positions"], device=device)
        self.limits = torch.tensor(config["torque_limits"], device=device)
        self.action_lower = torch.full_like(self.default, -config["action_clip"])
        self.action_upper = torch.full_like(self.default, config["action_clip"])
        if "joint_position_limits" in config:
            ranges = config["joint_position_limits"]
            margin = config.get("joint_limit_margin", 0.0)
            if set(ranges) != set(config["joint_names"]) or not math.isfinite(margin) or margin < 0:
                raise ValueError("Joint limits must cover joint_names and use a finite nonnegative margin")
            pairs = [ranges[name] for name in config["joint_names"]]
            if any(len(pair) != 2 or not all(math.isfinite(x) for x in pair)
                   or pair[0] + margin >= pair[1] - margin for pair in pairs):
                raise ValueError("Joint limits must be finite ordered pairs with room for the margin")
            bounds = self.default.new_tensor(pairs)
            lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
            if ((self.default < lower) | (self.default > upper)).any():
                raise ValueError("Default joint positions must lie inside the joint limit margin")
            self.action_lower = torch.maximum(self.action_lower, (lower - self.default) / config["action_scale"])
            self.action_upper = torch.minimum(self.action_upper, (upper - self.default) / config["action_scale"])

    def target(self, action):
        # Return the applied action too: observations, replay and imagination share its semantics.
        action = action.clamp(self.action_lower, self.action_upper)
        return self.default + self.config["action_scale"] * action, action

    def torque(self, target, state):
        torque = self.config["kp"] * (target - state.joint_position) - self.config["kd"] * state.joint_velocity
        return torch.clamp(torque, -self.limits, self.limits)
