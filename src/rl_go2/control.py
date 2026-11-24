import torch

from rl_go2.config import joint_position_bounds


class JointPositionControl:
    """Identical external PD controller in each physics substep, on both engines."""

    def __init__(self, config, device):
        self.config = config
        self.default = torch.tensor(config["default_joint_positions"], device=device)
        self.limits = torch.tensor(config["torque_limits"], device=device)
        self.action_lower = torch.full_like(self.default, -config["action_clip"])
        self.action_upper = torch.full_like(self.default, config["action_clip"])
        pairs = joint_position_bounds(config)
        if pairs is not None:
            margin = config.get("joint_limit_margin", 0.0)
            bounds = self.default.new_tensor(pairs)
            lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
            self.action_lower = torch.maximum(self.action_lower, (lower - self.default) / config["action_scale"])
            self.action_upper = torch.minimum(self.action_upper, (upper - self.default) / config["action_scale"])

    def target(self, action):
        # Return the applied action too: observations, replay and imagination share its semantics.
        action = action.clamp(self.action_lower, self.action_upper)
        return self.default + self.config["action_scale"] * action, action

    def torque(self, target, state):
        torque = self.config["kp"] * (target - state.joint_position) - self.config["kd"] * state.joint_velocity
        return torch.clamp(torque, -self.limits, self.limits)
