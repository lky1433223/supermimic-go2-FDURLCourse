import torch


class JointPositionControl:
    """Identical external PD controller in each physics substep, on both engines."""

    def __init__(self, config, device):
        self.config = config
        self.default = torch.tensor(config["default_joint_positions"], device=device)
        self.limits = torch.tensor(config["torque_limits"], device=device)

    def target(self, action):
        action = action.clamp(-self.config["action_clip"], self.config["action_clip"])
        return self.default + self.config["action_scale"] * action, action

    def torque(self, target, state):
        torque = self.config["kp"] * (target - state.joint_position) - self.config["kd"] * state.joint_velocity
        return torch.clamp(torque, -self.limits, self.limits)
