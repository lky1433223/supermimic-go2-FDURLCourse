"""Optional posture, contact and failure rewards for Go2 walking."""

import math

import torch


def walk_config(config: dict) -> dict:
    """Resolve grouped walking settings while accepting saved legacy tasks."""
    if "termination" in config:
        termination = config["termination"]
        tilt = termination["max_tilt_deg"]
        if not 0 < tilt < 180:
            raise ValueError("termination.max_tilt_deg must be in (0, 180)")
        config = {**config, "termination_min_height": termination["min_base_height_m"],
                  "termination_min_upright": math.cos(math.radians(tilt))}
    if "reward" in config:
        reward = config["reward"]
        names = ("tracking_linear", "tracking_yaw", "vertical_velocity", "roll_pitch_velocity",
                 "orientation", "height", "torque", "action_rate")
        config = {**config, "target_height": reward["height"]["target_m"],
                  "tracking_sigma": reward["tracking_sigma"],
                  "reward_weights": {name: reward[name]["weight"] for name in names}}
    return config


class WalkRewards:
    """Shared extra terms for measured and model-predicted walking transitions."""

    def __init__(self, task: dict, robot: dict, default: torch.Tensor, dt: float) -> None:
        reward = task.get("reward", {})
        contact = task.get("termination", {}).get("contact_failure", {})
        posture = reward.get("posture_deviation", {})
        gait = reward.get("gait_mismatch", {})
        self.contact_enabled = contact.get("enabled", False)
        self.force_threshold_n = contact.get("force_threshold_n", 10.0)
        self.exempt = tuple(contact.get("exempt", ("foot", "lower_calf")))
        self.failure_weight = reward.get("failure", {}).get("weight", 0.0)
        self.posture_weight = posture.get("weight", 0.0)
        self.gait_weight = gait.get("weight", 0.0)
        self.dt = dt
        if type(self.contact_enabled) is not bool or set(self.exempt) - {"foot", "lower_calf"}:
            raise ValueError("Invalid contact_failure enabled flag or exemptions")
        if not math.isfinite(self.force_threshold_n) or self.force_threshold_n <= 0:
            raise ValueError("contact_failure.force_threshold_n must be finite and positive")
        if any(not math.isfinite(weight) or weight > 0
               for weight in (self.failure_weight, self.posture_weight, self.gait_weight)):
            raise ValueError("Failure, posture and gait weights must be finite and nonpositive")
        if self.posture_weight:
            if posture.get("reference", "reset_pose") != "reset_pose":
                raise ValueError("posture_deviation.reference must be reset_pose")
            tolerance = posture["tolerance_rad"]
            if set(tolerance) != {"hip", "thigh", "calf"} or any(
                not math.isfinite(angle) or angle < 0 for angle in tolerance.values()
            ):
                raise ValueError("Set finite nonnegative hip, thigh and calf tolerances")
            joint_types = [name.split("_")[-2] for name in robot["joint_names"]]
            self.tolerance = default.new_tensor([tolerance[name] for name in joint_types])
            self.reference = default.new_tensor(robot["reset_joint_positions"]) if (
                "reset_joint_positions" in robot
            ) else default
        if self.gait_weight:
            self.period_s = gait["period_s"]
            self.stance_fraction = gait["stance_fraction"]
            self.command_threshold = gait["command_threshold"]
            if (gait["pattern"] != "trot" or not math.isfinite(self.period_s) or self.period_s <= 0
                    or not 0 < self.stance_fraction < 1 or not math.isfinite(self.command_threshold)
                    or self.command_threshold < 0):
                raise ValueError("Invalid gait_mismatch parameters")
            self.phase_offsets = default.new_tensor([0.0, 0.5, 0.5, 0.0])  # FR, FL, RR, RL.

    @property
    def requires_contacts(self) -> bool:
        """Whether the simulator must supply contact observations."""
        return self.contact_enabled or bool(self.gait_weight)

    def extra_reward(
        self, joint_position: torch.Tensor, failure: torch.Tensor, *,
        foot_contact: torch.Tensor | None = None, episode_step: torch.Tensor | None = None,
        commands: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score batched joint angles, failure indicators and optional foot probabilities."""
        reward = failure.to(joint_position.dtype) * self.failure_weight
        if self.posture_weight:
            excess = ((joint_position - self.reference).abs() - self.tolerance).clamp_min(0)
            reward = reward + self.dt * self.posture_weight * excess.square().mean(dim=-1)
        if self.gait_weight:
            assert foot_contact is not None and episode_step is not None and commands is not None
            phase = (episode_step[:, None] * (self.dt / self.period_s) + self.phase_offsets) % 1.0
            stance = phase < self.stance_fraction
            mismatch = torch.where(stance, 1.0 - foot_contact, foot_contact).mean(dim=-1)
            active = commands[:, :2].norm(dim=-1) + commands[:, 2].abs() > self.command_threshold
            reward = reward + self.dt * self.gait_weight * mismatch * active
        return reward
