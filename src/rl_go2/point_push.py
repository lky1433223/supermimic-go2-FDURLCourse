"""Batched point-push events shared by the simulator and imagination."""

from dataclasses import dataclass
import math
from typing import NamedTuple
from xml.etree import ElementTree

import torch

from rl_go2.backends import LinkPointForces
from rl_go2.state import RobotState, rotate_inverse, quat_mul


@dataclass(frozen=True)
class PointPushConfig:
    """Validated sampling ranges in seconds, degrees, and newtons."""

    first_wait: tuple[float, float]
    gap: tuple[float, float]
    duration: tuple[float, float]
    clean_probability: float
    base_probability: float
    force_ranges: tuple[tuple[float, float], tuple[float, float]]
    azimuth: tuple[float, float]
    elevation: tuple[float, float]
    overlap_probability: float
    overlap_delay: tuple[int, int]

    @classmethod
    def from_config(cls, config: dict) -> "PointPushConfig | None":
        """Read the optional point_push domain randomization section."""
        section = config.get("domain_randomization", {}).get("point_push", {})
        if not isinstance(section, dict) or type(section.get("enabled", False)) is not bool:
            raise ValueError("point_push must be a mapping with a boolean enabled flag")
        if not section.get("enabled", False):
            return None
        targets = section.get("targets")
        direction = section.get("direction")
        overlap = section.get("overlap", {"probability": 0.0, "second_start_delay_steps": [1, 1]})
        if not isinstance(targets, dict) or set(targets) != {"base", "foot"}:
            raise ValueError("point_push targets must define base and foot")
        if not all(isinstance(value, dict) for value in (*targets.values(), direction, overlap)):
            raise ValueError("point_push targets, direction and overlap must be mappings")
        base, foot = targets["base"], targets["foot"]
        base_probability = _probability(base.get("probability"), "base probability")
        if not math.isclose(base_probability + _probability(foot.get("probability"), "foot probability"), 1.0):
            raise ValueError("point_push base and foot probabilities must sum to 1")
        duration = section.get("duration_seconds")
        if _finite_number(duration):
            duration = (duration, duration)
        duration = _bounds(duration, "duration_seconds", 0)
        if duration[0] <= 0:
            raise ValueError("point_push duration_seconds must be positive")
        delay = _bounds(overlap.get("second_start_delay_steps"), "second_start_delay_steps", 1)
        if not all(type(value) is int for value in delay):
            raise ValueError("point_push second_start_delay_steps must be integers")
        return cls(
            _bounds(section.get("first_wait_seconds"), "first_wait_seconds", 0),
            _bounds(section.get("gap_seconds"), "gap_seconds", 0),
            duration,
            _probability(section.get("no_push_episode_probability"), "no_push_episode_probability"),
            base_probability,
            tuple(_bounds(target.get("force_range_n"), "force_range_n", 0) for target in (base, foot)),
            _bounds(direction.get("azimuth_range_deg"), "azimuth_range_deg", 0, 360),
            _bounds(direction.get("elevation_range_deg"), "elevation_range_deg", -90, 90),
            _probability(overlap.get("probability"), "overlap probability"),
            delay,
        )


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and abs(value) <= 1e30


def _probability(value: object, name: str) -> float:
    if not _finite_number(value) or not 0 <= value <= 1:
        raise ValueError(f"point_push {name} must be between 0 and 1")
    return float(value)


def _bounds(value: object, name: str, minimum: float, maximum: float = math.inf) -> tuple:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(_finite_number(item) for item in value)
            or not minimum <= value[0] <= value[1] <= maximum):
        raise ValueError(f"point_push {name} must be an ordered pair in [{minimum}, {maximum}]")
    return tuple(value)


class PushState(NamedTuple):
    """Per-environment group clock and per-link point-load state in model link order."""

    wait: torch.Tensor
    delay: torch.Tensor
    duration_steps: torch.Tensor
    points: torch.Tensor
    forces: torch.Tensor

    def replay_fields(self) -> dict[str, torch.Tensor]:
        """Return fields outside policy and critic context."""
        return {f"push/{name}": value for name, value in self._asdict().items()}

    @classmethod
    def from_replay(cls, record: dict[str, torch.Tensor]) -> "PushState":
        """Restore sampled push state at the start of a transition."""
        return cls(*(record[f"push/{name}"] for name in cls._fields))


class Go2PushGeometry:
    """Link orientations and collision surfaces for the Go2 MJCF push targets."""

    def __init__(self, config: dict, feet: tuple[str, ...], device: torch.device) -> None:
        robot = config["robot"]
        path = next(entity["file"] for entity in config["scene"]["entities"] if entity.get("robot"))
        asset = ElementTree.parse(path).getroot()
        base = asset.find(f"./worldbody/body[@name='{robot['base_link']}']")
        box = None if base is None else base.find("./geom[@class='collision'][@type='box']")
        foot = asset.find(".//default[@class='foot']/geom")
        if box is None or foot is None or any(key in box.attrib for key in ("pos", "quat", "euler")):
            raise ValueError("point_push requires the Go2 base collision box and spherical foot geometry")
        self.base_half_size = torch.tensor([float(x) for x in box.attrib["size"].split()], device=device)
        self.foot_center = torch.tensor([float(x) for x in foot.attrib["pos"].split()], device=device)
        self.foot_radius = float(foot.attrib["size"])
        # Python index sequences would create CPU tensors inside a captured rollout.
        self.joint_indices = torch.tensor(
            [robot["joint_names"].index(f"{leg}_{joint}_joint")
             for leg in feet for joint in ("hip", "thigh", "calf")],
            dtype=torch.long, device=device,
        )

    def link_quaternions(self, state: RobotState) -> torch.Tensor:
        """Return authored base and calf world orientations (batch, 5, 4), wxyz."""
        # The Go2 MJCF has identity rest rotations and X/Y/Y hinge axes.
        leg_angles = state.joint_position.index_select(1, self.joint_indices).unflatten(1, (-1, 3))
        hip, thigh, calf = leg_angles.unbind(dim=-1)
        hip_half = 0.5 * hip
        knee_chain = 0.5 * (thigh + calf)
        hc, hs = hip_half.cos(), hip_half.sin()
        kc, ks = knee_chain.cos(), knee_chain.sin()
        calf_local = torch.stack((hc * kc, hs * kc, hc * ks, hs * ks), dim=-1)
        calf_world = quat_mul(state.quaternion[:, None, :], calf_local)
        return torch.cat((state.quaternion[:, None, :], calf_world), dim=1)

    def sample_points(self, direction_local: torch.Tensor) -> torch.Tensor:
        """Sample local box or sphere points on the side facing the incoming force."""
        base_direction = direction_local[:, 0, :2]
        axis = base_direction.abs().argmax(dim=-1, keepdim=True)
        base = (2 * torch.rand_like(direction_local[:, 0]) - 1) * self.base_half_size
        side = -base_direction.gather(1, axis).sign() * self.base_half_size[axis]
        base = base.scatter(1, axis, side)
        direction = direction_local[:, 1:]
        azimuth = 2 * math.pi * torch.rand_like(direction[..., 0])
        z = 2 * torch.rand_like(azimuth) - 1
        radius = (1 - z.square()).clamp_min(0).sqrt()
        normal = torch.stack((radius * azimuth.cos(), radius * azimuth.sin(), z), dim=-1)
        normal = torch.where((normal * direction).sum(dim=-1, keepdim=True) > 0, -normal, normal)
        feet = self.foot_center + self.foot_radius * normal
        return torch.cat((base[:, None, :], feet), dim=1)


class PointPushScheduler:
    """Sample at onset and advance the same discrete event clock in both rollouts."""

    def __init__(self, spec: PointPushConfig, dt: float, geometry: Go2PushGeometry) -> None:
        self.spec = spec
        self.geometry = geometry
        self.duration_steps = tuple(max(1, round(seconds / dt)) for seconds in spec.duration)
        self.first_wait = tuple(round(seconds / dt) for seconds in spec.first_wait)
        self.gap = tuple(round(seconds / dt) for seconds in spec.gap)
        self.idle = -self.duration_steps[1]
        foot_probability = (1 - spec.base_probability) / 4
        self.link_weights = geometry.base_half_size.new_tensor([spec.base_probability, *([foot_probability] * 4)])
        if spec.overlap_probability and spec.overlap_delay[1] >= self.duration_steps[0]:
            raise ValueError("point_push overlap delay must be shorter than the minimum push duration")

    def reset(self, positions: torch.Tensor) -> PushState:
        """Sample independent first waits and clean episodes for the given batch."""
        batch = positions.shape[0]
        wait = torch.randint(self.first_wait[0], self.first_wait[1] + 1, (batch,), device=positions.device)
        wait = torch.where(torch.rand_like(positions[:, 0]) < self.spec.clean_probability, -1, wait)
        delay = torch.full((batch, 5), self.idle, dtype=torch.long, device=positions.device)
        empty = positions.new_zeros((batch, 5, 3))
        return PushState(wait, delay, torch.zeros_like(wait), empty, empty)

    def _schedule_group(self, state: PushState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        due = state.wait == 0
        device = state.wait.device
        weights = self.link_weights
        primary = self._sample_link(weights.expand(state.wait.shape[0], -1))
        delay = torch.full_like(state.delay, self.idle).scatter(1, primary, 0)
        extra = torch.zeros_like(state.wait)
        if self.spec.overlap_probability:
            slots = torch.arange(5, device=device)
            remaining = weights * (slots != primary)
            remaining = torch.where(remaining.sum(dim=1, keepdim=True) > 0, remaining, (slots != primary).float())
            secondary = self._sample_link(remaining)
            extra = torch.randint(*_inclusive(self.spec.overlap_delay), state.wait.shape, device=device)
            extra = torch.where(torch.rand_like(state.forces[:, 0, 0]) < self.spec.overlap_probability, extra, 0)
            delay = delay.scatter(1, secondary, torch.where(extra > 0, extra, self.idle)[:, None])
        gap = torch.randint(*_inclusive(self.gap), state.wait.shape, device=device)
        minimum, maximum = self.duration_steps
        if minimum == maximum:
            sampled_duration = torch.full_like(state.wait, minimum)
        else:
            sampled_duration = torch.randint(minimum, maximum + 1, state.wait.shape, device=device)
        duration = torch.where(due, sampled_duration, state.duration_steps)
        wait = torch.where(due, gap + duration + extra, state.wait)
        return wait, torch.where(due[:, None], delay, state.delay), duration

    @staticmethod
    def _sample_link(weights: torch.Tensor) -> torch.Tensor:
        cumulative = weights.cumsum(dim=1)
        threshold = torch.rand_like(cumulative[:, :1]) * cumulative[:, -1:]
        return (threshold >= cumulative).sum(dim=1, keepdim=True).clamp_max(weights.shape[1] - 1)

    def _sample_forces(self, reference: torch.Tensor) -> torch.Tensor:
        azimuth = _uniform_like(reference[..., 0], self.spec.azimuth).deg2rad()
        elevation = _uniform_like(reference[..., 0], self.spec.elevation).deg2rad()
        horizontal = elevation.cos()
        direction = torch.stack((horizontal * azimuth.cos(), horizontal * azimuth.sin(), elevation.sin()), dim=-1)
        magnitude = torch.cat((_uniform_like(azimuth[:, :1], self.spec.force_ranges[0]),
                               _uniform_like(azimuth[:, 1:], self.spec.force_ranges[1])), dim=1)
        return direction * magnitude[..., None]

    def prepare(self, state: PushState, link_quat: torch.Tensor) -> PushState:
        """Sample loads starting on this control step and preserve ongoing loads."""
        wait, delay, duration = self._schedule_group(state)
        starting = (delay == 0)[..., None]
        force = self._sample_forces(state.forces)
        point = self.geometry.sample_points(rotate_inverse(link_quat, force))
        return PushState(wait, delay, duration, torch.where(starting, point, state.points),
                         torch.where(starting, force, state.forces))

    def advance(self, state: PushState) -> PushState:
        """Advance by one control step without mutating the rollout seed."""
        return PushState((state.wait - 1).clamp_min(-1), (state.delay - 1).clamp_min(self.idle),
                         state.duration_steps, state.points, state.forces)

    def loads(self, state: PushState) -> LinkPointForces:
        """Return local points and active world forces in model link order."""
        active = ((state.delay <= 0) & (state.delay > -state.duration_steps[:, None]))[..., None]
        return LinkPointForces(state.points, torch.where(active, state.forces, 0.0))

    def wrenches(self, state: PushState, link_quat: torch.Tensor) -> torch.Tensor:
        """Return active world force/moment at the predicted link orientations."""
        return self.loads(state).wrenches_world(link_quat)


def _inclusive(bounds: tuple[int, int]) -> tuple[int, int]:
    return bounds[0], bounds[1] + 1


def _uniform_like(reference: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    low, high = bounds
    return low + (high - low) * torch.rand_like(reference)
