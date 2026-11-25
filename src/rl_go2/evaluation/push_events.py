"""Generate and apply reproducible point forces for evaluation trials."""
from dataclasses import dataclass
import math
from random import Random
from typing import TYPE_CHECKING, Literal

import torch

from rl_go2.backends import LinkPointForces
from rl_go2.config import FOOT_NAMES
from rl_go2.point_push import Go2PushGeometry

if TYPE_CHECKING:
    from rl_go2.env import QuadrupedEnv
    from .task import EvaluationPlan, Trial


@dataclass(frozen=True)
class PushSpec:
    """Force magnitude, duration, and target class for an evaluation condition."""

    target: str
    force_n: float
    duration_seconds: float = 0.20
    foot_draw: int = 0
    trigger_height_map: str | None = None


@dataclass(frozen=True)
class PushEvent:
    """A local contact point and a force in world or onset-heading coordinates."""

    start_step: int | None
    duration_steps: int
    link_name: str
    point_local_m: tuple[float, float, float]
    force_n: tuple[float, float, float]
    force_frame: Literal['world', 'heading']
    entry_bounds_xy_m: tuple[float, float, float, float] | None = None

    @property
    def end_step(self) -> int | None:
        return None if self.start_step is None else self.start_step + self.duration_steps

    def ready(self, step: int, position: torch.Tensor) -> bool:
        """Trigger at the scheduled step or on first entry of the base into the region."""
        if self.entry_bounds_xy_m is None:
            return step == self.start_step
        x_min, x_max, y_min, y_max = self.entry_bounds_xy_m
        x, y = position[:2].tolist()
        return x_min <= x < x_max and y_min <= y < y_max

    def loads(self, env: 'QuadrupedEnv') -> LinkPointForces:
        """Resolve force in world coordinates at push onset and hold it fixed."""
        link_names = env.backend.link_names
        index = link_names.index(self.link_name)
        points = env.state.position.new_zeros((env.num_envs, len(link_names), 3))
        forces = torch.zeros_like(points)
        points[:, index] = points.new_tensor(self.point_local_m)
        force = forces.new_tensor(self.force_n)
        if self.force_frame == 'heading':
            w, x, y, z = env.state.quaternion.unbind(dim=-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
            force = torch.stack((-force[1] * yaw.sin(), force[1] * yaw.cos(), torch.zeros_like(yaw)), dim=-1)
        forces[:, index] = force
        return LinkPointForces(points, forces)


def sample_event(config: dict, geometry: Go2PushGeometry, dt: float, trial: 'Trial', seed: int) -> PushEvent:
    """Sample in reference link coordinates so placement is independent of policy."""
    spec = trial.condition.point_push
    if spec is None:
        raise ValueError('A push event requires a point-push condition')
    sample_key = f'point-push:{seed}:{trial.name}:{spec.target}'
    rng = Random(sample_key)
    if spec.target == 'base':
        index = 0
        direction = (0.0, 1.0, 0.0)
    else:
        indices = list(range(1, len(FOOT_NAMES) + 1))
        rng.shuffle(indices)
        index = indices[spec.foot_draw]
        rng = Random(f'{sample_key}:{spec.foot_draw}')
        azimuth = math.radians(rng.uniform(0, 360))
        elevation = math.radians(rng.uniform(-25, 25))
        direction = (math.cos(elevation) * math.cos(azimuth),
                     math.cos(elevation) * math.sin(azimuth), math.sin(elevation))
    link = config['robot']['base_link'] if index == 0 else f'{FOOT_NAMES[index - 1]}_calf'
    directions = torch.tensor(direction, device='cpu').expand(1, len(FOOT_NAMES) + 1, 3)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(rng.getrandbits(63))
        point = tuple(geometry.sample_points(directions)[0, index].tolist())

    duration_steps = max(1, round(spec.duration_seconds / dt))
    entry_bounds = None
    start = None
    if spec.trigger_height_map is None:
        stop = trial.motion.stop_at_seconds
        start = round((5.0 if stop is None else stop + 2.0) / dt)
    else:
        height_map = config['scene'].get('height_maps', {}).get(spec.trigger_height_map)
        if height_map is None:
            raise ValueError(f'Point push requires height map {spec.trigger_height_map!r} in the trial scene')
        heights = height_map['height_map_m']
        tile = height_map['tile_size_m']
        x_min, y_center = height_map['start_x_m'], height_map['center_y_m']
        half_width = len(heights[0]) * tile / 2
        entry_bounds = (x_min, x_min + len(heights) * tile, y_center - half_width, y_center + half_width)
    if (start or 0) + duration_steps >= math.ceil(trial.motion.duration_seconds / dt):
        raise ValueError(f'{trial.name} must leave time for recovery after the push')
    return PushEvent(start, duration_steps, link, point, tuple(axis * spec.force_n for axis in direction),
                     'heading' if index == 0 else 'world', entry_bounds)


def build_push_events(config: dict, plan: 'EvaluationPlan', seeds: list[int]) -> dict[tuple, PushEvent]:
    """Resolve all declared pushes before execution for reporting and replay."""
    trials = [trial for trial in plan.trials if trial.condition.point_push is not None]
    if not trials:
        return {}
    geometries = {}
    events = {}
    for trial in trials:
        trial_config = plan.scene_config(config, trial)
        if trial.scene not in geometries:
            geometries[trial.scene] = Go2PushGeometry(trial_config, FOOT_NAMES, torch.device('cpu'))
        dt = trial_config['scene']['sim_options']['dt'] * trial_config['robot']['control_decimation']
        for seed in seeds:
            events[trial.event_key(seed)] = sample_event(trial_config, geometries[trial.scene], dt, trial, seed)
    return events
