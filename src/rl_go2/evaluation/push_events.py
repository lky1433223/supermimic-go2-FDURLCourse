"""Generate and apply reproducible point forces for evaluation trials."""
from dataclasses import dataclass
import math
from random import Random
from typing import TYPE_CHECKING

import torch

from rl_go2.backends import LinkPointForces
from rl_go2.config import FOOT_NAMES
from rl_go2.point_push import Go2PushGeometry

if TYPE_CHECKING:
    from rl_go2.env import QuadrupedEnv
    from .task import EvaluationPlan, Trial


@dataclass(frozen=True)
class PushSpec:
    """Force magnitude and target class declared by an evaluation condition."""

    target: str
    force_n: float


@dataclass(frozen=True)
class PushEvent:
    """A sampled local point and world force, shared by paired trials."""

    start_step: int
    end_step: int
    link_name: str
    point_local_m: tuple[float, float, float]
    force_world_n: tuple[float, float, float]

    def loads(self, env: 'QuadrupedEnv') -> LinkPointForces:
        link_names = env.backend.link_names
        index = link_names.index(self.link_name)
        points = env.state.position.new_zeros((env.num_envs, len(link_names), 3))
        forces = torch.zeros_like(points)
        points[:, index] = points.new_tensor(self.point_local_m)
        forces[:, index] = forces.new_tensor(self.force_world_n)
        return LinkPointForces(points, forces)


def sample_event(config: dict, geometry: Go2PushGeometry, dt: float, trial: 'Trial', seed: int) -> PushEvent:
    """Sample in reference link coordinates so placement is independent of policy."""
    spec = trial.condition.point_push
    if spec is None:
        raise ValueError('A push event requires a point-push condition')
    rng = Random(f'point-push:{seed}:{trial.name}:{spec.target}')
    azimuth = math.radians(rng.uniform(0, 360))
    elevation = math.radians(rng.uniform(-25, 25))
    direction = (math.cos(elevation) * math.cos(azimuth),
                 math.cos(elevation) * math.sin(azimuth), math.sin(elevation))
    index = 0 if spec.target == 'base' else rng.randrange(1, len(FOOT_NAMES) + 1)
    link = config['robot']['base_link'] if index == 0 else f'{FOOT_NAMES[index - 1]}_calf'
    directions = torch.tensor(direction, device='cpu').expand(1, len(FOOT_NAMES) + 1, 3)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(rng.getrandbits(63))
        point = tuple(geometry.sample_points(directions)[0, index].tolist())

    stop = trial.motion.stop_at_seconds
    start = round((5.0 if stop is None else stop + 2.0) / dt)
    end = start + max(1, round(0.20 / dt))
    if end >= math.ceil(trial.motion.duration_seconds / dt):
        raise ValueError(f'{trial.name} must leave time for recovery after the push')
    return PushEvent(start, end, link, point, tuple(axis * spec.force_n for axis in direction))


def build_push_events(config: dict, plan: 'EvaluationPlan', seeds: list[int]) -> dict[tuple, PushEvent]:
    """Resolve all declared pushes before execution for reporting and replay."""
    trials = [trial for trial in plan.trials if trial.condition.point_push is not None]
    if not trials:
        return {}
    geometry = Go2PushGeometry(config, FOOT_NAMES, torch.device('cpu'))
    dt = config['scene']['sim_options']['dt'] * config['robot']['control_decimation']
    return {
        (seed, trial.condition.name, trial.name): sample_event(config, geometry, dt, trial, seed)
        for seed in seeds for trial in trials
    }
