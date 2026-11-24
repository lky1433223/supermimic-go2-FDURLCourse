"""Physical evaluation conditions, independent of robot motion tasks."""
from copy import deepcopy
from dataclasses import dataclass
from random import Random

from rl_go2.config import FOOT_NAMES, base_com_offset_ranges, base_mass_scale_range, foot_friction_ranges
from .push_events import PushSpec


@dataclass(frozen=True)
class FrictionPhase:
    duration_seconds: float
    multiplier: float


@dataclass(frozen=True)
class EvaluationCondition:
    name: str
    foot_ground_friction: dict | None = None
    base_mass: dict | None = None
    base_com: dict | None = None
    friction_phases: tuple[FrictionPhase, ...] = ()
    point_push: PushSpec | None = None

    def validate(self) -> None:
        """Validate all physical fields before loading the evaluated policy."""
        config = self.apply({})
        base_mass_scale_range(config)
        base_com_offset_ranges(config)
        ranges = foot_friction_ranges(config)
        if self.point_push is not None:
            if self.point_push.target not in ('base', 'foot'):
                raise ValueError('Point push target must be base or foot')
            if not 0 < self.point_push.force_n < float('inf'):
                raise ValueError('Point push force must be finite and positive')
            if not 0 < self.point_push.duration_seconds < float('inf'):
                raise ValueError('Point push duration must be finite and positive')
        if self.friction_phases:
            if ranges is None:
                raise ValueError('Friction phases require enabled foot friction')
            if any(lo != hi for foot in ranges[0] for lo, hi in foot):
                raise ValueError('Friction phases require fixed per-foot base values')
            for phase in self.friction_phases:
                if not 0 < phase.duration_seconds < float('inf') or not 0 < phase.multiplier < float('inf'):
                    raise ValueError('Phase duration and multiplier must be finite and positive')

    def apply(self, config: dict) -> dict:
        """Build the environment configuration for this condition."""
        domain = {}
        parameters = {'foot_ground_friction': self.foot_ground_friction,
                      'base_mass': self.base_mass, 'base_com': self.base_com}
        for key, value in parameters.items():
            if value is None:
                domain.pop(key, None)
            else:
                domain[key] = deepcopy(value)
        return {**config, 'domain_randomization': domain}

    def phase_schedule(self, dt: float) -> dict[int, FrictionPhase]:
        schedule = {}
        elapsed = 0.0
        for phase in self.friction_phases:
            step = round(elapsed / dt)
            if phase.duration_seconds <= 0 or round((elapsed + phase.duration_seconds) / dt) <= step:
                raise ValueError('Friction phases must span at least one control step')
            schedule[step] = phase
            elapsed += phase.duration_seconds
        return schedule

    def apply_phase(self, env, phase: FrictionPhase) -> None:
        import torch

        ratio = env.foot_friction.new_tensor([phase.multiplier])
        env.backend.set_foot_friction_multipliers(ratio, torch.arange(env.num_envs, device=env.device))
        base = env.foot_friction.new_tensor([
            [self.foot_ground_friction['base_ranges'][foot][term][0]
             for term in ('sliding', 'torsional', 'rolling')] for foot in FOOT_NAMES
        ])
        env.foot_friction.copy_((base * ratio[0]).reshape(1, 12))

@dataclass(frozen=True)
class PhysicalSample:
    """A repeatable draw of body parameters and the nominal friction multiplier."""

    name: str
    com_offset_m: tuple[float, float, float]
    mass_scale: float
    friction_multiplier: float

    @classmethod
    def sample(cls, seed: int) -> 'PhysicalSample':
        rng = Random(seed)
        offset = (rng.uniform(-0.01, 0.01), rng.uniform(-0.03, 0.03), rng.uniform(-0.01, 0.01))
        return cls(f'random-{seed}', offset, rng.uniform(0.9, 1.1), rng.uniform(0.75, 1.25))

    def condition(
        self,
        name: str,
        sliding: tuple[float, float, float, float],
        *,
        friction_multiplier: float = 1.0,
        friction_phases: tuple[FrictionPhase, ...] = (),
    ) -> EvaluationCondition:
        """Use fixed bounds so each simulator and each motion reuse the chosen draw."""
        return EvaluationCondition(
            name=name,
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [friction_multiplier, friction_multiplier],
                'base_ranges': {
                    foot: {'sliding': [value, value], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]}
                    for foot, value in zip(('FL', 'FR', 'RL', 'RR'), sliding, strict=True)
                },
            },
            base_mass={'enabled': True, 'scale_range': [self.mass_scale, self.mass_scale]},
            base_com={'enabled': True, 'offset_ranges_m': {
                axis: [value, value] for axis, value in zip('xyz', self.com_offset_m, strict=True)
            }},
            friction_phases=friction_phases,
        )


NOMINAL_SAMPLE = PhysicalSample('nominal', (0.0, 0.0, 0.0), 1.0, 1.0)
PHYSICAL_SAMPLES = (NOMINAL_SAMPLE, PhysicalSample.sample(1), PhysicalSample.sample(2))
NOMINAL_SLIDING = (0.8, 0.8, 0.8, 0.8)
