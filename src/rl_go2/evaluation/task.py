"""Declarative contract shared by all four evaluation tasks."""
from dataclasses import dataclass
import math

from .conditions import EvaluationCondition
from .motion_tasks import MotionTask


@dataclass(frozen=True)
class Trial:
    condition: EvaluationCondition
    motion: MotionTask
    metrics: str
    group: str = ''
    steps: int | None = None
    continue_after_reset: bool = False
    reset_before_start: bool = False

    @property
    def name(self) -> str:
        return self.motion.name


@dataclass(frozen=True)
class EvaluationPlan:
    trials: tuple[Trial, ...]
    report: dict
    episode_seconds: float


class EvaluationTask:
    """Subclasses declare conditions and motions; this class expands their product."""

    TASK: str
    TASK_DESCRIPTION: str
    conditions: tuple[EvaluationCondition, ...]
    motions: dict[str, tuple[MotionTask, ...]]
    metrics: str
    report: dict
    episode_seconds: float
    continue_after_reset: bool
    reset_before_start: bool
    backends: tuple[str, ...] = ('mujoco', 'genesis')
    supports_parallel: bool = False
    default_steps: int | None = None

    def build_plan(self, steps: int | None = None) -> EvaluationPlan:
        """Validate declarations and pair each condition with each motion."""
        trials = []
        for condition in self.conditions:
            condition.validate()
            for group, motions in self.motions.items():
                for motion in motions:
                    motion.validate()
                    phase_duration = sum(phase.duration_seconds for phase in condition.friction_phases)
                    if condition.friction_phases and not math.isclose(phase_duration, motion.duration_seconds):
                        raise ValueError(f'{condition.name} phases must cover {motion.name}')
                    trials.append(Trial(
                        condition, motion, self.metrics, group=group, steps=steps or self.default_steps,
                        continue_after_reset=self.continue_after_reset,
                        reset_before_start=self.reset_before_start,
                    ))
        if not trials:
            raise ValueError(f'{self.TASK} must declare at least one condition and motion')
        if self.episode_seconds < max(trial.motion.duration_seconds for trial in trials):
            raise ValueError('Episode duration must cover every motion')
        return EvaluationPlan(tuple(trials), self.report.copy(), self.episode_seconds)
