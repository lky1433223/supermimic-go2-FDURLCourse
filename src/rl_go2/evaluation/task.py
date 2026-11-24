"""Declarative contract shared by evaluation tasks."""
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
import math
from pathlib import Path

from rl_go2.config import load_scene

from .conditions import EvaluationCondition
from .motion_tasks import MotionTask


@dataclass(frozen=True)
class SceneSpec:
    """A named scene file with optional overrides for its terrain entities."""

    name: str
    file: Path
    entity_overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationGroup:
    """Scenes, conditions and motions to combine within one group."""

    name: str
    scenes: tuple[SceneSpec, ...]
    conditions: tuple[EvaluationCondition, ...]
    motions: tuple[MotionTask, ...]


@dataclass(frozen=True)
class Trial:
    condition: EvaluationCondition
    motion: MotionTask
    metrics: str
    group: str = ''
    steps: int | None = None
    continue_after_reset: bool = False
    reset_before_start: bool = False
    scene: str | None = None

    @property
    def name(self) -> str:
        return self.motion.name

    def event_key(self, seed: int) -> tuple[int, str | None, str, str, str]:
        """Identify an event within a plan, including its scene and group."""
        return seed, self.scene, self.group, self.condition.name, self.name


@dataclass(frozen=True)
class EvaluationPlan:
    trials: tuple[Trial, ...]
    report: dict
    episode_seconds: float
    scenes: dict[str, dict] = field(default_factory=dict)
    separate_conditions: bool = False

    def select_trials(
        self, *, conditions: list[str] | None = None, motions: list[str] | None = None
    ) -> 'EvaluationPlan':
        """Filter by declared names while preserving execution order."""
        if conditions is None and motions is None:
            return self
        condition_names = set(conditions or ())
        motion_names = set(motions or ())
        unknown_conditions = condition_names - {trial.condition.name for trial in self.trials}
        unknown_motions = motion_names - {trial.name for trial in self.trials}
        if unknown_conditions:
            raise ValueError(f'Unknown condition names: {", ".join(sorted(unknown_conditions))}')
        if unknown_motions:
            raise ValueError(f'Unknown trial motion names: {", ".join(sorted(unknown_motions))}')
        selected = tuple(
            trial for trial in self.trials
            if (not condition_names or trial.condition.name in condition_names)
            and (not motion_names or trial.name in motion_names)
        )
        if not selected:
            raise ValueError('No trials match the selected conditions and motions')
        scene_names = {trial.scene for trial in selected}
        return replace(self, trials=selected, scenes={
            name: scene for name, scene in self.scenes.items() if name in scene_names
        })

    def scene_config(self, config: dict, trial: Trial) -> dict:
        """Select the resolved scene, falling back to the training configuration."""
        if trial.scene is None:
            return config
        return {**config, 'scene': self.scenes[trial.scene]}


class EvaluationTask:
    """Expand groups of declared scenes, conditions and motions into trials."""

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
    default_seeds: tuple[int, ...] = (1,)
    environment_task: str | None = None
    evaluation_groups: tuple[EvaluationGroup, ...] = ()
    separate_conditions: bool = False

    def trial_combinations(self) -> Iterator[tuple[str, SceneSpec | None, EvaluationCondition, MotionTask]]:
        """Yield declarations in environment reuse order."""
        if self.evaluation_groups:
            for group in self.evaluation_groups:
                if not all((group.name, group.scenes, group.conditions, group.motions)):
                    raise ValueError('Evaluation groups require a name, scenes, conditions and motions')
                for scene in group.scenes:
                    for condition in group.conditions:
                        for motion in group.motions:
                            yield group.name, scene, condition, motion
            return
        for condition in self.conditions:
            for group, motions in self.motions.items():
                for motion in motions:
                    yield group, None, condition, motion

    def build_plan(self, steps: int | None = None) -> EvaluationPlan:
        """Validate declarations, resolve each scene once and construct trials."""
        trials = []
        specs: dict[str, SceneSpec] = {}
        conditions: dict[str, EvaluationCondition] = {}
        scenes: dict[str, dict] = {}
        keys = set()
        for group, scene, condition, motion in self.trial_combinations():
            if scene is not None:
                if not scene.name or specs.get(scene.name, scene) != scene:
                    raise ValueError(f'Invalid or conflicting scene name: {scene.name!r}')
                if scene.name not in specs:
                    specs[scene.name] = scene
                    scenes[scene.name] = load_scene(scene.file, entity_overrides=scene.entity_overrides)
            if condition.name not in conditions:
                condition.validate()
                conditions[condition.name] = condition
            if conditions[condition.name] != condition:
                raise ValueError(f'Conflicting condition name: {condition.name!r}')
            motion.validate()
            phase_duration = sum(phase.duration_seconds for phase in condition.friction_phases)
            if condition.friction_phases and not math.isclose(phase_duration, motion.duration_seconds):
                raise ValueError(f'{condition.name} phases must cover {motion.name}')
            trial = Trial(
                conditions[condition.name], motion, self.metrics, group=group, steps=steps or self.default_steps,
                continue_after_reset=self.continue_after_reset, reset_before_start=self.reset_before_start,
                scene=scene.name if scene is not None else None,
            )
            key = trial.event_key(0)
            if key in keys:
                raise ValueError(f'Duplicate trial: {key[1:]}')
            keys.add(key)
            trials.append(trial)
        if not trials:
            raise ValueError(f'{self.TASK} must declare at least one condition and motion')
        if self.episode_seconds < max(trial.motion.duration_seconds for trial in trials):
            raise ValueError('Episode duration must cover every motion')
        return EvaluationPlan(tuple(trials), self.report.copy(), self.episode_seconds, scenes,
                              separate_conditions=self.separate_conditions)
