"""Approach each obstacle with forward and lateral walking."""

from dataclasses import replace
from math import sqrt

from rl_go2.config import FLAT_SCENE_PATH

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..metrics import TERRAIN_SUCCESS_CRITERIA
from ..motion_tasks import MotionTask
from ..task import EvaluationGroup, EvaluationTask, SceneSpec


STEP_SCENE = FLAT_SCENE_PATH.with_name('single_step.yaml')
STAIRS_SCENE = FLAT_SCENE_PATH.with_name('stairs.yaml')
RAMP_SCENE = FLAT_SCENE_PATH.with_name('ramp.yaml')
ROUGH_SCENE = FLAT_SCENE_PATH.with_name('rough_ground.yaml')
CASE_SECONDS = 8
NOMINAL = NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING)
OBSTACLES = (
    SceneSpec('step-5cm', STEP_SCENE, {'step': {'height_m': 0.05}}),
    SceneSpec('step-10cm', STEP_SCENE, {'step': {'height_m': 0.10}}),
    SceneSpec('stairs-5cm', STAIRS_SCENE, {'stairs': {'step_height_m': 0.05}}),
    SceneSpec('stairs-10cm', STAIRS_SCENE, {'stairs': {'step_height_m': 0.10}}),
    SceneSpec('ramp-10deg', RAMP_SCENE, {'ramp': {'slope_deg': 10.0}}),
    SceneSpec('ramp-20deg', RAMP_SCENE, {'ramp': {'slope_deg': 20.0}}),
    *(SceneSpec(f'rough-{seed}', ROUGH_SCENE, {'rough': {
        'seed': seed, 'max_height_level': 5, 'max_adjacent_delta_levels': 5,
    }}) for seed in (101, 102)),
)
# A -90 degree yaw maps body +Y to world +X, toward the obstacle.
SIDEWAYS_ROTATION = (sqrt(0.5), 0.0, 0.0, -sqrt(0.5))


class TerrainTask(EvaluationTask):
    TASK = 'terrain'
    TASK_DESCRIPTION = 'Walk forward and sideways onto steps, stairs, ramps, and seeded rough ground.'

    evaluation_groups = (
        EvaluationGroup(
            name='forward',
            scenes=OBSTACLES,
            conditions=(NOMINAL,),
            motions=(MotionTask('forward_0.5', 0.5, duration_seconds=CASE_SECONDS),),
        ),
        EvaluationGroup(
            name='lateral',
            scenes=tuple(replace(scene, name=f'{scene.name}-side', robot_rotation=SIDEWAYS_ROTATION)
                         for scene in OBSTACLES),
            conditions=(NOMINAL,),
            motions=(MotionTask('lateral_0.5', 0.0, vy=0.5, duration_seconds=CASE_SECONDS),),
        ),
    )
    metrics = 'terrain'
    report = {
        'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': CASE_SECONDS,
        'success_criteria': TERRAIN_SUCCESS_CRITERIA,
    }
    episode_seconds = CASE_SECONDS
    continue_after_reset = False
    reset_before_start = True


class StairsTask(TerrainTask):
    """Run the terrain motions on continuous stair flights only."""

    TASK = 'stairs'
    TASK_DESCRIPTION = 'Walk forward and sideways onto 10-step flights with 5 cm and 10 cm rises.'
    evaluation_groups = tuple(
        replace(group, scenes=tuple(scene for scene in group.scenes if scene.name.startswith('stairs-')))
        for group in TerrainTask.evaluation_groups
    )
