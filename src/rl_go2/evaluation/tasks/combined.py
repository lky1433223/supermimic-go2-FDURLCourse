"""Evaluate coordinated motion with training-range disturbances applied together."""
from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..metrics import RECOVERY_CRITERIA, TERRAIN_SUCCESS_CRITERIA
from ..motion_tasks import MotionTask
from ..push_events import PushSpec
from ..task import EvaluationGroup, EvaluationTask, SceneSpec
from .sensor_action import ACTION_DELAY_STEPS, OBSERVATION_NOISE
from .terrain import CASE_SECONDS, ROUGH_SCENE, SIDEWAYS_ROTATION


ROUGH = SceneSpec('rough-101', ROUGH_SCENE, {'rough': {
    'seed': 101, 'max_height_level': 5, 'max_adjacent_delta_levels': 3,
}})
BODY = replace(NOMINAL_SAMPLE, mass_scale=1.50, com_offset_m=(0.0, 0.05, 0.0))
CONDITION = replace(
    BODY.condition('combined', NOMINAL_SLIDING, friction_multiplier=0.75),
    observation_noise=OBSERVATION_NOISE,
    action_delay_steps=ACTION_DELAY_STEPS,
    point_push=PushSpec('base', 100.0, duration_seconds=0.20, trigger_height_map='rough'),
)


class CombinedTask(EvaluationTask):
    """Check tracking and recovery on rough ground with several disturbances."""

    TASK = 'combined'
    TASK_DESCRIPTION = 'Track three motions on rough ground with combined disturbances and an entry-triggered push.'
    backends = ('mujoco',)
    environment_task = 'flat_walk'
    evaluation_groups = (
        EvaluationGroup(
            name='forward',
            scenes=(ROUGH,),
            conditions=(CONDITION,),
            motions=(MotionTask('forward_0.5', 0.5, duration_seconds=CASE_SECONDS),
                     MotionTask('forward_2.0', 2.0, duration_seconds=CASE_SECONDS)),
        ),
        EvaluationGroup(
            name='lateral',
            scenes=(replace(ROUGH, name='rough-101-side', robot_rotation=SIDEWAYS_ROTATION),),
            conditions=(CONDITION,),
            motions=(MotionTask('lateral_0.5', 0.0, vy=0.5, duration_seconds=CASE_SECONDS),),
        ),
    )
    metrics = 'terrain'
    report = {
        'velocity_axes': ['vx', 'vy', 'yaw_rate'],
        'case_seconds': CASE_SECONDS,
        'success_criteria': TERRAIN_SUCCESS_CRITERIA,
        'recovery_criteria': RECOVERY_CRITERIA,
    }
    episode_seconds = CASE_SECONDS
    continue_after_reset = False
    reset_before_start = True
