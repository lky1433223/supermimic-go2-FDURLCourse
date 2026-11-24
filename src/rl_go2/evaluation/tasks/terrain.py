"""Compare fixed terrain presets under the same tracking command."""

from rl_go2.config import FLAT_SCENE_PATH

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..motion_tasks import MotionTask
from ..task import EvaluationGroup, EvaluationTask, SceneSpec


STEP_SCENE = FLAT_SCENE_PATH.with_name('single_step.yaml')
RAMP_SCENE = FLAT_SCENE_PATH.with_name('ramp.yaml')
ROUGH_SCENE = FLAT_SCENE_PATH.with_name('rough_ground.yaml')


class TerrainTask(EvaluationTask):
    TASK = 'terrain'
    TASK_DESCRIPTION = 'Compare straight walking on flat ground, single steps, ramps, and seeded rough ground.'

    evaluation_groups = (
        EvaluationGroup(
            name='crossing',
            scenes=(
                SceneSpec('flat', FLAT_SCENE_PATH),
                SceneSpec('step-2cm', STEP_SCENE, {'step': {'height_m': 0.02}}),
                SceneSpec('step-4cm', STEP_SCENE, {'step': {'height_m': 0.04}}),
                SceneSpec('step-6cm', STEP_SCENE, {'step': {'height_m': 0.06}}),
                SceneSpec('ramp-5deg', RAMP_SCENE, {'ramp': {'slope_deg': 5.0}}),
                SceneSpec('ramp-10deg', RAMP_SCENE, {'ramp': {'slope_deg': 10.0}}),
                SceneSpec('rough-101', ROUGH_SCENE, {'rough': {'seed': 101}}),
                SceneSpec('rough-102', ROUGH_SCENE, {'rough': {'seed': 102}}),
            ),
            conditions=(NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING),),
            motions=(MotionTask('forward_0.5', 0.5, duration_seconds=8),),
        ),
    )
    metrics = 'tracking'
    report = {'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 8}
    episode_seconds = 8
    continue_after_reset = False
    reset_before_start = True
