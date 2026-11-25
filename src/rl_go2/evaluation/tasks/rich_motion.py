"""Track varied velocity commands with nominal physical parameters."""
from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..motion_tasks import MotionTask
from ..task import EvaluationTask


class RichMotionTask(EvaluationTask):
    TASK = 'rich-motion'
    TASK_DESCRIPTION = 'Track 37 in-range and out-of-range commands under nominal physics.'

    conditions = (NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING),)
    motions = {
        'in-range': (
            MotionTask('stand', 0),
            MotionTask('vx-1.0_yaw0', -1.0),
            MotionTask('vx-0.5_yaw0', -0.5),
            MotionTask('vx-0.2_yaw0', -0.2),
            MotionTask('vx0.2_yaw0', 0.2),
            MotionTask('vx0.5_yaw0', 0.5),
            MotionTask('vx1.0_yaw0', 1.0),
            MotionTask('vx1.5_yaw0', 1.5),
            MotionTask('vx2.0_yaw0', 2.0),
            MotionTask('vy-1.0', 0, vy=-1.0),
            MotionTask('vy-0.5', 0, vy=-0.5),
            MotionTask('vy0.5', 0, vy=0.5),
            MotionTask('vy1.0', 0, vy=1.0),
            MotionTask('yaw-1.0', 0, -1.0),
            MotionTask('yaw-0.5', 0, -0.5),
            MotionTask('yaw-0.25', 0, -0.25),
            MotionTask('yaw0.25', 0, 0.25),
            MotionTask('yaw0.5', 0, 0.5),
            MotionTask('yaw1.0', 0, 1.0),
            MotionTask('vx0.5_yaw-0.5', 0.5, -0.5),
            MotionTask('vx0.5_yaw0.5', 0.5, 0.5),
            MotionTask('vx1.5_yaw-1.0', 1.5, -1.0),
            MotionTask('vx1.5_yaw1.0', 1.5, 1.0),
            MotionTask('vx-0.5_vy-0.5', -0.5, vy=-0.5),
            MotionTask('vx-0.5_vy0.5', -0.5, vy=0.5),
            MotionTask('vx0.5_vy-0.5', 0.5, vy=-0.5),
            MotionTask('vx0.5_vy0.5', 0.5, vy=0.5),
            MotionTask('vx1.0_vy-0.5_yaw-0.5', 1.0, -0.5, vy=-0.5),
            MotionTask('vx1.0_vy0.5_yaw0.5', 1.0, 0.5, vy=0.5),
            MotionTask('vx2.0_vy-1.0_yaw-1.0', 2.0, -1.0, vy=-1.0),
            MotionTask('vx2.0_vy1.0_yaw1.0', 2.0, 1.0, vy=1.0),
        ),
        'out-of-range': (
            MotionTask('vx-1.25_yaw0', -1.25),
            MotionTask('vx2.50_yaw0', 2.50),
            MotionTask('vy-1.25', 0, vy=-1.25),
            MotionTask('vy1.25', 0, vy=1.25),
            MotionTask('yaw-1.25', 0, -1.25),
            MotionTask('yaw1.25', 0, 1.25),
        ),
    }

    metrics = 'tracking'
    report = {'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20}
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True


_IN_RANGE_BY_NAME = {motion.name: motion for motion in RichMotionTask.motions['in-range']}
GENERALIZATION_MOTIONS = tuple(_IN_RANGE_BY_NAME[name] for name in (
    'stand',
    'vx0.5_yaw0',
    'vx-0.5_yaw0',
    'vy0.5',
    'yaw0.5',
    'vx2.0_yaw0',
    'vx0.5_yaw0.5',
    'vx0.5_yaw-0.5',
))
