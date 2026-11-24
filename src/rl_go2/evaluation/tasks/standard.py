"""Standard velocity tracking under explicitly declared physical conditions."""
from ..conditions import NOMINAL_SLIDING, PHYSICAL_SAMPLES
from ..motion_tasks import MotionTask
from ..task import EvaluationTask


class StandardTask(EvaluationTask):
    TASK = 'standard'
    TASK_DESCRIPTION = 'Run 27 tracking motions under each of three physical samples.'

    conditions = tuple(
        sample.condition(sample.name, NOMINAL_SLIDING, friction_multiplier=sample.friction_multiplier)
        for sample in PHYSICAL_SAMPLES
    )
    motions = {
        'standard': (
            MotionTask('vx0.2_yaw0', 0.2),
            MotionTask('vx0.2_yaw0.4', 0.2, 0.4),
            MotionTask('vx0.2_yaw-0.4', 0.2, -0.4),
            MotionTask('vx0.4_yaw0', 0.4),
            MotionTask('vx0.4_yaw0.4', 0.4, 0.4),
            MotionTask('vx0.4_yaw-0.4', 0.4, -0.4),
            MotionTask('vx0.6_yaw0', 0.6),
            MotionTask('vx0.6_yaw0.4', 0.6, 0.4),
            MotionTask('vx0.6_yaw-0.4', 0.6, -0.4),
            MotionTask('vx0.8_yaw0', 0.8),
            MotionTask('vx0.8_yaw0.4', 0.8, 0.4),
            MotionTask('vx0.8_yaw-0.4', 0.8, -0.4),
            MotionTask('vx0.5_yaw0', 0.5),
            MotionTask('vx0.5_yaw0.1', 0.5, 0.1),
            MotionTask('vx0.5_yaw-0.1', 0.5, -0.1),
            MotionTask('vx0.5_yaw0.3', 0.5, 0.3),
            MotionTask('vx0.5_yaw-0.3', 0.5, -0.3),
            MotionTask('vx0.5_yaw0.5', 0.5, 0.5),
            MotionTask('vx0.5_yaw-0.5', 0.5, -0.5),
        ),
        'ood': (
            MotionTask('stand', 0),
            MotionTask('walk_to_stop', 0.5, stop_at_seconds=8),
            MotionTask('vx1.0_yaw0', 1.0),
            MotionTask('vx1.2_yaw0', 1.2),
            MotionTask('vx0.5_yaw0.6', 0.5, 0.6),
            MotionTask('vx0.5_yaw-0.6', 0.5, -0.6),
            MotionTask('vx0.5_yaw0.8', 0.5, 0.8),
            MotionTask('vx0.5_yaw-0.8', 0.5, -0.8),
        ),
    }

    metrics = 'tracking'
    report = {
        'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20,
        'stop_criteria': {'planar_speed_m_s': 0.1, 'abs_yaw_rad_s': 0.15, 'hold_seconds': 0.5},
    }
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True
