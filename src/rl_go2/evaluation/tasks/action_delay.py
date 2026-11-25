"""Compare fixed policy-command delays under the same flat-ground motions."""

from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..motion_tasks import FRICTION_MOTIONS
from ..task import EvaluationTask


NOMINAL = NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING)


class ActionDelayTask(EvaluationTask):
    """Measure tracking with zero, one or two control steps of action delay."""

    TASK = 'action-delay'
    TASK_DESCRIPTION = 'Compare 0, 1, and 2 control steps of action delay with clean observations.'
    environment_task = 'flat_walk'
    separate_conditions = True
    conditions = (
        replace(NOMINAL, action_delay_steps=(0, 0)),
        replace(NOMINAL, name='delay-1-step', action_delay_steps=(1, 1)),
        replace(NOMINAL, name='delay-2-steps', action_delay_steps=(2, 2)),
    )
    motions = {'tracking': FRICTION_MOTIONS}
    metrics = 'tracking'
    report = {
        'velocity_axes': ['vx', 'vy', 'yaw_rate'],
        'case_seconds': 20,
        'stop_criteria': {'planar_speed_m_s': 0.1, 'abs_yaw_rad_s': 0.15, 'hold_seconds': 0.5},
    }
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True
    default_seeds = (1, 2, 3)
