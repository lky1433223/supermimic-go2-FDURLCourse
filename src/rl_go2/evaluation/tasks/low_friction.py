"""Compare fixed foot friction across shared motions and body samples."""
from ..conditions import NOMINAL_SLIDING, PHYSICAL_SAMPLES
from ..motion_tasks import FRICTION_MOTIONS
from ..task import EvaluationTask


# Sliding coefficients are ordered FL, FR, RL, RR.
SLIDING_CONDITIONS = (
    ('fixed', NOMINAL_SLIDING),
    ('uniform-01', (0.1, 0.1, 0.1, 0.1)),
    ('uniform-02', (0.2, 0.2, 0.2, 0.2)),
    ('left-low', (0.1, 0.6, 0.2, 0.8)),
    ('right-low', (0.6, 0.1, 0.8, 0.2)),
)


class LowFrictionTask(EvaluationTask):
    TASK = 'low-friction'
    TASK_DESCRIPTION = 'Run six motions under five fixed friction settings and three body samples.'

    conditions = tuple(
        sample.condition(f'{sample.name}-{name}', sliding)
        for sample in PHYSICAL_SAMPLES
        for name, sliding in SLIDING_CONDITIONS
    )
    motions = {'tracking': FRICTION_MOTIONS}

    metrics = 'tracking'
    report = {
        'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20,
        'stop_criteria': {'planar_speed_m_s': 0.1, 'abs_yaw_rad_s': 0.15, 'hold_seconds': 0.5},
    }
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True
