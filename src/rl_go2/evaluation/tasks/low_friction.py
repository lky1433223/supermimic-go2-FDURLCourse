"""Compare fixed low friction on nominal and training-limit body conditions."""
from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE
from ..task import EvaluationTask
from .rich_motion import GENERALIZATION_MOTIONS


# Sliding coefficients are ordered FL, FR, RL, RR.
SLIDING_CONDITIONS = (
    ('uniform-01', (0.1, 0.1, 0.1, 0.1)),
    ('uniform-02', (0.2, 0.2, 0.2, 0.2)),
    ('left-low', (0.1, 0.6, 0.2, 0.8)),
    ('right-low', (0.6, 0.1, 0.8, 0.2)),
)

BODY_CONDITIONS = (
    NOMINAL_SAMPLE,
    replace(NOMINAL_SAMPLE, name='mass-max', mass_scale=1.50),
    replace(NOMINAL_SAMPLE, name='com-y-max', com_offset_m=(0.0, 0.05, 0.0)),
)


class LowFrictionTask(EvaluationTask):
    TASK = 'low-friction'
    TASK_DESCRIPTION = 'Run eight motions under four low-friction settings and three body conditions.'

    conditions = tuple(
        sample.condition(f'{sample.name}-{name}', sliding)
        for sample in BODY_CONDITIONS
        for name, sliding in SLIDING_CONDITIONS
    )
    motions = {'tracking': GENERALIZATION_MOTIONS}

    metrics = 'tracking'
    report = {'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20}
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True
