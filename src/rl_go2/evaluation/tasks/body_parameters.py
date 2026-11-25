"""Evaluate trunk mass and center-of-mass changes on flat ground."""
from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..task import EvaluationTask
from .rich_motion import GENERALIZATION_MOTIONS


# Offsets are in meters along the body X/Y/Z axes.
COM_OFFSETS_M = (
    ('nominal', (0.0, 0.0, 0.0)),
    ('x-minus-5cm', (-0.05, 0.0, 0.0)),
    ('x-plus-5cm', (0.05, 0.0, 0.0)),
    ('y-minus-5cm', (0.0, -0.05, 0.0)),
    ('y-plus-5cm', (0.0, 0.05, 0.0)),
    ('z-minus-5cm', (0.0, 0.0, -0.05)),
    ('z-plus-5cm', (0.0, 0.0, 0.05)),
    ('xyz-plus-5-5-5cm', (0.05, 0.05, 0.05)),
)


class BodyParameterTask(EvaluationTask):
    """Provide flat-ground tracking defaults for body-parameter evaluations."""

    environment_task = 'flat_walk'
    motions = {'tracking': GENERALIZATION_MOTIONS}
    metrics = 'tracking'
    report = {'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20}
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True


class MassTask(BodyParameterTask):
    """Test the training mass bounds and a heavier, laterally shifted body."""

    TASK = 'mass'
    TASK_DESCRIPTION = 'Run eight motions under four trunk mass and center-of-mass conditions.'
    conditions = (
        NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING),
        replace(NOMINAL_SAMPLE, mass_scale=0.85).condition('mass-0.85', NOMINAL_SLIDING),
        replace(NOMINAL_SAMPLE, mass_scale=1.50).condition('mass-1.50', NOMINAL_SLIDING),
        replace(NOMINAL_SAMPLE, mass_scale=1.50, com_offset_m=(0.0, 0.05, 0.0)).condition(
            'mass-1.50-com-y-plus-5cm', NOMINAL_SLIDING,
        ),
    )


class ComOffsetTask(BodyParameterTask):
    """Sweep center-of-mass offset with nominal mass and inertia."""

    TASK = 'com-offset'
    TASK_DESCRIPTION = 'Run eight motions at eight fixed trunk center-of-mass offsets.'
    conditions = tuple(
        replace(NOMINAL_SAMPLE, com_offset_m=offset_m).condition(name, NOMINAL_SLIDING)
        for name, offset_m in COM_OFFSETS_M
    )
