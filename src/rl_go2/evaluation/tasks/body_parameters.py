"""Sweep body mass and center of mass independently on flat ground."""
from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..motion_tasks import FRICTION_MOTIONS
from ..task import EvaluationTask


MASS_SCALES = (0.50, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50)

# Offsets are in meters along the body X/Y/Z axes.
COM_OFFSETS_M = (
    ('nominal', (0.0, 0.0, 0.0)),
    ('x-minus-1cm', (-0.01, 0.0, 0.0)),
    ('x-plus-1cm', (0.01, 0.0, 0.0)),
    ('x-minus-3cm', (-0.03, 0.0, 0.0)),
    ('x-plus-3cm', (0.03, 0.0, 0.0)),
    ('x-minus-5cm', (-0.05, 0.0, 0.0)),
    ('x-plus-5cm', (0.05, 0.0, 0.0)),
    ('y-minus-3cm', (0.0, -0.03, 0.0)),
    ('y-plus-3cm', (0.0, 0.03, 0.0)),
    ('y-minus-4cm', (0.0, -0.04, 0.0)),
    ('y-plus-4cm', (0.0, 0.04, 0.0)),
    ('z-minus-1cm', (0.0, 0.0, -0.01)),
    ('z-plus-1cm', (0.0, 0.0, 0.01)),
    ('xyz-plus-3-3-1cm', (0.03, 0.03, 0.01)),
)


class BodyParameterTask(EvaluationTask):
    """Use the same undisturbed tracking motions for both physical sweeps."""

    environment_task = 'flat_walk'
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


class MassTask(BodyParameterTask):
    """Sweep mass and inertia with zero center-of-mass offset."""

    TASK = 'mass'
    TASK_DESCRIPTION = 'Run six motions at seven fixed trunk mass scales.'
    conditions = tuple(
        replace(NOMINAL_SAMPLE, mass_scale=scale).condition(
            'nominal' if scale == 1.0 else f'mass-{scale:.2f}', NOMINAL_SLIDING
        )
        for scale in MASS_SCALES
    )


class ComOffsetTask(BodyParameterTask):
    """Sweep center-of-mass offset with nominal mass and inertia."""

    TASK = 'com-offset'
    TASK_DESCRIPTION = 'Run six motions at fourteen fixed trunk center-of-mass offsets.'
    conditions = tuple(
        replace(NOMINAL_SAMPLE, com_offset_m=offset_m).condition(name, NOMINAL_SLIDING)
        for name, offset_m in COM_OFFSETS_M
    )
