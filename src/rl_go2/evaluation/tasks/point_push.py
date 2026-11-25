"""Measure recovery at and beyond the Go2 point-push training limits."""
from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..metrics import RECOVERY_CRITERIA
from ..push_events import PushSpec
from .body_parameters import BodyParameterTask


NOMINAL = NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING)


class PointPushTask(BodyParameterTask):
    """Test four lateral trunk pushes and two sampled foot pushes per motion."""

    TASK = 'point-push'
    TASK_DESCRIPTION = 'Measure recovery from lateral trunk pushes and two sampled foot pushes per motion.'
    backends = ('mujoco',)
    conditions = (
        replace(NOMINAL, name='base-50n-300ms', point_push=PushSpec('base', 50.0, duration_seconds=0.30)),
        replace(NOMINAL, name='base-100n-200ms', point_push=PushSpec('base', 100.0, duration_seconds=0.20)),
        replace(NOMINAL, name='base-200n-100ms', point_push=PushSpec('base', 200.0, duration_seconds=0.10)),
        replace(NOMINAL, name='base-200n-300ms', point_push=PushSpec('base', 200.0, duration_seconds=0.30)),
        replace(NOMINAL, name='foot-20n-draw-1', point_push=PushSpec('foot', 20.0)),
        replace(NOMINAL, name='foot-20n-draw-2', point_push=PushSpec('foot', 20.0, foot_draw=1)),
    )
    report = {
        **BodyParameterTask.report,
        'recovery_criteria': RECOVERY_CRITERIA,
    }
