"""Measure recovery at and beyond the Go2 point-push training limits."""
from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..metrics import RECOVERY_CRITERIA
from ..push_events import PushSpec
from .body_parameters import BodyParameterTask


BASELINE = NOMINAL_SAMPLE.condition('no-push', NOMINAL_SLIDING)


class PointPushTask(BodyParameterTask):
    """Test one matched transient force per motion on nominal physics."""

    TASK = 'point-push'
    TASK_DESCRIPTION = 'Compare no push, training-limit pushes, and out-of-range pushes.'
    backends = ('mujoco',)
    default_seeds = (1, 2, 3)
    conditions = (
        BASELINE,
        replace(BASELINE, name='base-limit', point_push=PushSpec('base', 20.0)),
        replace(BASELINE, name='base-ood', point_push=PushSpec('base', 30.0)),
        replace(BASELINE, name='base-80n', point_push=PushSpec('base', 80.0)),
        replace(BASELINE, name='base-160n', point_push=PushSpec('base', 160.0)),
        replace(BASELINE, name='foot-limit', point_push=PushSpec('foot', 8.0)),
        replace(BASELINE, name='foot-ood', point_push=PushSpec('foot', 12.0)),
    )
    report = {
        **BodyParameterTask.report,
        'recovery_criteria': RECOVERY_CRITERIA,
    }
