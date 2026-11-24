"""A fixed straight-walking smoke test with explicit physical conditions."""
from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..motion_tasks import MotionTask
from ..task import EvaluationTask


class RolloutTask(EvaluationTask):
    TASK = 'rollout'
    TASK_DESCRIPTION = 'Run fixed 0.5 m/s straight walking for a step budget.'

    conditions = (NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING),)
    motions = {'': (MotionTask('forward_0.5', 0.5, yaw_rate=0, duration_seconds=20),)}
    metrics = 'rollout'
    report = {}
    episode_seconds = 20
    continue_after_reset = True
    reset_before_start = False
    supports_parallel = True
    default_steps = 1000
