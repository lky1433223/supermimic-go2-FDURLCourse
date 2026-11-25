"""Run the shared motion set as a smoke test under nominal physics."""
from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..task import EvaluationTask
from .rich_motion import GENERALIZATION_MOTIONS


class RolloutTask(EvaluationTask):
    TASK = 'rollout'
    TASK_DESCRIPTION = 'Run eight nominal motion checks with a step budget per motion.'

    conditions = (NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING),)
    motions = {'': GENERALIZATION_MOTIONS}
    metrics = 'rollout'
    report = {}
    episode_seconds = 20
    continue_after_reset = True
    reset_before_start = True
    supports_parallel = True
    default_steps = 1000
