"""Evaluate sensor degradation and action delay separately and together."""

from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..task import EvaluationTask
from .rich_motion import GENERALIZATION_MOTIONS


# Match the training noise bounds, with delay fixed at the training upper bound.
OBSERVATION_NOISE = {
    'frame_delay_steps': [1, 1],
    'angular_velocity': [{'type': 'uniform', 'low': -0.20, 'high': 0.20}],
    'projected_gravity': [{'type': 'uniform', 'low': -0.05, 'high': 0.05}],
    'joint_position': [{'type': 'uniform', 'low': -0.01, 'high': 0.01}],
    'joint_velocity': [{'type': 'uniform', 'low': -1.50, 'high': 1.50}],
}
ACTION_DELAY_STEPS = (1, 1)
NOMINAL = NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING)


class SensorActionTask(EvaluationTask):
    """Track the same motions with sensor errors, delayed actions, or both."""

    TASK = 'sensor-action'
    TASK_DESCRIPTION = 'Compare noisy delayed observations, delayed actions, and their combination.'
    environment_task = 'flat_walk'
    separate_conditions = True
    conditions = (
        replace(NOMINAL, name='sensor-only', observation_noise=OBSERVATION_NOISE),
        replace(NOMINAL, name='action-only', action_delay_steps=ACTION_DELAY_STEPS),
        replace(NOMINAL, name='combined', observation_noise=OBSERVATION_NOISE,
                action_delay_steps=ACTION_DELAY_STEPS),
    )
    motions = {'tracking': GENERALIZATION_MOTIONS}
    metrics = 'tracking'
    report = {'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20}
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True
    default_seeds = (1, 2, 3)
