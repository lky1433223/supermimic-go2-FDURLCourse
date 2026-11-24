"""Compare clean and perturbed actor inputs under the same flat-ground motions."""

from dataclasses import replace

from ..conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from ..motion_tasks import FRICTION_MOTIONS
from ..task import EvaluationTask


UNIFORM_NOISE = {
    'angular_velocity': [{'type': 'uniform', 'low': -0.05, 'high': 0.05}],
    'attitude': [{'type': 'uniform', 'low': -0.02, 'high': 0.02}],
    'joint_position': [{'type': 'uniform', 'low': -0.01, 'high': 0.01}],
    'joint_velocity': [{'type': 'uniform', 'low': -0.25, 'high': 0.25}],
}
COMBINED_NOISE = {
    'angular_velocity': [*UNIFORM_NOISE['angular_velocity'], {'type': 'gaussian', 'std': 0.015},
                         {'type': 'delay', 'steps': 1}],
    'attitude': [*UNIFORM_NOISE['attitude'], {'type': 'delay', 'steps': 1}],
    'joint_position': [{'type': 'bias', 'offset': 0.005}, *UNIFORM_NOISE['joint_position']],
    'joint_velocity': [*UNIFORM_NOISE['joint_velocity'], {'type': 'dropout', 'probability': 0.05}],
}
# Unitree's Go2 RL Lab noise bounds, with gravity perturbation modeled as body tilt.
# Delay all fields together to represent receiving an older LowState sample.
LOWSTATE_DELAY = {'type': 'delay', 'steps': 1}
GO2_REFERENCE_NOISE = {
    'angular_velocity': [{'type': 'uniform', 'low': -0.2, 'high': 0.2}, LOWSTATE_DELAY],
    'attitude': [{'type': 'uniform', 'low': [-0.05, -0.05, 0.0], 'high': [0.05, 0.05, 0.0]}, LOWSTATE_DELAY],
    'joint_position': [{'type': 'uniform', 'low': -0.01, 'high': 0.01}, LOWSTATE_DELAY],
    'joint_velocity': [{'type': 'uniform', 'low': -1.5, 'high': 1.5}, LOWSTATE_DELAY],
}
NOMINAL = NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING)


class SensorNoiseTask(EvaluationTask):
    """Track the same commands with different fixed sensor-noise conditions."""

    TASK = 'sensor-noise'
    TASK_DESCRIPTION = 'Compare nominal, uniform, combined, and Go2 reference sensor noise.'
    environment_task = 'flat_walk'
    separate_conditions = True
    conditions = (
        NOMINAL,
        replace(NOMINAL, name='uniform', observation_noise=UNIFORM_NOISE),
        replace(NOMINAL, name='combined', observation_noise=COMBINED_NOISE),
        replace(NOMINAL, name='go2-reference', observation_noise=GO2_REFERENCE_NOISE),
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
