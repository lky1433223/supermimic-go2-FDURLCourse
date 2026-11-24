"""Straight walking under fixed and timed friction conditions."""
from ..conditions import EvaluationCondition, FrictionPhase
from ..motion_tasks import MotionTask
from ..task import EvaluationTask


class FrictionSwitchTask(EvaluationTask):
    TASK = 'friction-switch'
    TASK_DESCRIPTION = 'Compare fixed friction with a timed 0.8 to 0.4 to 0.8 change.'

    conditions = (
        EvaluationCondition(
            name='fixed',
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [1.0, 1.0],
                'base_ranges': {
                    foot: {'sliding': [0.8, 0.8], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]}
                    for foot in ('FR', 'FL', 'RR', 'RL')
                },
            },
            base_mass={'enabled': False},
            base_com={'enabled': False},
            friction_phases=(FrictionPhase(5, 1.0), FrictionPhase(5, 1.0), FrictionPhase(5, 1.0)),
        ),
        EvaluationCondition(
            name='switch',
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [1.0, 1.0],
                'base_ranges': {
                    foot: {'sliding': [0.8, 0.8], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]}
                    for foot in ('FR', 'FL', 'RR', 'RL')
                },
            },
            base_mass={'enabled': False},
            base_com={'enabled': False},
            friction_phases=(FrictionPhase(5, 1.0), FrictionPhase(5, 0.5), FrictionPhase(5, 1.0)),
        ),
    )
    motions = {'': (MotionTask('forward_0.5', 0.5, yaw_rate=0, duration_seconds=15),)}
    metrics = 'friction'
    report = {'command': [0.5, 0, 0], 'segment_seconds': 5}
    episode_seconds = 16
    continue_after_reset = False
    reset_before_start = False
    backends = ('mujoco',)
