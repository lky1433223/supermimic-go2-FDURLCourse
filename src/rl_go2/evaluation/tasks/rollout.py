"""A fixed straight-walking smoke test with explicit physical conditions."""
from ..conditions import EvaluationCondition
from ..motion_tasks import MotionTask
from ..task import EvaluationTask


class RolloutTask(EvaluationTask):
    TASK = 'rollout'
    TASK_DESCRIPTION = 'Run fixed 0.5 m/s straight walking for a step budget.'

    conditions = (
        EvaluationCondition(
            name='nominal',
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [0.75, 1.25],
                'base_ranges': {
                    foot: {'sliding': [0.8, 0.8], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]}
                    for foot in ('FR', 'FL', 'RR', 'RL')
                },
            },
            base_mass={'enabled': True, 'scale_range': [0.9, 1.1]},
            base_com={'enabled': True, 'offset_ranges_m': {
                'x': [-0.01, 0.01], 'y': [-0.03, 0.03], 'z': [-0.01, 0.01],
            }},
        ),
    )
    motions = {'': (MotionTask('forward_0.5', 0.5, yaw_rate=0, duration_seconds=20),)}
    metrics = 'rollout'
    report = {}
    episode_seconds = 20
    continue_after_reset = True
    reset_before_start = False
    supports_parallel = True
    default_steps = 1000
