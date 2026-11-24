"""Velocity tracking under four explicit low-friction conditions."""
from ..conditions import EvaluationCondition
from ..motion_tasks import MotionTask
from ..task import EvaluationTask


class LowFrictionTask(EvaluationTask):
    TASK = 'low-friction'
    TASK_DESCRIPTION = 'Run 27 tracking cases under each of four low-friction conditions.'

    conditions = (
        EvaluationCondition(
            name='uniform-01',
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [1.0, 1.0],
                'base_ranges': {
                    'FR': {'sliding': [0.1, 0.1], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'FL': {'sliding': [0.1, 0.1], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RR': {'sliding': [0.1, 0.1], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RL': {'sliding': [0.1, 0.1], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                },
            },
            base_mass={'enabled': True, 'scale_range': [0.9, 1.1]},
            base_com={'enabled': True, 'offset_ranges_m': {
                'x': [-0.01, 0.01], 'y': [-0.03, 0.03], 'z': [-0.01, 0.01],
            }},
        ),
        EvaluationCondition(
            name='uniform-02',
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [1.0, 1.0],
                'base_ranges': {
                    'FR': {'sliding': [0.2, 0.2], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'FL': {'sliding': [0.2, 0.2], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RR': {'sliding': [0.2, 0.2], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RL': {'sliding': [0.2, 0.2], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                },
            },
            base_mass={'enabled': True, 'scale_range': [0.9, 1.1]},
            base_com={'enabled': True, 'offset_ranges_m': {
                'x': [-0.01, 0.01], 'y': [-0.03, 0.03], 'z': [-0.01, 0.01],
            }},
        ),
        EvaluationCondition(
            name='left-low',
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [1.0, 1.0],
                'base_ranges': {
                    'FR': {'sliding': [0.6, 0.6], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'FL': {'sliding': [0.1, 0.1], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RR': {'sliding': [0.8, 0.8], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RL': {'sliding': [0.2, 0.2], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                },
            },
            base_mass={'enabled': True, 'scale_range': [0.9, 1.1]},
            base_com={'enabled': True, 'offset_ranges_m': {
                'x': [-0.01, 0.01], 'y': [-0.03, 0.03], 'z': [-0.01, 0.01],
            }},
        ),
        EvaluationCondition(
            name='right-low',
            foot_ground_friction={
                'enabled': True,
                'env_multiplier_range': [1.0, 1.0],
                'base_ranges': {
                    'FR': {'sliding': [0.1, 0.1], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'FL': {'sliding': [0.6, 0.6], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RR': {'sliding': [0.2, 0.2], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                    'RL': {'sliding': [0.8, 0.8], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]},
                },
            },
            base_mass={'enabled': True, 'scale_range': [0.9, 1.1]},
            base_com={'enabled': True, 'offset_ranges_m': {
                'x': [-0.01, 0.01], 'y': [-0.03, 0.03], 'z': [-0.01, 0.01],
            }},
        ),
    )
    motions = {
        'standard': (
            MotionTask('vx0.2_yaw0', 0.2),
            MotionTask('vx0.2_yaw0.4', 0.2, 0.4),
            MotionTask('vx0.2_yaw-0.4', 0.2, -0.4),
            MotionTask('vx0.4_yaw0', 0.4),
            MotionTask('vx0.4_yaw0.4', 0.4, 0.4),
            MotionTask('vx0.4_yaw-0.4', 0.4, -0.4),
            MotionTask('vx0.6_yaw0', 0.6),
            MotionTask('vx0.6_yaw0.4', 0.6, 0.4),
            MotionTask('vx0.6_yaw-0.4', 0.6, -0.4),
            MotionTask('vx0.8_yaw0', 0.8),
            MotionTask('vx0.8_yaw0.4', 0.8, 0.4),
            MotionTask('vx0.8_yaw-0.4', 0.8, -0.4),
            MotionTask('vx0.5_yaw0', 0.5),
            MotionTask('vx0.5_yaw0.1', 0.5, 0.1),
            MotionTask('vx0.5_yaw-0.1', 0.5, -0.1),
            MotionTask('vx0.5_yaw0.3', 0.5, 0.3),
            MotionTask('vx0.5_yaw-0.3', 0.5, -0.3),
            MotionTask('vx0.5_yaw0.5', 0.5, 0.5),
            MotionTask('vx0.5_yaw-0.5', 0.5, -0.5),
        ),
        'ood': (
            MotionTask('stand', 0),
            MotionTask('walk_to_stop', 0.5, stop_at_seconds=8),
            MotionTask('vx1.0_yaw0', 1.0),
            MotionTask('vx1.2_yaw0', 1.2),
            MotionTask('vx0.5_yaw0.6', 0.5, 0.6),
            MotionTask('vx0.5_yaw-0.6', 0.5, -0.6),
            MotionTask('vx0.5_yaw0.8', 0.5, 0.8),
            MotionTask('vx0.5_yaw-0.8', 0.5, -0.8),
        ),
    }

    metrics = 'tracking'
    report = {
        'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20,
        'stop_criteria': {'planar_speed_m_s': 0.1, 'abs_yaw_rad_s': 0.15, 'hold_seconds': 0.5},
    }
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True
