"""Compare timed friction changes across shared motions and body samples."""
from ..conditions import FrictionPhase, NOMINAL_SLIDING, PHYSICAL_SAMPLES
from ..motion_tasks import FRICTION_MOTIONS
from ..task import EvaluationTask


FRICTION_SCHEDULES = (
    ('fixed', (FrictionPhase(5, 1.0), FrictionPhase(5, 1.0), FrictionPhase(10, 1.0))),
    ('switch', (FrictionPhase(5, 1.0), FrictionPhase(5, 0.5), FrictionPhase(10, 1.0))),
    ('switch-low', (FrictionPhase(5, 1.0), FrictionPhase(5, 0.25), FrictionPhase(10, 1.0))),
    ('switch-short', (FrictionPhase(5, 1.0), FrictionPhase(2, 0.25), FrictionPhase(13, 1.0))),
)


class FrictionSwitchTask(EvaluationTask):
    TASK = 'friction-switch'
    TASK_DESCRIPTION = 'Run six motions under four friction schedules and three body samples.'

    conditions = tuple(
        sample.condition(f'{sample.name}-{name}', NOMINAL_SLIDING, friction_phases=phases)
        for sample in PHYSICAL_SAMPLES
        for name, phases in FRICTION_SCHEDULES
    )
    motions = {'tracking': FRICTION_MOTIONS}
    metrics = 'friction'
    report = {
        'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': 20,
        'stop_criteria': {'planar_speed_m_s': 0.1, 'abs_yaw_rad_s': 0.15, 'hold_seconds': 0.5},
    }
    episode_seconds = 20
    continue_after_reset = False
    reset_before_start = True
    backends = ('mujoco',)
