"""Reuse standard tracking cases with fixed per-foot sliding friction."""
from rl_go2.config import FOOT_NAMES, foot_friction_ranges

from . import standard

TASK = 'low-friction'
TASK_DESCRIPTION = 'Run standard tracking cases with fixed per-foot sliding friction.'


def validate_parameters(parameters):
    sliding = parameters.get('sliding_friction')
    if set(parameters) != {'sliding_friction'} or not isinstance(sliding, dict) or set(sliding) != set(FOOT_NAMES):
        raise ValueError('low-friction requires parameters.sliding_friction for FL, FR, RL and RR')
    if any(type(value) not in (int, float) or not 0.01 <= value <= 3.4028234663852886e38
           for value in sliding.values()):
        raise ValueError('Sliding friction must be at least 0.01 and fit in float32')


def configure(config, parameters):
    standard.configure(config, {})
    sliding = parameters['sliding_friction']
    domain = config.setdefault('domain_randomization', {})
    previous = domain.get('foot_ground_friction', {}).get('base_ranges', {})
    domain['foot_ground_friction'] = {
        'enabled': True,
        'env_multiplier_range': [1.0, 1.0],
        'base_ranges': {
            foot: {
                'sliding': [sliding[foot], sliding[foot]],
                'torsional': previous.get(foot, {}).get('torsional', [0.02, 0.02]),
                'rolling': previous.get(foot, {}).get('rolling', [0.01, 0.01]),
            }
            for foot in FOOT_NAMES
        },
    }
    foot_friction_ranges(config)


run = standard.run
