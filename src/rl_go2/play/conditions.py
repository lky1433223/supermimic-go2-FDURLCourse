"""Fixed disturbance settings and presets for the interactive player."""

from dataclasses import asdict, dataclass, replace

from rl_go2.env import QuadrupedEnv
from rl_go2.evaluation.conditions import NOMINAL_SLIDING
from rl_go2.evaluation.tasks.sensor_action import OBSERVATION_NOISE


@dataclass(frozen=True)
class PlayCondition:
    """Store fixed settings as percentages, centimeters and control steps."""

    friction: int = 80
    mass: int = 100
    com_x: int = 0
    com_y: int = 0
    com_z: int = 0
    noise: int = 0
    sensor_delay: int = 0
    action_delay: int = 0

    def name(self) -> str:
        """Use the matching preset name or indicate custom settings."""
        return next((label for label, preset in PRESETS if preset == self), 'Custom')

    def apply(self, env: QuadrupedEnv, previous: 'PlayCondition') -> bool:
        """Apply updated settings and report whether the robot was reset."""
        if self.friction != previous.friction:
            env.set_fixed_friction_multiplier(self.friction / 100 / NOMINAL_SLIDING[0])
        if replace(self, friction=previous.friction) == previous:
            return False
        noise = {
            channel: entries for channel, entries in OBSERVATION_NOISE.items()
            if self.noise and channel != 'frame_delay_steps'
        }
        noise['frame_delay_steps'] = [self.sensor_delay, self.sensor_delay]
        env.reset_with_episode_parameters(
            mass_scale=self.mass / 100,
            com_offset_m=(self.com_x / 100, self.com_y / 100, self.com_z / 100),
            observation_noise=noise,
            action_delay_steps=self.action_delay,
        )
        return True


NOMINAL = PlayCondition()
PRESETS = (
    ('Nominal', NOMINAL),
    ('Low friction', replace(NOMINAL, friction=20)),
    ('Heavy', replace(NOMINAL, mass=150)),
    ('Lateral CoM', replace(NOMINAL, com_y=5)),
    ('Noise and delays', replace(NOMINAL, noise=1, sensor_delay=1, action_delay=1)),
    ('Combined', PlayCondition(friction=60, mass=150, com_y=5, noise=1, sensor_delay=1, action_delay=1)),
)
# Field, label, minimum, maximum, increment. The preset row comes first.
SETTINGS = (
    ('friction', 'Sliding friction', 10, 100, 5),
    ('mass', 'Base mass', 85, 150, 5),
    ('com_x', 'CoM X', -5, 5, 1),
    ('com_y', 'CoM Y', -5, 5, 1),
    ('com_z', 'CoM Z', -5, 5, 1),
    ('noise', 'Observation noise', 0, 1, 1),
    ('sensor_delay', 'Observation delay', 0, 1, 1),
    ('action_delay', 'Action delay', 0, 1, 1),
)


def adjust_condition(condition: PlayCondition, row: int, direction: int) -> PlayCondition:
    """Step within the selected field or cycle presets."""
    if row == 0:
        for index, (_, preset) in enumerate(PRESETS):
            if condition == preset:
                return PRESETS[(index + direction) % len(PRESETS)][1]
        return PRESETS[0 if direction > 0 else -1][1]
    field, _, low, high, increment = SETTINGS[row - 1]
    value = asdict(condition)[field]
    return replace(condition, **{field: min(high, max(low, value + increment * direction))})


def condition_rows(condition: PlayCondition, step_ms: float) -> tuple[tuple[str, str], ...]:
    """Return settings labels and formatted values in navigation order."""
    values = (
        f'{condition.friction / 100:.2f}',
        f'{condition.mass / 100:.2f}x',
        f'{condition.com_x:+d} cm', f'{condition.com_y:+d} cm', f'{condition.com_z:+d} cm',
        'On' if condition.noise else 'Off',
        f'{condition.sensor_delay * step_ms:g} ms', f'{condition.action_delay * step_ms:g} ms',
    )
    return (('Preset', condition.name()), *(
        (label, value) for (_, label, _, _, _), value in zip(SETTINGS, values, strict=True)
    ))
