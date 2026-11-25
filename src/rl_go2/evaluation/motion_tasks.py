"""Velocity commands and durations for individual evaluation motions."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class MotionTask:
    """A body-frame velocity command in m/s and rad/s."""

    name: str
    vx: float
    yaw_rate: float = 0.0
    duration_seconds: float = 20.0
    stop_at_seconds: float | None = None
    vy: float = field(default=0.0, kw_only=True)

    def validate(self) -> None:
        """Require finite commands and a positive duration."""
        if not all(math.isfinite(value) for value in (*self.command, self.duration_seconds)):
            raise ValueError('Motion commands and duration must be finite')
        if self.duration_seconds <= 0:
            raise ValueError('Motion duration must be positive')
        if self.stop_at_seconds is not None and not 0 < self.stop_at_seconds < self.duration_seconds:
            raise ValueError('Stop time must be inside the motion')

    @property
    def command(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.yaw_rate)

    def command_schedule(self, dt: float) -> dict[int, tuple[float, float, float]]:
        schedule = {0: self.command}
        if self.stop_at_seconds is not None:
            schedule[round(self.stop_at_seconds / dt)] = (0, 0, 0)
        return schedule


FRICTION_MOTIONS = (
    MotionTask('stand', 0),
    MotionTask('vx0.2_yaw0', 0.2),
    MotionTask('vx0.5_yaw0', 0.5),
    MotionTask('vx0.5_yaw0.4', 0.5, 0.4),
    MotionTask('vx0.5_yaw-0.4', 0.5, -0.4),
    MotionTask('walk_to_stop', 0.5, stop_at_seconds=8),
)
