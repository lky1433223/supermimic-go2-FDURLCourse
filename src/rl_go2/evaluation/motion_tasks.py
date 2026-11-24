"""Velocity commands and durations for individual evaluation motions."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MotionTask:
    name: str
    vx: float
    yaw_rate: float = 0.0
    duration_seconds: float = 20.0
    stop_at_seconds: float | None = None

    def validate(self) -> None:
        """Require finite commands and a positive duration."""
        if not all(math.isfinite(value) for value in (self.vx, self.yaw_rate, self.duration_seconds)):
            raise ValueError('Motion commands and duration must be finite')
        if self.duration_seconds <= 0:
            raise ValueError('Motion duration must be positive')
        if self.stop_at_seconds is not None and not 0 < self.stop_at_seconds < self.duration_seconds:
            raise ValueError('Stop time must be inside the motion')

    @property
    def command(self) -> tuple[float, float, float]:
        return (self.vx, 0, self.yaw_rate)

    def command_schedule(self, dt: float) -> dict[int, tuple[float, float, float]]:
        schedule = {0: self.command}
        if self.stop_at_seconds is not None:
            schedule[round(self.stop_at_seconds / dt)] = (0, 0, 0)
        return schedule
