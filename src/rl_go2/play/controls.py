"""Thread-safe keyboard state and bounded body-frame velocity commands."""
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic


@dataclass
class Controls:
    """Own input state shared by the HTTP panel and simulation thread.

    Linear speed is in m/s and yaw speed in rad/s. A heartbeat timeout clears
    held keys when the browser disappears without sending a release event.
    """

    max_speed: float
    max_yaw: float
    speed: float = 0.5
    yaw: float = 0.5
    paused: bool = False
    stopped: bool = False
    reset_requested: bool = False
    keys: set[str] = field(default_factory=set)
    last_input: float = field(default_factory=monotonic)
    telemetry: dict = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.speed = min(self.speed, self.max_speed)
        self.yaw = min(self.yaw, self.max_yaw)

    def receive(self, keys: list[str], action: str = '') -> None:
        """Replace held keys and apply one discrete UI action."""
        with self.lock:
            self.last_input = monotonic()
            self.keys = set(keys) & {'w', 'a', 's', 'd'}
            if action == 'up':
                self.speed = min(self.max_speed, round(self.speed + 0.1, 2))
            elif action == 'down':
                self.speed = max(0.0, round(self.speed - 0.1, 2))
            elif action == 'right':
                self.yaw = min(self.max_yaw, round(self.yaw + 0.1, 2))
            elif action == 'left':
                self.yaw = max(0.0, round(self.yaw - 0.1, 2))
            elif action == 'pause':
                self.paused = not self.paused
                self.keys.clear()
            elif action == 'reset':
                self.reset_requested = True
                self.keys.clear()
            elif action == 'stop':
                self.keys.clear()
            elif action == 'quit':
                self.stopped = True
                self.keys.clear()

    def sample(self) -> tuple[list[float], bool, bool, bool]:
        """Return command, pause, exit and consume the reset request."""
        with self.lock:
            if monotonic() - self.last_input > 0.6:
                self.keys.clear()
            command = [self.speed * (int('w' in self.keys) - int('s' in self.keys)),
                       0.0, self.yaw * (int('a' in self.keys) - int('d' in self.keys))]
            reset = self.reset_requested
            self.reset_requested = False
            return command, self.paused, self.stopped, reset

    def status(self) -> dict:
        """Return a consistent JSON-compatible panel snapshot."""
        with self.lock:
            return {**self.telemetry, 'speed': self.speed, 'yaw': self.yaw,
                    'max_speed': self.max_speed, 'max_yaw': self.max_yaw,
                    'paused': self.paused, 'stopped': self.stopped}

    def publish(self, **values: object) -> None:
        """Update simulation telemetry without exposing mutable shared state."""
        with self.lock:
            self.telemetry.update(values)
