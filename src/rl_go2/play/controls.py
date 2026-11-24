"""Thread-safe native keyboard state and body-frame velocity commands."""
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Controls:
    """Own keyboard state. Linear and yaw speeds use m/s and rad/s."""

    max_speed: float
    max_yaw: float
    speed: float = 0.5
    yaw: float = 0.5
    paused: bool = False
    stopped: bool = False
    reset_requested: bool = False
    keys: set[str] = field(default_factory=set)
    telemetry: dict = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.speed = min(self.speed, self.max_speed)
        self.yaw = min(self.yaw, self.max_yaw)

    def key_event(self, key: str, pressed: bool, repeat: bool = False) -> None:
        """Apply a native key event, ignoring repeats for discrete actions."""
        with self.lock:
            if key in {'w', 'a', 's', 'd'}:
                if pressed:
                    self.keys.add(key)
                else:
                    self.keys.discard(key)
                return
            if not pressed or repeat:
                return
            if key == 'up':
                self.speed = min(self.max_speed, round(self.speed + 0.1, 2))
            elif key == 'down':
                self.speed = max(0.0, round(self.speed - 0.1, 2))
            elif key == 'right':
                self.yaw = min(self.max_yaw, round(self.yaw + 0.1, 2))
            elif key == 'left':
                self.yaw = max(0.0, round(self.yaw - 0.1, 2))
            elif key == 'pause':
                self.paused = not self.paused
                self.keys.clear()
            elif key == 'reset':
                self.reset_requested = True
                self.keys.clear()
            elif key == 'stop':
                self.keys.clear()
            elif key == 'quit':
                self.stopped = True
                self.keys.clear()

    def clear_keys(self) -> None:
        """Clear movement when focus is lost, a fall occurs, or the robot resets."""
        with self.lock:
            self.keys.clear()

    def sample(self) -> tuple[list[float], bool, bool, bool]:
        """Return command, pause, exit and consume the reset request."""
        with self.lock:
            command = [self.speed * (int('w' in self.keys) - int('s' in self.keys)),
                       0.0, self.yaw * (int('a' in self.keys) - int('d' in self.keys))]
            reset = self.reset_requested
            self.reset_requested = False
            return command, self.paused, self.stopped, reset

    def status(self) -> dict:
        """Return a consistent HUD snapshot."""
        with self.lock:
            return {**self.telemetry, 'speed': self.speed, 'yaw': self.yaw,
                    'max_speed': self.max_speed, 'max_yaw': self.max_yaw,
                    'paused': self.paused, 'stopped': self.stopped}

    def publish(self, **values: object) -> None:
        """Update simulation telemetry without exposing mutable shared state."""
        with self.lock:
            self.telemetry.update(values)


def hud_text(status: dict) -> str:
    """Format a compact ASCII HUD supported by both native font renderers."""
    command = status.get('command', [0.0, 0.0])
    actual = status.get('actual', [0.0, 0.0])
    state = 'PAUSED' if status['paused'] else 'RUNNING'
    return '\n'.join([
        f"SuperMimic | {status.get('backend', '')} | {state} | 1x",
        str(status.get('policy', '')),
        f"Speed {status['speed']:.2f} / {status['max_speed']:.2f} m/s"
        f"   Turn {status['yaw']:.2f} / {status['max_yaw']:.2f} rad/s",
        f"Command vx/yaw {command[0]:+.2f} / {command[1]:+.2f}",
        f"Actual  vx/yaw {actual[0]:+.2f} / {actual[1]:+.2f}"
        f"   Time {status.get('seconds', 0.0):.1f} s",
        f"Training vx / vy / yaw: {status.get('training_ranges', [])}",
        'W/S forward/back | A/D turn | Arrows adjust speed/turn',
        'Space stop | P pause | R reset | Esc exit',
        'Mouse drag orbit | Right drag pan | Scroll zoom',
    ])
