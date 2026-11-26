"""Thread-safe native keyboard state and body-frame velocity commands."""
from dataclasses import dataclass, field
from threading import Lock

from .conditions import NOMINAL, SETTINGS, PlayCondition, adjust_condition


MOTION_KEYS = (('w', 's'), ('a', 'd'), ('q', 'e'))


@dataclass
class Controls:
    """Own keyboard state. Linear and yaw speeds use m/s and rad/s."""

    command_limits: tuple[float, float, float]  # Maximum magnitudes for vx, vy, yaw.
    speed_limits: dict[str, float] = field(init=False)
    speeds: dict[str, float] = field(init=False)
    details: bool = False
    paused: bool = False
    stopped: bool = False
    reset_requested: bool = False
    terrain_requested: bool = False
    condition: PlayCondition = NOMINAL
    condition_draft: PlayCondition | None = None
    condition_row: int = 0
    condition_request: PlayCondition | None = None
    keys: set[str] = field(default_factory=set)
    telemetry: dict = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.speed_limits = {
            key: limit for pair, limit in zip(MOTION_KEYS, self.command_limits, strict=True) for key in pair
        }
        self.speeds = {key: min(0.5, limit) for key, limit in self.speed_limits.items()}

    def key_event(self, key: str, pressed: bool, repeat: bool = False) -> None:
        """Apply a native key event, ignoring repeats for discrete actions."""
        with self.lock:
            if key in self.speeds:
                if pressed and not repeat and self.condition_draft is None:
                    self.keys.add(key)
                elif not pressed:
                    self.keys.discard(key)
                return
            if not pressed or repeat:
                return
            if key == 'conditions':
                self.keys.clear()
                self.condition_draft = self.condition if self.condition_draft is None else None
            elif self.condition_draft is not None and key != 'quit':
                self._edit_condition(key)
            elif key == 'details':
                self.details = not self.details
            elif key in {'up', 'down'} and len(self.keys) == 1:
                (motion,) = self.keys
                delta = 0.1 if key == 'up' else -0.1
                speed = round(self.speeds[motion] + delta, 2)
                self.speeds[motion] = max(0.0, min(self.speed_limits[motion], speed))
            elif key == 'pause':
                self.paused = not self.paused
                self.keys.clear()
            elif key == 'next_terrain':
                self.terrain_requested = True
            elif key == 'reset':
                self.reset_requested = True
                self.keys.clear()
            elif key == 'stop':
                self.keys.clear()
            elif key == 'quit':
                self.stopped = True
                self.keys.clear()

    def _edit_condition(self, key: str) -> None:
        assert self.condition_draft is not None
        if key in {'up', 'down'}:
            direction = -1 if key == 'up' else 1
            self.condition_row = (self.condition_row + direction) % (1 + len(SETTINGS))
        elif key in {'left', 'right'}:
            direction = -1 if key == 'left' else 1
            self.condition_draft = adjust_condition(self.condition_draft, self.condition_row, direction)
        elif key == 'apply_condition':
            self.condition_request = self.condition_draft

    def clear_keys(self) -> None:
        """Clear movement when focus is lost, a fall occurs, or the robot resets."""
        with self.lock:
            self.keys.clear()

    def sample(self) -> tuple[list[float], bool, bool, bool]:
        """Return command, pause, exit and consume the reset request."""
        with self.lock:
            command = []
            for positive, negative in MOTION_KEYS:
                positive_held = positive in self.keys
                negative_held = negative in self.keys
                if positive_held == negative_held:
                    command.append(0.0)
                elif positive_held:
                    command.append(self.speeds[positive])
                else:
                    command.append(-self.speeds[negative])
            reset = self.reset_requested
            self.reset_requested = False
            return command, self.paused or self.condition_draft is not None, self.stopped, reset

    def consume_terrain_switch(self) -> bool:
        """Return and clear the next-terrain request from the native viewer."""
        with self.lock:
            requested = self.terrain_requested
            self.terrain_requested = False
            return requested

    def consume_condition_request(self) -> PlayCondition | None:
        """Return the most recent settings confirmed by Enter."""
        with self.lock:
            requested = self.condition_request
            self.condition_request = None
            return requested

    def confirm_condition(self, condition: PlayCondition) -> None:
        """Display the settings after the simulation has applied them."""
        with self.lock:
            self.condition = condition

    def status(self) -> dict:
        """Return a consistent HUD snapshot."""
        with self.lock:
            return {**self.telemetry, 'speeds': self.speeds.copy(),
                    'condition': self.condition, 'condition_draft': self.condition_draft,
                    'condition_row': self.condition_row,
                    'paused': self.paused, 'stopped': self.stopped, 'details': self.details}

    def publish(self, **values: object) -> None:
        """Update simulation telemetry without exposing mutable shared state."""
        with self.lock:
            self.telemetry.update(values)
