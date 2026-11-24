"""Shared layout for the native playback HUD, in logical pixels."""

from dataclasses import dataclass
from textwrap import wrap


Color = tuple[int, int, int]
TEXT: Color = (239, 245, 252)
MUTED: Color = (161, 179, 197)
CYAN: Color = (105, 224, 233)
PANEL = (15, 24, 36, 229)
MARGIN = 16
WIDTH = 400


@dataclass(frozen=True)
class Text:
    """A left-aligned label positioned from the panel's top-left corner."""

    x: int
    y: int
    value: str
    color: Color = TEXT
    large: bool = False


@dataclass(frozen=True)
class Panel:
    """Panel height, divider positions and labels in logical pixels."""

    height: int
    dividers: tuple[int, ...]
    labels: tuple[Text, ...]


def layout_hud(status: dict) -> Panel:
    """Arrange live telemetry first and append configuration when F1 is open."""
    forward, turn = status.get('command', (0.0, 0.0))
    actual_forward, actual_turn = status.get('actual', (0.0, 0.0))
    paused = status['paused']
    backend = status.get('backend')
    labels = [
        Text(16, 35, 'Go2 / SuperMimic', large=True),
        Text(292, 35, 'PAUSED' if paused else 'RUNNING',
             (255, 204, 112) if paused else (121, 224, 162)),
        Text(16, 51, f"{'MuJoCo' if backend == 'mujoco' else 'Genesis'}  /  1x target", MUTED),
        Text(166, 51, f"{status.get('terrain', '')}  {status.get('terrain_progress', '')}  [N]", CYAN),
        Text(16, 77, 'BODY VELOCITY', MUTED),
        Text(166, 77, 'TARGET', MUTED),
        Text(268, 77, 'ACTUAL', CYAN),
        Text(16, 109, 'Forward (m/s)'),
        Text(166, 109, f'{forward:+.2f}', large=True),
        Text(268, 109, f'{actual_forward:+.2f}', CYAN, large=True),
        Text(16, 136, 'Turn (rad/s)'),
        Text(166, 136, f'{turn:+.2f}', large=True),
        Text(268, 136, f'{actual_turn:+.2f}', CYAN, large=True),
        Text(16, 159, f"Speed {status['speed']:.2f} m/s  Up/Down", MUTED),
        Text(208, 159, f"Turn {status['yaw']:.2f} rad/s  Left/Right", MUTED),
        Text(16, 190, 'MOUSE FORCE', MUTED),
    ]
    dividers = [60, 171]
    bottom = 201
    body = status.get('mouse_body')
    if backend == 'mujoco' and body and body != 'none':
        labels.extend([
            Text(16, 223, f"{status.get('mouse_force_n', 0.0):.1f} N", CYAN, large=True),
            Text(166, 223, body),
        ])
        bottom = 234
    else:
        message = 'No grab' if backend == 'mujoco' else 'Spring drag'
        labels.append(Text(166, 190, message))
    dividers.append(bottom)
    details = status.get('details', False)
    labels.extend([
        Text(16, bottom + 19, 'W/S  Move    A/D  Turn    Space  Stop', MUTED),
        Text(16, bottom + 38, 'P  Pause    R  Reset    Esc  Exit', MUTED),
        Text(294, bottom + 38, 'F1  Hide' if details else 'F1  Details', CYAN),
    ])
    bottom += 50
    if details:
        dividers.append(bottom)
        iteration = status.get('iteration')
        iteration_text = f'Iteration {iteration}' if iteration is not None else 'Iteration unavailable'
        lines = [
            f"{iteration_text}  /  Sim {status.get('seconds', 0.0):.1f} s",
            *wrap(str(status.get('policy', '')), width=46),
            f"Training vx/vy/yaw {status.get('training_ranges', '')}",
            str(status.get('physics', '')),
            'Left drag body: force  /  Empty space: orbit',
            'Right drag: pan  /  Scroll: zoom',
        ]
        for line in lines:
            for paragraph in wrap(line, width=46):
                bottom += 18
                labels.append(Text(16, bottom, paragraph, MUTED))
        bottom += 12
    return Panel(bottom, tuple(dividers), tuple(labels))
