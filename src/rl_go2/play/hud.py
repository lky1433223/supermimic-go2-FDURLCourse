"""Shared layout for the native playback HUD, in logical pixels."""

from dataclasses import dataclass
from textwrap import wrap

from .conditions import condition_rows


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
    if status['condition_draft'] is not None:
        return layout_condition_menu(status)

    forward, lateral, turn = status.get('command', (0.0, 0.0, 0.0))
    actual_forward, actual_lateral, actual_turn = status.get('actual', (0.0, 0.0, 0.0))
    speeds = status['speeds']
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
        Text(16, 136, 'Lateral (m/s)'),
        Text(166, 136, f'{lateral:+.2f}', large=True),
        Text(268, 136, f'{actual_lateral:+.2f}', CYAN, large=True),
        Text(16, 163, 'Turn (rad/s)'),
        Text(166, 163, f'{turn:+.2f}', large=True),
        Text(268, 163, f'{actual_turn:+.2f}', CYAN, large=True),
        Text(16, 186, f"W/S {speeds['w']:.2f}/{speeds['s']:.2f}", MUTED),
        Text(137, 186, f"A/D {speeds['a']:.2f}/{speeds['d']:.2f}", MUTED),
        Text(258, 186, f"Q/E {speeds['q']:.2f}/{speeds['e']:.2f}", MUTED),
        Text(16, 217, 'MOUSE FORCE', MUTED),
    ]
    dividers = [60, 198]
    bottom = 228
    body = status.get('mouse_body')
    if backend == 'mujoco' and body and body != 'none':
        labels.extend([
            Text(16, 250, f"{status.get('mouse_force_n', 0.0):.1f} N", CYAN, large=True),
            Text(166, 250, body),
        ])
        bottom = 261
    else:
        message = 'No grab' if backend == 'mujoco' else 'Spring drag'
        labels.append(Text(166, 217, message))
    dividers.append(bottom)
    details = status.get('details', False)
    condition = status['condition']
    step_ms = status.get('step_ms', 20.0)
    labels.extend([
        Text(16, bottom + 19, f'{condition.name()}   [F2 Conditions]', CYAN),
        Text(16, bottom + 38,
             f"Friction {condition.friction / 100:.2f}   Mass {condition.mass / 100:.2f}x"
             f"   CoM {condition.com_x:+d}/{condition.com_y:+d}/{condition.com_z:+d} cm", MUTED),
        Text(16, bottom + 57,
             f"Noise {'on' if condition.noise else 'off'}   Delay obs/action "
             f"{condition.sensor_delay * step_ms:g}/{condition.action_delay * step_ms:g} ms", MUTED),
    ])
    bottom += 69
    dividers.append(bottom)
    labels.extend([
        Text(16, bottom + 19, 'W/S  Forward/Back   A/D  Strafe   Q/E  Turn', MUTED),
        Text(16, bottom + 38, 'Up/Down  Adjust with one movement key held', MUTED),
        Text(16, bottom + 57, 'Space  Stop   P  Pause   R  Reset   Esc  Exit', MUTED),
        Text(317, bottom + 57, 'F1  Hide' if details else 'F1  Details', CYAN),
    ])
    bottom += 69
    if details:
        dividers.append(bottom)
        iteration = status.get('iteration')
        iteration_text = f'Iteration {iteration}' if iteration is not None else 'Iteration unavailable'
        lines = [
            f"{iteration_text}  /  Sim {status.get('seconds', 0.0):.1f} s",
            *wrap(str(status.get('policy', '')), width=46),
            f"Training vx/vy/yaw {status.get('training_ranges', '')}",
            'Left drag body: force  /  Empty space: orbit',
            'Right drag: pan  /  Scroll: zoom',
        ]
        for line in lines:
            for paragraph in wrap(line, width=46):
                bottom += 18
                labels.append(Text(16, bottom, paragraph, MUTED))
        bottom += 12
    return Panel(bottom, tuple(dividers), tuple(labels))


def layout_condition_menu(status: dict) -> Panel:
    """Display the draft settings, selected row and apply state."""
    draft = status['condition_draft']
    rows = condition_rows(draft, status.get('step_ms', 20.0))
    labels = [
        Text(16, 35, 'Playback Conditions', large=True),
        Text(16, 56, f"Active: {status['condition'].name()}", MUTED),
        Text(244, 56, 'Unapplied changes' if draft != status['condition'] else 'Simulation paused', CYAN),
    ]
    for index, (label, value) in enumerate(rows):
        selected = index == status['condition_row']
        color = CYAN if selected else TEXT
        y = 90 + index * 23
        if selected:
            labels.append(Text(16, y, '>', color))
        labels.extend((Text(34, y, label, color), Text(223, y, value, color)))
    bottom = 90 + len(rows) * 23 - 8
    labels.extend([
        Text(16, bottom + 19, 'Up/Down  Select   Left/Right  Change', MUTED),
        Text(16, bottom + 38, 'Enter  Apply   F2  Close', MUTED),
    ])
    return Panel(bottom + 50, (66, bottom), tuple(labels))
