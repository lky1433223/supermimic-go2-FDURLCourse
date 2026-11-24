"""Genesis native key events and a lightweight text HUD plugin."""
import pyglet
from pyglet.window import key, mouse
from genesis.vis.viewer_plugins import EVENT_HANDLED, MouseInteractionPlugin, ViewerPlugin

from .controls import Controls
from .hud import MARGIN, PANEL, WIDTH, layout_hud


class ForceDragPlugin(MouseInteractionPlugin):
    """Use Genesis spring-force dragging and release grips on focus/control changes."""

    def __init__(self) -> None:
        super().__init__(use_force=True, spring_slack=0.02)

    def on_deactivate(self) -> None:
        self.on_mouse_release(0, 0, mouse.LEFT, 0)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol in (key.P, key.R, key.SPACE, key.ESCAPE):
            self.on_mouse_release(0, 0, mouse.LEFT, 0)


class PlaybackPlugin(ViewerPlugin):
    """Consume playback keys before Genesis default shortcuts and draw a HUD."""

    def __init__(self, controls: Controls):
        super().__init__()
        self.controls = controls
        self.background = None
        self.dividers: list[pyglet.shapes.Rectangle] = []
        self.labels: list[pyglet.text.Label] = []
        self.held: set[int] = set()
        self.mapping = {key.W: 'w', key.A: 'a', key.S: 's', key.D: 'd',
                        key.UP: 'up', key.DOWN: 'down', key.LEFT: 'left', key.RIGHT: 'right',
                        key.SPACE: 'stop', key.P: 'pause', key.R: 'reset', key.ESCAPE: 'quit', key.F1: 'details'}

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol not in self.mapping:
            return None
        self.controls.key_event(self.mapping[symbol], True, symbol in self.held)
        self.held.add(symbol)
        return EVENT_HANDLED

    def on_key_release(self, symbol: int, modifiers: int) -> bool | None:
        self.held.discard(symbol)
        if symbol in self.mapping:
            self.controls.key_event(self.mapping[symbol], False)
            return EVENT_HANDLED
        return None

    def on_deactivate(self) -> None:
        """Pyglet dispatches focus events to registered plugin handlers."""
        self.held.clear()
        self.controls.clear_keys()

    def on_draw(self) -> None:
        panel = layout_hud(self.controls.status())
        top = self.viewer.height - MARGIN
        if self.background is None:
            self.background = pyglet.shapes.Rectangle(MARGIN, 0, WIDTH, 0, color=PANEL)
        self.background.y = top - panel.height
        self.background.height = panel.height
        self.background.draw()
        while len(self.dividers) < len(panel.dividers):
            self.dividers.append(pyglet.shapes.Rectangle(
                MARGIN + 16, 0, WIDTH - 32, 1, color=(64, 82, 102, 204)))
        for divider, offset in zip(self.dividers, panel.dividers):
            divider.y = top - offset
            divider.draw()
        while len(self.labels) < len(panel.labels):
            self.labels.append(pyglet.text.Label('', font_name='Menlo', font_size=10))
        for label, entry in zip(self.labels, panel.labels):
            label.x = MARGIN + entry.x
            label.y = top - entry.y
            label.font_size = 12 if entry.large else 10
            label.color = (*entry.color, 255)
            label.text = entry.value
            label.draw()

    def on_close(self) -> None:
        self.controls.clear_keys()
        if self.background is not None:
            self.background.delete()
            self.background = None
        for item in (*self.labels, *self.dividers):
            item.delete()
        self.labels.clear()
        self.dividers.clear()


class GenesisPlayViewer:
    """Attach playback input/HUD to the existing Genesis window."""

    def __init__(self, viewer, controls: Controls):
        self.viewer = viewer
        self.plugin = PlaybackPlugin(controls)
        self.force_drag = ForceDragPlugin()
        viewer.add_plugin(self.plugin)
        viewer.add_plugin(self.force_drag)

    def refresh(self) -> bool:
        """Keep events and HUD responsive even while physics is paused."""
        if not self.viewer.is_alive():
            return False
        self.viewer.update()
        return self.viewer.is_alive()

    def close(self) -> None:
        """Detach the plugin before the backend destroys the window."""
        if self.viewer.is_alive():
            self.viewer.remove_plugin(self.force_drag)
            self.viewer.remove_plugin(self.plugin)
        self.plugin.on_close()

    def release_forces(self) -> None:
        """Release the spring grab after a reset or pause."""
        self.force_drag.on_mouse_release(0, 0, mouse.LEFT, 0)
