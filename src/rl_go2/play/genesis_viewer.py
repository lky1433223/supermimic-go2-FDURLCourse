"""Genesis native key events and a lightweight text HUD plugin."""
import pyglet
from pyglet.window import key
from genesis.vis.viewer_plugins import EVENT_HANDLED, ViewerPlugin

from .controls import Controls, hud_text


class PlaybackPlugin(ViewerPlugin):
    """Consume playback keys before Genesis default shortcuts and draw a HUD."""

    def __init__(self, controls: Controls):
        super().__init__()
        self.controls = controls
        self.label = None
        self.held: set[int] = set()
        self.mapping = {key.W: 'w', key.A: 'a', key.S: 's', key.D: 'd',
                        key.UP: 'up', key.DOWN: 'down', key.LEFT: 'left', key.RIGHT: 'right',
                        key.SPACE: 'stop', key.P: 'pause', key.R: 'reset', key.ESCAPE: 'quit'}

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
        if self.label is None:
            self.label = pyglet.text.Label(
                '', font_name='Menlo', font_size=12, x=16, y=0,
                width=1000, multiline=True, anchor_y='top',
                color=(255, 255, 255, 255))
        self.label.y = self.viewer.height - 16
        self.label.width = max(100, self.viewer.width - 32)
        self.label.text = hud_text(self.controls.status())
        self.label.draw()

    def on_close(self) -> None:
        self.controls.clear_keys()
        if self.label is not None:
            self.label.delete()
            self.label = None


class GenesisPlayViewer:
    """Attach playback input/HUD to the existing Genesis window."""

    def __init__(self, viewer, controls: Controls):
        self.viewer = viewer
        self.plugin = PlaybackPlugin(controls)
        viewer.add_plugin(self.plugin)

    def refresh(self) -> bool:
        """Keep events and HUD responsive even while physics is paused."""
        if not self.viewer.is_alive():
            return False
        self.viewer.update()
        return self.viewer.is_alive()

    def close(self) -> None:
        """Detach the plugin before the backend destroys the window."""
        if self.viewer.is_alive():
            self.viewer.remove_plugin(self.plugin)
        self.plugin.on_close()
