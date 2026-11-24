"""MuJoCo rendering with GLFW press/release callbacks and an in-window HUD."""
import ctypes
from copy import copy
import math
import sys

import glfw
import mujoco
import numpy as np

from .controls import Controls, hud_text


def playback_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Make a render-only copy with the brighter evaluation-video lighting."""
    visual = copy(model)
    visual.vis.headlight.active = 1
    visual.vis.headlight.ambient[:] = (0.6, 0.6, 0.6)
    visual.vis.headlight.diffuse[:] = (0.7, 0.7, 0.7)
    visual.vis.headlight.specular[:] = (0.2, 0.2, 0.2)
    planes = visual.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    visual.geom_rgba[planes] = (0.75, 0.78, 0.8, 1.0)
    return visual


class MujocoPlayViewer:
    """Own a GLFW window and MuJoCo render resources on the calling thread."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, controls: Controls):
        if sys.platform == 'darwin' and not ctypes.CDLL(None).pthread_main_np():
            raise RuntimeError('play.py requires plain python on macOS. Run: uv run --extra train python scripts/play.py ...')
        self.model = playback_model(model)
        self.data = data
        self.controls = controls
        self.last_cursor = (0.0, 0.0)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        self.camera.distance = 3.0
        self.camera.azimuth = 135
        self.camera.elevation = -25
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed')
        self.window = glfw.create_window(1280, 800, 'Go2 | SuperMimic | MuJoCo', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Cannot create MuJoCo playback window')
        try:
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_window_focus_callback(self.window, self._on_focus)
            glfw.set_cursor_pos_callback(self.window, self._on_cursor)
            glfw.set_scroll_callback(self.window, self._on_scroll)
        except Exception:
            glfw.destroy_window(self.window)
            glfw.terminate()
            raise

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        mapping = {glfw.KEY_W: 'w', glfw.KEY_A: 'a', glfw.KEY_S: 's', glfw.KEY_D: 'd',
                   glfw.KEY_UP: 'up', glfw.KEY_DOWN: 'down', glfw.KEY_LEFT: 'left',
                   glfw.KEY_RIGHT: 'right', glfw.KEY_SPACE: 'stop', glfw.KEY_P: 'pause',
                   glfw.KEY_R: 'reset', glfw.KEY_ESCAPE: 'quit'}
        if key in mapping:
            self.controls.key_event(mapping[key], action != glfw.RELEASE, action == glfw.REPEAT)

    def _on_focus(self, window, focused: int) -> None:
        if not focused:
            self.controls.clear_keys()

    def _on_cursor(self, window, x: float, y: float) -> None:
        dx, dy = x - self.last_cursor[0], y - self.last_cursor[1]
        self.last_cursor = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not left and not right:
            return
        height = max(glfw.get_window_size(window)[1], 1)
        if right:
            azimuth = math.radians(self.camera.azimuth)
            elevation = math.radians(self.camera.elevation)
            right_axis = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
            up_axis = np.array([-math.sin(elevation) * math.cos(azimuth),
                                -math.sin(elevation) * math.sin(azimuth), math.cos(elevation)])
            self.camera.lookat += self.camera.distance * (-dx * right_axis + dy * up_axis) / height
            return
        self.camera.azimuth -= 180 * dx / height
        self.camera.elevation = max(-89.0, min(89.0, self.camera.elevation - 180 * dy / height))

    def _on_scroll(self, window, dx: float, dy: float) -> None:
        self.camera.distance = max(0.1, min(100.0, self.camera.distance * math.exp(-0.1 * dy)))

    def refresh(self) -> bool:
        """Poll native events and render the scene and text overlay."""
        glfw.poll_events()
        if glfw.window_should_close(self.window):
            self.controls.clear_keys()
            return False
        width, height = glfw.get_framebuffer_size(self.window)
        if width == 0 or height == 0:
            return True
        glfw.make_context_current(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.options, None, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, hud_text(self.controls.status()), '', self.context)
        glfw.swap_buffers(self.window)
        return True

    def close(self) -> None:
        """Free GL resources while their window context is still current."""
        glfw.make_context_current(self.window)
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()
