"""MuJoCo rendering with GLFW press/release callbacks and an in-window HUD."""
import ctypes
from copy import copy
import math
import sys

import glfw
import mujoco
import numpy as np

from rl_go2.backends.mujoco import MujocoBackend
from rl_go2.mujoco_visuals import brighten_scene
from .controls import Controls
from .hud import MARGIN, PANEL, WIDTH, layout_hud

# These groups match the visual and collision classes in the Go2 MJCF.
VISUAL_GEOM_GROUP = 2
COLLISION_GEOM_GROUP = 3


def playback_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Make a render-only copy with the brighter evaluation-video lighting."""
    visual = copy(model)
    brighten_scene(visual)
    visual.vis.headlight.specular[:] = (0.2, 0.2, 0.2)
    return visual


class MujocoPlayViewer:
    """Own a GLFW window and MuJoCo render resources on the calling thread."""

    def __init__(self, backend: MujocoBackend, controls: Controls) -> None:
        if sys.platform == 'darwin' and not ctypes.CDLL(None).pthread_main_np():
            raise RuntimeError('play.py requires plain python on macOS. Run: uv run --extra train python scripts/play.py ...')
        self.backend = backend
        self.model = playback_model(backend.model)
        # Clipping distances scale with model extent, which grows with patch separation.
        self.model.vis.map.znear = min(self.model.vis.map.znear, 0.03 / self.model.stat.extent)
        self.data = backend.data
        self.controls = controls
        self.last_cursor = (0.0, 0.0)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        self.options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
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
            self.hud_font_scale = 100
            self.context = mujoco.MjrContext(self.model, self.hud_font_scale)
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_window_focus_callback(self.window, self._on_focus)
            glfw.set_cursor_pos_callback(self.window, self._on_cursor)
            glfw.set_scroll_callback(self.window, self._on_scroll)
            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        except Exception:
            glfw.destroy_window(self.window)
            glfw.terminate()
            raise
        backend.set_interactive_force_callback(self.apply_forces)

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        if key == glfw.KEY_C and action == glfw.PRESS:
            show_collision = not self.options.geomgroup[COLLISION_GEOM_GROUP]
            self.options.geomgroup[COLLISION_GEOM_GROUP] = show_collision
            self.options.geomgroup[VISUAL_GEOM_GROUP] = not show_collision
            return
        mapping = {glfw.KEY_W: 'w', glfw.KEY_S: 's', glfw.KEY_A: 'a', glfw.KEY_D: 'd',
                   glfw.KEY_Q: 'q', glfw.KEY_E: 'e', glfw.KEY_UP: 'up', glfw.KEY_DOWN: 'down',
                   glfw.KEY_SPACE: 'stop', glfw.KEY_P: 'pause',
                   glfw.KEY_R: 'reset', glfw.KEY_ESCAPE: 'quit', glfw.KEY_F1: 'details',
                   glfw.KEY_N: 'next_terrain'}
        if key in mapping:
            if action == glfw.PRESS and mapping[key] in {'pause', 'reset', 'stop', 'quit'}:
                self.release_forces()
            self.controls.key_event(mapping[key], action != glfw.RELEASE, action == glfw.REPEAT)

    def _on_focus(self, window, focused: int) -> None:
        if not focused:
            self.release_forces()
            self.controls.clear_keys()

    def center_camera(self, position: tuple[float, float, float]) -> None:
        """Center the free camera on the terrain spawn in world meters."""
        self.camera.lookat[:] = position

    def release_forces(self) -> None:
        """Drop the selected body so no perturbation is applied on the next step."""
        self.perturb.active = 0
        self.perturb.select = 0
        self.controls.publish(mouse_body='none', mouse_force_n=0.0)

    def _on_mouse_button(self, window, button: int, action: int, mods: int) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        self.release_forces()
        if action != glfw.PRESS or self.controls.status()['paused']:
            return
        x, y = glfw.get_cursor_pos(window)
        self.last_cursor = (x, y)
        width, height = glfw.get_window_size(window)
        if width <= 0 or height <= 0:
            return
        point = np.zeros(3)
        ids = np.full(3, -1, dtype=np.int32)
        body = mujoco.mjv_select(self.model, self.data, self.options, width / height,
                                 x / width, 1.0 - y / height, self.scene,
                                 point, ids[0:1], ids[1:2], ids[2:3])
        if body <= 0 or self.model.body_mass[body] <= 0:
            return
        self.perturb.select = body
        self.perturb.localpos[:] = self.data.xmat[body].reshape(3, 3).T @ (point - self.data.xpos[body])
        mujoco.mjv_initPerturb(self.model, self.data, self.scene, self.perturb)
        self.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
        self.controls.publish(mouse_body=self.model.body(body).name)

    def apply_forces(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Write native perturbation forces after the backend clears its force buffer."""
        if not self.perturb.active:
            return
        mujoco.mjv_applyPerturbForce(model, data, self.perturb)
        force = float(np.linalg.norm(data.xfrc_applied[self.perturb.select, :3]))
        self.controls.publish(mouse_force_n=force)

    def _on_cursor(self, window, x: float, y: float) -> None:
        dx, dy = x - self.last_cursor[0], y - self.last_cursor[1]
        self.last_cursor = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not left and not right:
            return
        height = max(glfw.get_window_size(window)[1], 1)
        if self.perturb.active:
            mujoco.mjv_movePerturb(self.model, self.data, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                                  dx / height, dy / height, self.scene, self.perturb)
            return
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
            self.release_forces()
            self.controls.clear_keys()
            return False
        width, height = glfw.get_framebuffer_size(self.window)
        if width == 0 or height == 0:
            return True
        glfw.make_context_current(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.options, self.perturb, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        self._draw_hud(width, height)
        glfw.swap_buffers(self.window)
        return True

    def _draw_hud(self, width: int, height: int) -> None:
        panel = layout_hud(self.controls.status())
        logical_width, _ = glfw.get_window_size(self.window)
        scale = width / max(1, logical_width)
        left = round(MARGIN * scale)
        top = height - left
        panel_width = round(WIDTH * scale)
        panel_height = round(panel.height * scale)
        background = mujoco.MjrRect(left, top - panel_height, panel_width, panel_height)
        mujoco.mjr_rectangle(background, *(channel / 255 for channel in PANEL))
        for offset in panel.dividers:
            divider = mujoco.MjrRect(left + round(16 * scale), top - round(offset * scale),
                                     panel_width - round(32 * scale), max(1, round(scale)))
            mujoco.mjr_rectangle(divider, 0.25, 0.32, 0.40, 0.8)

        # Each rectangle sets the GL viewport. Text coordinates use the full framebuffer.
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0, 0, 0, 0)
        font_scale = min(range(50, 301, 50), key=lambda size: abs(size - 85 * scale))
        if font_scale != self.hud_font_scale:
            mujoco.mjr_changeFont(font_scale, self.context)
            self.hud_font_scale = font_scale
        for label in panel.labels:
            font = mujoco.mjtFont.mjFONT_BIG if label.large else mujoco.mjtFont.mjFONT_NORMAL
            mujoco.mjr_text(font, label.value, self.context,
                            (left + label.x * scale) / width, (top - label.y * scale) / height,
                            *(channel / 255 for channel in label.color))

    def close(self) -> None:
        """Free GL resources while their window context is still current."""
        self.release_forces()
        self.backend.set_interactive_force_callback(None)
        glfw.make_context_current(self.window)
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()
