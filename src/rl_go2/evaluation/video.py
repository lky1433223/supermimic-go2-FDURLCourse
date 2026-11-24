"""Record MuJoCo evaluation transitions as an annotated MP4."""

from contextlib import ExitStack
from copy import copy
import math
from pathlib import Path

import numpy as np

from rl_go2.config import FOOT_NAMES
from rl_go2.mujoco_visuals import brighten_scene


# MuJoCo measures azimuth along the viewing direction, toward the robot.
VIDEO_VIEW_AZIMUTH = {"default": 115, "front": 180, "rear": 0, "left": -90, "right": 90}


class EvaluationVideo:
    """One video across trials, including the last pose before an automatic reset."""

    fps = 20
    codec = "libx264"
    crf = 24
    max_bitrate_bps = 2_000_000
    rate_buffer_bits = 4_000_000

    def __init__(self, path, *, views=("default",)):
        self.path = Path(path)
        self.views = tuple(views)
        self.width, self.height = (640, 480) if len(self.views) == 1 else (448, 336)
        self.columns = min(2, len(self.views)) if len(self.views) <= 4 else 3
        self.rows = math.ceil(len(self.views) / self.columns)

    @property
    def encoding(self):
        return {
            "codec": self.codec,
            "width": self.columns * self.width,
            "height": self.rows * self.height,
            "fps": self.fps,
            "crf": self.crf,
            "max_bitrate_bps": self.max_bitrate_bps,
            "rate_buffer_bits": self.rate_buffer_bits,
        }

    def __enter__(self):
        import cv2
        import imageio.v2 as imageio
        import mujoco

        self.cv2 = cv2
        self.mujoco = mujoco
        self.renderer_resources = ExitStack()
        self.source_model = None
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 1.4
        self.camera.elevation = -14
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(self.path), fps=self.fps, codec=self.codec, quality=None,
            output_params=["-crf", str(self.crf), "-maxrate", str(self.max_bitrate_bps),
                           "-bufsize", str(self.rate_buffer_bits)],
        )
        return self

    def __exit__(self, *exc):
        try:
            self.renderer_resources.close()
        finally:
            self.writer.close()

    def _draw_ground_grid(self):
        if self.ground_id is None:
            return
        if not np.allclose(self.data.geom_xmat[self.ground_id, 6:], (0, 0, 1)):
            return

        # Snap the visible patch to the world grid as the camera follows the robot.
        spacing = 0.5
        origin = np.floor(self.camera.lookat[:2] / spacing) * spacing
        xs, ys = origin[:, None] + spacing * np.arange(-10, 11)
        height = self.data.geom_xpos[self.ground_id, 2] + 0.001
        scene = self.renderer.scene
        line_type = self.mujoco.mjtGeom.mjGEOM_LINE
        size = np.zeros(3)
        rotation = np.eye(3).ravel()
        color = np.array((0.36, 0.40, 0.43, 1.0), dtype=np.float32)
        for x, y in zip(xs, ys):
            for start, end in (((x, ys[0], height), (x, ys[-1], height)),
                               ((xs[0], y, height), (xs[-1], y, height))):
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                start, end = np.array(start), np.array(end)
                self.mujoco.mjv_initGeom(geom, line_type, size, start, rotation, color)
                self.mujoco.mjv_connector(geom, line_type, 1.0, start, end)
                scene.ngeom += 1

    def _render_view(self, view, heading, lines):
        self.camera.azimuth = VIDEO_VIEW_AZIMUTH[view]
        if view != "default":
            self.camera.azimuth += heading
        self.renderer.update_scene(self.data, camera=self.camera)
        self._draw_ground_grid()
        image = self.renderer.render()

        self.cv2.rectangle(image, (0, 0), (self.width, 20 * len(lines) + 8), (22, 22, 22), -1)
        for index, line in enumerate(lines):
            self.cv2.putText(image, line, (10, 20 * (index + 1)), self.cv2.FONT_HERSHEY_SIMPLEX,
                             0.48, (245, 245, 245), 1, self.cv2.LINE_AA)
        return image

    def _render_frame(self, heading, lines):
        if len(self.views) == 1:
            return self._render_view(self.views[0], heading, lines)

        frame = np.zeros((self.rows * self.height, self.columns * self.width, 3), dtype=np.uint8)
        for index, view in enumerate(self.views):
            tile = self._render_view(view, heading, [f"{view.title()} | {lines[0]}", *lines[1:]])
            row, column = divmod(index, self.columns)
            y, x = row * self.height, column * self.width
            frame[y:y + self.height, x:x + self.width] = tile
        return frame

    def frame(self, env, transition, *, step, label, done=False):
        """Use a one-based control step within the trial for timing and frame sampling."""
        elapsed = step * env.dt
        current_frame = math.floor(elapsed * self.fps + 1e-9)
        previous_frame = math.floor((step - 1) * env.dt * self.fps + 1e-9)
        if current_frame == previous_frame and not done:
            return

        backend = env.backend
        if backend.model is not self.source_model:
            self.renderer_resources.close()
            # Keep video lighting and appearance on a separate model.
            self.model = copy(backend.model)
            brighten_scene(self.model)
            planes = self.model.geom_type == self.mujoco.mjtGeom.mjGEOM_PLANE
            self.ground_id = next(iter(np.flatnonzero(planes)), None)
            self.renderer = self.renderer_resources.enter_context(
                self.mujoco.Renderer(self.model, height=self.height, width=self.width)
            )
            self.source_model = backend.model
            self.data = self.mujoco.MjData(self.model)

        state = transition.next_state
        self.data.qpos[:3] = state.position[0].detach().cpu().numpy()
        self.data.qpos[3:7] = state.quaternion[0].detach().cpu().numpy()
        self.data.qpos[backend.q_indices] = state.joint_position[0].detach().cpu().numpy()
        self.mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [*state.position[0, :2].tolist(), 0.2]
        forward_x, forward_y = self.data.xmat[backend.base].reshape(3, 3)[:2, 0]
        heading = math.degrees(math.atan2(forward_y, forward_x))

        lines = [f"{label} | {elapsed:.2f} s"]
        command = transition.context.get("commands")
        if command is not None:
            linear, angular, _ = env.task.body_state(state)
            vx, vy, yaw = command[0].tolist()
            lines.append(f"Target  vx {vx:+.2f}  vy {vy:+.2f}  yaw {yaw:+.2f}")
            lines.append(
                f"Actual  vx {linear[0, 0]:+.2f}  vy {linear[0, 1]:+.2f}  yaw {angular[0, 2]:+.2f}"
            )
        friction = transition.context.get("foot_ground_friction")
        if friction is not None:
            sliding = friction.reshape(len(FOOT_NAMES), 3)[:, 0].tolist()
            lines.append("Sliding friction  " + "  ".join(f"{foot} {value:.2f}" for foot, value in zip(FOOT_NAMES, sliding)))
        fell = transition.terminated.item()
        if fell:
            lines.append("FALL")

        image = self._render_frame(heading, lines)
        count = max(1, current_frame - previous_frame)
        if fell:
            count += self.fps // 2
        for _ in range(count):
            self.writer.append_data(image)
