"""Record MuJoCo evaluation transitions as an annotated MP4."""

from contextlib import ExitStack
import math
from pathlib import Path

from rl_go2.config import FOOT_NAMES


class EvaluationVideo:
    """One video across trials, including the last pose before an automatic reset."""

    fps = 25
    width = 640
    height = 480

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        import cv2
        import imageio.v2 as imageio
        import mujoco

        self.cv2 = cv2
        self.mujoco = mujoco
        self.renderer_resources = ExitStack()
        self.model = None
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 1.8
        self.camera.azimuth = 135
        self.camera.elevation = -20
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=7)
        return self

    def __exit__(self, *exc):
        try:
            self.renderer_resources.close()
        finally:
            self.writer.close()

    def frame(self, env, transition, *, step, label, done=False):
        """Use a one-based control step within the trial for timing and frame sampling."""
        elapsed = step * env.dt
        current_frame = math.floor(elapsed * self.fps + 1e-9)
        previous_frame = math.floor((step - 1) * env.dt * self.fps + 1e-9)
        if current_frame == previous_frame and not done:
            return

        backend = env.backend
        if backend.model is not self.model:
            self.renderer_resources.close()
            self.renderer = self.renderer_resources.enter_context(
                self.mujoco.Renderer(backend.model, height=self.height, width=self.width)
            )
            self.model = backend.model
            self.data = self.mujoco.MjData(self.model)

        state = transition.next_state
        self.data.qpos[:3] = state.position[0].detach().cpu().numpy()
        self.data.qpos[3:7] = state.quaternion[0].detach().cpu().numpy()
        self.data.qpos[backend.q_indices] = state.joint_position[0].detach().cpu().numpy()
        self.mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [*state.position[0, :2].tolist(), 0.2]
        self.renderer.update_scene(self.data, camera=self.camera)
        image = self.renderer.render()

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
        self.cv2.rectangle(image, (0, 0), (self.width, 20 * len(lines) + 8), (22, 22, 22), -1)
        for index, line in enumerate(lines):
            self.cv2.putText(image, line, (10, 20 * (index + 1)), self.cv2.FONT_HERSHEY_SIMPLEX,
                             0.48, (245, 245, 245), 1, self.cv2.LINE_AA)

        count = max(1, current_frame - previous_frame)
        if fell:
            count += self.fps // 2
        for _ in range(count):
            self.writer.append_data(image)
