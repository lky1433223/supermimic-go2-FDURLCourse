"""Native playback window contract and backend selection."""
from typing import Protocol

from rl_go2.env import QuadrupedEnv
from .controls import Controls


class PlayViewer(Protocol):
    """A window that owns native input and displays the shared HUD."""

    def refresh(self) -> bool:
        """Pump events, draw, and return whether the window remains open."""
        ...

    def close(self) -> None:
        """Release window-owned resources before the simulation closes."""
        ...


def create_viewer(env: QuadrupedEnv, controls: Controls, backend: str) -> PlayViewer:
    """Attach the explicit adapter for the selected simulation backend."""
    if backend == 'mujoco':
        from .mujoco_viewer import MujocoPlayViewer
        return MujocoPlayViewer(env.backend.model, env.backend.data, controls)
    if backend == 'genesis':
        from .genesis_viewer import GenesisPlayViewer
        return GenesisPlayViewer(env.backend.scene.viewer, controls)
    raise ValueError(f'Unsupported playback backend: {backend}')
