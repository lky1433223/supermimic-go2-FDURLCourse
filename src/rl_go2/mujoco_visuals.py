"""Shared visual settings for MuJoCo windows and recordings."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco


def brighten_scene(model: 'mujoco.MjModel') -> None:
    """Set headlight and ground colors in the supplied model."""
    import mujoco

    headlight = model.vis.headlight
    headlight.active = 1
    headlight.ambient[:] = (0.6, 0.6, 0.6)
    headlight.diffuse[:] = (0.7, 0.7, 0.7)
    planes = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    model.geom_rgba[planes] = (0.75, 0.78, 0.8, 1.0)
