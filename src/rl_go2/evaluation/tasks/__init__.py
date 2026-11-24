"""Registered evaluation task instances."""
from .body_parameters import ComOffsetTask, MassTask
from .friction_switch import FrictionSwitchTask
from .low_friction import LowFrictionTask
from .point_push import PointPushTask
from .rollout import RolloutTask
from .standard import StandardTask
from .terrain import TerrainTask

TASKS = {task.TASK: task for task in (
    StandardTask(), LowFrictionTask(), FrictionSwitchTask(), RolloutTask(), MassTask(), ComOffsetTask(), PointPushTask(),
    TerrainTask(),
)}
