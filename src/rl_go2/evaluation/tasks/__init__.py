"""Registered evaluation task instances."""
from .body_parameters import ComOffsetTask, MassTask
from .low_friction import LowFrictionTask
from .point_push import PointPushTask
from .rich_motion import RichMotionTask
from .rollout import RolloutTask
from .sensor_action import SensorActionTask
from .terrain import TerrainTask

TASKS = {task.TASK: task for task in (
    RichMotionTask(), LowFrictionTask(), RolloutTask(), MassTask(), ComOffsetTask(), PointPushTask(),
    TerrainTask(), SensorActionTask(),
)}
