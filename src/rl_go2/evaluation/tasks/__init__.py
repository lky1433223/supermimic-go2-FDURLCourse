"""Registered evaluation task instances."""
from .body_parameters import ComOffsetTask, MassTask
from .combined import CombinedTask
from .low_friction import LowFrictionTask
from .point_push import PointPushTask
from .rich_motion import RichMotionTask
from .rollout import RolloutTask
from .sensor_action import SensorActionTask
from .terrain import StairsTask, TerrainTask

TASKS = {task.TASK: task for task in (
    RichMotionTask(), LowFrictionTask(), RolloutTask(), MassTask(), ComOffsetTask(), PointPushTask(),
    TerrainTask(), StairsTask(), SensorActionTask(), CombinedTask(),
)}
