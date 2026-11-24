"""Registered evaluation task instances."""
from .friction_switch import FrictionSwitchTask
from .low_friction import LowFrictionTask
from .rollout import RolloutTask
from .standard import StandardTask

TASKS = {task.TASK: task for task in (
    StandardTask(), LowFrictionTask(), FrictionSwitchTask(), RolloutTask(),
)}
