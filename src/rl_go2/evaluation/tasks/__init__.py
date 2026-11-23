"""Registered evaluation tasks."""
from . import friction_switch, low_friction, rollout, standard

TASKS = {task.TASK: task for task in (standard, low_friction, friction_switch, rollout)}
