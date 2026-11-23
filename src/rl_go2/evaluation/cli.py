"""Parse the evaluation command and task configuration."""
import argparse
import json
from pathlib import Path

from .runner import run_evaluation
from .tasks import TASKS, rollout, standard


def resolve_task(task_name, path):
    document = json.loads(path.read_text()) if path is not None else {}
    if not isinstance(document, dict) or document.keys() - {'task', 'condition', 'parameters'}:
        raise ValueError('Task JSON must be an object with task, condition and parameters fields')
    if path is not None:
        configured = document.get('task')
        if not isinstance(configured, str) or configured not in TASKS:
            raise ValueError('Task JSON must name a registered task')
        if task_name is not None and task_name != configured:
            raise ValueError('--task must match the task in --config')
        task_name = configured

    condition = document.get('condition', 'nominal')
    if (not isinstance(condition, str) or not condition.strip() or condition in ('.', '..')
            or Path(condition).name != condition or '/' in condition or '\\' in condition or '\0' in condition):
        raise ValueError('condition must be a single directory name')
    parameters = document.get('parameters', {})
    if not isinstance(parameters, dict):
        raise ValueError('parameters must be a JSON object')
    task = TASKS[task_name or standard.TASK]
    validator = getattr(task, 'validate_parameters', None)
    if validator is not None:
        validator(parameters)
    elif parameters:
        raise ValueError(f'{task.TASK} does not accept JSON parameters')
    return task, condition, parameters


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a saved Go2 policy.')
    parser.add_argument('--list-tasks', action='store_true', help='List evaluation tasks and exit')
    parser.add_argument('--task', choices=TASKS, help='Evaluation task (default: standard)')
    parser.add_argument('--config', type=Path, help='Task JSON with task, condition and parameters')
    parser.add_argument('--run-dir', type=Path, help='Training directory (required for evaluations)')
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument('--checkpoint', type=Path, help='SuperMimic model checkpoint')
    weights.add_argument('--policy', type=Path, help='Exported TorchScript policy (default: training/policy.pt)')
    parser.add_argument('--iteration', type=int, help='Training iteration of the evaluated weights')
    parser.add_argument('--backend', choices=('mujoco', 'genesis'), default='mujoco', help='Default: mujoco')
    parser.add_argument('--seeds', type=int, nargs='+', help='Default: training seed for rollout, 1 for other tasks')
    parser.add_argument('--steps', type=int, help=f'Rollout steps per seed (default: {rollout.DEFAULT_STEPS})')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu', help='Default: cpu')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of simulation environments (default: 1)')
    parser.add_argument('--viewer', action='store_true', help='Open a simulation viewer')
    parser.add_argument('--video', action='store_true', help='Record replay.mp4 with MuJoCo')
    parser.add_argument('--output', type=Path, help='New directory for this evaluation')
    args = parser.parse_args()
    if args.list_tasks:
        for name, task in TASKS.items():
            print(f'{name}: {task.TASK_DESCRIPTION}')
        return
    if args.run_dir is None:
        parser.error('--run-dir is required for evaluations')

    try:
        task, condition, parameters = resolve_task(args.task, args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    is_rollout = task is rollout
    if args.steps is not None and (not is_rollout or args.steps < 1):
        parser.error('--steps requires rollout and a positive step count')
    if args.num_envs < 1 or (args.iteration is not None and args.iteration < 0):
        parser.error('--num-envs must be positive and --iteration must be nonnegative')
    if not is_rollout and (args.device != 'cpu' or args.num_envs != 1):
        parser.error('Only rollout supports device and environment count overrides')
    if (args.video or task.TASK == 'friction-switch') and args.backend != 'mujoco':
        parser.error('Video and friction-switch require MuJoCo')
    if args.backend == 'mujoco' and (args.device != 'cpu' or args.num_envs != 1):
        parser.error('MuJoCo requires CPU and one environment')
    if is_rollout and args.steps is None:
        args.steps = rollout.DEFAULT_STEPS

    try:
        run_evaluation(args, task, condition, parameters)
    except (OSError, ValueError) as error:
        parser.error(str(error))
