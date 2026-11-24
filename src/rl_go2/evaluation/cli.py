"""Parse the evaluation command and task configuration."""
import argparse
from pathlib import Path

from .runner import run_evaluation
from .tasks import TASKS
from .video import VIDEO_VIEW_AZIMUTH


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a saved Go2 policy.')
    parser.add_argument('--list-tasks', action='store_true', help='List evaluation tasks and exit')
    parser.add_argument('--task', choices=TASKS, help='Evaluation task (default: standard)')
    parser.add_argument('--run-dir', type=Path, help='Training directory (required for evaluations)')
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument('--checkpoint', type=Path, help='SuperMimic model checkpoint')
    weights.add_argument('--policy', type=Path, help='Exported TorchScript policy (default: training/policy.pt)')
    parser.add_argument('--iteration', type=int, help='Training iteration of the evaluated weights')
    parser.add_argument('--backend', choices=('mujoco', 'genesis'), default='mujoco', help='Default: mujoco')
    parser.add_argument('--seeds', type=int, nargs='+', help='Default: training seed for rollout, 1 for other tasks')
    parser.add_argument('--steps', type=int, help='Rollout steps per seed (default: 1000)')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu', help='Default: cpu')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of simulation environments (default: 1)')
    parser.add_argument('--viewer', action='store_true', help='Open a simulation viewer')
    parser.add_argument('--video', action='store_true', help='Record replay.mp4 with MuJoCo')
    parser.add_argument('--video-view', choices=VIDEO_VIEW_AZIMUTH, nargs='+',
                        help='One or more camera views for --video, in display order (default: default)')
    parser.add_argument('--output', type=Path, help='New directory for this evaluation')
    args = parser.parse_args()
    if args.list_tasks:
        for name, task in TASKS.items():
            print(f'{name}: {task.TASK_DESCRIPTION}')
        return
    if args.run_dir is None:
        parser.error('--run-dir is required for evaluations')

    task = TASKS[args.task or 'standard']
    is_rollout = task.supports_parallel
    if args.steps is not None and (not is_rollout or args.steps < 1):
        parser.error('--steps requires rollout and a positive step count')
    if args.num_envs < 1 or (args.iteration is not None and args.iteration < 0):
        parser.error('--num-envs must be positive and --iteration must be nonnegative')
    if not is_rollout and (args.device != 'cpu' or args.num_envs != 1):
        parser.error('Only rollout supports device and environment count overrides')
    if args.video_view is not None and not args.video:
        parser.error('--video-view requires --video')
    if args.video_view and len(set(args.video_view)) != len(args.video_view):
        parser.error('--video-view values must be distinct')
    if args.backend not in task.backends:
        parser.error(f'{task.TASK} supports backends: {", ".join(task.backends)}')
    if args.video and args.backend != 'mujoco':
        parser.error('Video requires MuJoCo')
    if args.backend == 'mujoco' and (args.device != 'cpu' or args.num_envs != 1):
        parser.error('MuJoCo requires CPU and one environment')
    if is_rollout and args.steps is None:
        args.steps = task.default_steps

    try:
        run_evaluation(args, task)
    except (OSError, ValueError) as error:
        parser.error(str(error))
