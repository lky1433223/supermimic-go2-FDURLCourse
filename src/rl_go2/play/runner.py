"""Load a policy and run real-time, keyboard-driven simulation."""
import argparse
import json
from pathlib import Path
import time
import webbrowser

import torch

from rl_go2.env import QuadrupedEnv
from rl_go2.evaluation.conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from rl_go2.evaluation.runner import load_policy
from rl_go2.state import rotate_inverse
from .controls import Controls
from .panel import serve_panel


def load_inputs(args: argparse.Namespace) -> tuple[dict, torch.nn.Module, Path]:
    """Load run artifacts and relocate repository assets copied from another host."""
    training = args.run_dir.resolve()
    if (training / 'training').is_dir():
        training /= 'training'
    config = json.loads((training / 'config.json').read_text())
    repo = Path(__file__).resolve().parents[3]
    for entity in config['scene']['entities']:
        if 'file' not in entity:
            continue
        original = Path(entity['file'])
        if original.is_file():
            continue
        parts = original.parts
        if 'assets' not in parts:
            raise FileNotFoundError(f'Cannot relocate scene asset: {original}')
        local = repo.joinpath(*parts[parts.index('assets'):])
        if not local.is_file():
            raise FileNotFoundError(f'Scene asset missing: {original} (local candidate: {local})')
        entity['file'] = str(local)
    config.update(backend=args.backend, device=args.device, num_envs=1,
                  curriculum={'name': 'none'}, seed=args.seed)
    config = NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING).apply(config)
    # Interactive commands are supplied every step. Keep the task's periodic
    # sampling from replacing the command during a step or normal timeout.
    config['task'].update(command_resample_seconds=1e9, episode_seconds=1e9)
    # Playback does not produce iteration-indexed assessment artifacts.
    loading_args = argparse.Namespace(**vars(args))
    if loading_args.iteration is None:
        loading_args.iteration = 0
    policy, path, _ = load_policy(loading_args, training, config)
    return config, policy, path


def refresh_viewer(env: QuadrupedEnv, backend: str) -> bool:
    """Service window events while paused and report whether it is still open."""
    if backend == 'mujoco':
        viewer = env.backend.viewer
        if not viewer.is_running():
            return False
        viewer.sync()
        return viewer.is_running()
    viewer = env.backend.scene.viewer
    if not viewer.is_alive():
        return False
    viewer.update()
    return viewer.is_alive()


def run_loop(env: QuadrupedEnv, policy: torch.nn.Module, controls: Controls,
             backend: str, steps: int | None) -> None:
    """Advance at most real-time and smooth body-frame velocity changes over 0.15 s."""
    command = torch.zeros(1, 3, device=env.device)
    count = 0
    deadline = time.monotonic()
    with torch.no_grad():
        while steps is None or count < steps:
            if not refresh_viewer(env, backend):
                break
            target, paused, stopped, reset = controls.sample()
            if stopped:
                break
            if reset:
                env.reset()
                command.zero_()
                controls.receive([], 'stop')
                target = [0.0, 0.0, 0.0]
            if paused:
                command.zero_()
                env.task.commands.zero_()
                controls.publish(command=[0.0, 0.0])
                deadline = time.monotonic()
                time.sleep(0.01)
                continue
            desired = command.new_tensor([target])
            command.lerp_(desired, min(1.0, env.dt / 0.15))
            env.task.commands.copy_(command)
            _, _, done, _ = env.step(policy(env.observe()))
            if bool(done.any()):
                controls.receive([], 'stop')
                command.zero_()
                env.task.commands.zero_()
            count += 1
            linear = rotate_inverse(env.state.quaternion, env.state.linear_velocity)
            angular = rotate_inverse(env.state.quaternion, env.state.angular_velocity)
            controls.publish(command=[float(command[0, 0]), float(command[0, 2])],
                             actual=[float(linear[0, 0]), float(angular[0, 2])],
                             seconds=count * env.dt)
            # Discard lag instead of accelerating to catch up after a slow frame.
            deadline = max(deadline + env.dt, time.monotonic())
            time.sleep(max(0.0, deadline - time.monotonic()))


def main(argv: list[str] | None = None) -> None:
    """Start an interactive single-robot policy session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--backend', choices=('genesis', 'mujoco'), default='mujoco')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument('--policy', type=Path)
    weights.add_argument('--checkpoint', type=Path)
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, help='Optional finite interactive smoke run')
    parser.add_argument('--no-browser', action='store_true', help='Print panel URL without opening it')
    args = parser.parse_args(argv)
    if args.backend == 'mujoco' and args.device != 'cpu':
        parser.error('MuJoCo requires --device cpu')
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config, policy, path = load_inputs(args)
    ranges = config['task']['command_ranges']
    controls = Controls(max_speed=max(abs(v) for v in ranges[0]),
                        max_yaw=max(abs(v) for v in ranges[2]))
    controls.publish(backend=args.backend, policy=f'{args.run_dir.name} / {path.name}',
                     training_ranges=ranges)
    env = QuadrupedEnv(config, viewer=True)
    try:
        policy = policy.to(env.device)
        with serve_panel(controls) as url:
            print(f'Keyboard control panel: {url}', flush=True)
            print('Focus the browser panel for WASD. Close the 3D window or press Esc to exit.', flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            run_loop(env, policy, controls, args.backend, args.steps)
    except KeyboardInterrupt:
        pass
    finally:
        controls.receive([], 'quit')
        env.close()
