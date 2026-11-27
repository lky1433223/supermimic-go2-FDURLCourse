"""Load a policy and run real-time, keyboard-driven simulation."""
import argparse
import json
from pathlib import Path
import time

import torch

from rl_go2.config import REPO_ROOT, config_at_iteration, resolve_run_asset
from rl_go2.env import QuadrupedEnv
from rl_go2.evaluation.conditions import NOMINAL_SAMPLE, NOMINAL_SLIDING
from rl_go2.evaluation.runner import load_policy
from rl_go2.state import rotate_inverse
from .conditions import NOMINAL
from .controls import Controls
from .terrain import playback_scene, terrain_order
from .viewer import PlayViewer, create_viewer


def load_inputs(args: argparse.Namespace) -> tuple[dict, torch.nn.Module, Path, int | None]:
    """Load run artifacts and relocate repository assets copied from another host."""
    training = args.run_dir.resolve()
    if (training / 'training').is_dir():
        training /= 'training'
    config = json.loads((training / 'config.json').read_text())
    policy, path, iteration = load_policy(args, training, config, require_iteration=False)
    config = config_at_iteration(config, iteration)
    for entity in config['scene']['entities']:
        if 'file' in entity:
            entity['file'] = resolve_run_asset(entity['file'])
    config['scene'] = playback_scene(config['scene'], config['robot'], args.seed, REPO_ROOT)
    config.update(backend=args.backend, device=args.device, num_envs=1,
                  curriculum={'name': 'none'}, seed=args.seed)
    config['viewer_options'] = {'enable_help_text': False, 'enable_default_keybinds': False}
    config['vis_options'] = {
        'shadow': False,
        'ambient_light': (0.35, 0.35, 0.35),
        'background_color': (0.18, 0.22, 0.27),
        'lights': [
            {'type': 'directional', 'dir': (-1, -1, -2),
             'color': (1.0, 0.97, 0.93), 'intensity': 4.0},
            {'type': 'directional', 'dir': (1, 1, -1),
             'color': (0.88, 0.94, 1.0), 'intensity': 2.5},
        ],
    }
    config = NOMINAL_SAMPLE.condition('nominal', NOMINAL_SLIDING).apply(config)
    # Interactive commands are supplied every step. Keep the task's periodic
    # sampling from replacing the command during a step or normal timeout.
    config['task'].update(command_resample_seconds=1e9, episode_seconds=1e9)
    return config, policy, path, iteration


def select_playback_terrain(env: QuadrupedEnv, viewer: PlayViewer, controls: Controls,
                            order: list[int], cursor: int) -> None:
    """Release any grab, reset onto the selected terrain, and update the view."""
    index = order[cursor]
    patch = env.terrain_patches[index]
    viewer.release_forces()
    env.select_terrain(index)
    viewer.center_camera(tuple(patch['position']))
    controls.publish(terrain=patch['name'], terrain_progress=f'{cursor + 1}/{len(order)}',
                     command=[0.0, 0.0, 0.0], actual=[0.0, 0.0, 0.0])


def run_loop(env: QuadrupedEnv, policy: torch.nn.Module, controls: Controls,
             viewer: PlayViewer, steps: int | None) -> None:
    """Advance at most real-time and smooth body-frame velocity changes over 0.15 s."""
    command = torch.zeros(1, 3, device=env.device)
    order = terrain_order(env.terrain_patches)
    cursor = 0
    condition = NOMINAL
    controls.publish(step_ms=env.dt * 1000)
    select_playback_terrain(env, viewer, controls, order, cursor)
    count = 0
    deadline = time.monotonic()
    with torch.no_grad():
        while steps is None or count < steps:
            if not viewer.refresh():
                break
            target, paused, stopped, reset = controls.sample()
            if stopped:
                break
            requested = controls.consume_condition_request()
            if requested is not None:
                if requested.apply(env, condition):
                    viewer.release_forces()
                    command.zero_()
                    controls.publish(actual=[0.0, 0.0, 0.0])
                condition = requested
                controls.confirm_condition(condition)
            switch = controls.consume_terrain_switch()
            if switch or reset:
                if switch:
                    cursor = (cursor + 1) % len(order)
                    select_playback_terrain(env, viewer, controls, order, cursor)
                else:
                    viewer.release_forces()
                    env.reset()
                command.zero_()
                controls.clear_keys()
                target = [0.0, 0.0, 0.0]
            if paused:
                viewer.release_forces()
                command.zero_()
                env.task.commands.zero_()
                controls.publish(command=[0.0, 0.0, 0.0])
                deadline = time.monotonic()
                time.sleep(0.01)
                continue
            desired = command.new_tensor([target])
            command.lerp_(desired, min(1.0, env.dt / 0.15))
            env.task.commands.copy_(command)
            _, _, done, _ = env.step(policy(env.observe()))
            if bool(done.any()):
                viewer.release_forces()
                controls.clear_keys()
                command.zero_()
                env.task.commands.zero_()
            count += 1
            linear = rotate_inverse(env.state.quaternion, env.state.linear_velocity)
            angular = rotate_inverse(env.state.quaternion, env.state.angular_velocity)
            controls.publish(command=command[0].tolist(),
                             actual=[float(linear[0, 0]), float(linear[0, 1]), float(angular[0, 2])],
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
    args = parser.parse_args(argv)
    if args.backend == 'mujoco' and args.device != 'cpu':
        parser.error('MuJoCo requires --device cpu')
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config, policy, path, iteration = load_inputs(args)
    ranges = config['task']['command_ranges']
    forward, lateral, yaw = (max(abs(value) for value in axis) for axis in ranges)
    controls = Controls((forward, lateral, yaw))
    controls.publish(backend=args.backend, policy=str(path), iteration=iteration,
                     training_ranges=ranges)
    env = QuadrupedEnv(config, viewer=args.backend == 'genesis')
    viewer = None
    try:
        policy = policy.to(env.device)
        viewer = create_viewer(env, controls, args.backend)
        print('Focus the simulation window for W/S, A/D, Q/E and F2 conditions.', flush=True)
        run_loop(env, policy, controls, viewer, args.steps)
    except KeyboardInterrupt:
        pass
    finally:
        controls.key_event('quit', True)
        try:
            if viewer is not None:
                viewer.close()
        finally:
            env.close()
