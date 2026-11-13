"""Reproducible real-simulator speed tracking evaluation, independent of training loss."""
import argparse
from copy import deepcopy
import json
import math
import sys
from pathlib import Path

import torch

from rl_go2.env import QuadrupedEnv
from rl_go2.trainers import output_directory
from rl_go2.trainers.supermimic import Actor


CASE_SECONDS = 20
TRANSIENT_SECONDS = 2
STOP_AT = 8
STOP_SPEED = 0.1
STOP_YAW = 0.15
STOP_HOLD = 0.5


def standard_cases():
    for speed in (0.2, 0.4, 0.6, 0.8):
        for yaw in (0, 0.4, -0.4):
            yield 'standard', f'vx{speed}_yaw{yaw}', speed, yaw, False
    for yaw in (0, 0.1, -0.1, 0.3, -0.3, 0.5, -0.5):
        yield 'standard', f'vx0.5_yaw{yaw}', 0.5, yaw, False
    yield 'ood', 'stand', 0, 0, False
    yield 'ood', 'walk_to_stop', 0.5, 0, True
    for speed in (1.0, 1.2):
        yield 'ood', f'vx{speed}_yaw0', speed, 0, False
    for yaw in (0.6, -0.6, 0.8, -0.8):
        yield 'ood', f'vx0.5_yaw{yaw}', 0.5, yaw, False


def stop_metrics(velocities, positions, switch_step, dt, completed):
    hold_steps = math.ceil(STOP_HOLD / dt)
    tail_steps = round(TRANSIENT_SECONDS / dt)
    hold = 0
    settled = None
    for index, velocity in enumerate(velocities[switch_step:]):
        slow = velocity[:2].norm().item() < STOP_SPEED and abs(velocity[2].item()) < STOP_YAW
        hold = hold + 1 if slow else 0
        if hold >= hold_steps:
            settled = (index + 1) * dt
            break
    return {
        'time_to_settle_seconds': settled,
        'final_2s_displacement_m': (
            (positions[-1] - positions[-1 - tail_steps]).norm().item()
            if completed and tail_steps > 0 and len(positions) > tail_steps else None
        ),
    }


def evaluate_standard(config, policy, seeds, backend='mujoco'):
    config = deepcopy(config)
    config.update(backend=backend, device='cpu', num_envs=1, curriculum={'name': 'none'})
    config['task'].update(episode_seconds=CASE_SECONDS, command_resample_seconds=CASE_SECONDS)
    results = []
    with torch.inference_mode():
        for seed in seeds:
            config['seed'] = seed
            torch.manual_seed(seed)
            env = QuadrupedEnv(config, record_transitions=True)
            try:
                case_steps = env.max_episode_length
                transient_steps = round(TRANSIENT_SECONDS / env.dt)
                switch_step = round(STOP_AT / env.dt)
                for group, name, speed, yaw, stop in standard_cases():
                    env.reset()
                    command = env.task.commands.new_tensor([speed, 0, yaw])
                    segments = [{'start_seconds': 0, 'command': [speed, 0, yaw]}]
                    if stop:
                        segments.append({'start_seconds': switch_step * env.dt, 'command': [0, 0, 0]})
                    env.task.commands.copy_(command)
                    errors, velocities = [], []
                    positions = [env.state.position[0, :2].clone()]
                    fell = False
                    for step in range(case_steps):
                        if stop and step == switch_step:
                            env.task.commands.zero_()
                        _, _, _, extras = env.step(policy(env.observe()))
                        tr = extras['transition']
                        linear, angular, _ = env.task.body_state(tr.next_state)
                        velocity = torch.cat((linear[0, :2], angular[0, 2:3]))
                        if not torch.isfinite(velocity).all():
                            raise RuntimeError(f'Non-finite velocity in {name} at step {step + 1}')
                        errors.append((velocity - tr.context['commands'][0]).square())
                        velocities.append(velocity)
                        positions.append(tr.next_state.position[0, :2])
                        fell = tr.terminated.item()
                        if fell or tr.truncated.item():
                            break
                    squared_error = torch.stack(errors)
                    elapsed = len(errors) * env.dt
                    result = {
                        'seed': seed, 'group': group, 'case': name, 'segments': segments,
                        'steps': len(errors), 'seconds': elapsed,
                        'fell': fell, 'fall_time_seconds': elapsed if fell else None,
                        'velocity_rmse': squared_error.mean(0).sqrt().tolist(),
                        'velocity_rmse_after_2s': (
                            squared_error[transient_steps:].mean(0).sqrt().tolist()
                            if len(errors) > transient_steps else None
                        ),
                    }
                    if stop:
                        result.update(stop_metrics(velocities, positions, switch_step, env.dt, not fell))
                    if env.foot_friction is not None:
                        result['foot_ground_friction'] = env.foot_friction[0].reshape(4, 3).tolist()
                    if env.base_mass_kg is not None:
                        result['base_mass_kg'] = env.base_mass_kg.item()
                    if env.base_com_offset_m is not None:
                        result['base_com_offset_m'] = env.base_com_offset_m[0].tolist()
                    results.append(result)
            finally:
                env.close()
    return {
        'protocol': 'standard', 'backend': backend, 'seeds': list(seeds),
        'velocity_axes': ['vx', 'vy', 'yaw_rate'], 'case_seconds': CASE_SECONDS,
        'stop_criteria': {'planar_speed_m_s': STOP_SPEED, 'abs_yaw_rad_s': STOP_YAW,
                          'hold_seconds': STOP_HOLD},
        'groups': {
            group: {'trials': sum(row['group'] == group for row in results),
                    'falls': sum(row['group'] == group and row['fell'] for row in results)}
            for group in ('standard', 'ood')
        },
        'per_case': results,
    }


def evaluate(config, policy, seeds, steps, backend='mujoco'):
    config = deepcopy(config)
    config.update(backend=backend, device='cpu', num_envs=1, curriculum={'name': 'none'})
    results = []
    with torch.inference_mode():
        for seed in seeds:
            config['seed'] = seed
            torch.manual_seed(seed)
            env = QuadrupedEnv(config, record_transitions=True)
            errors, commands, velocities, actions, rewards = [], [], [], [], []
            falls = timeouts = 0
            try:
                for _ in range(steps):
                    observation, reward, _, extras = env.step(policy(env.observe()))
                    tr = extras['transition']
                    linear, angular, _ = env.task.body_state(tr.next_state)
                    velocity = torch.cat((linear[:, :2], angular[:, 2:3]), -1)
                    if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
                        raise RuntimeError('Non-finite evaluation output')
                    errors.append((velocity - tr.context['commands']).square())
                    commands.append(tr.context['commands'])
                    velocities.append(velocity)
                    actions.append(tr.action)
                    rewards.append(reward)
                    falls += int(tr.terminated.sum())
                    timeouts += int(tr.truncated.sum())
                results.append({'seed': seed, 'steps': steps, 'seconds': steps * env.dt,
                                'falls': falls, 'timeouts': timeouts,
                                'squared_error_sum': torch.cat(errors).sum(0).tolist(),
                                'velocity_rmse': torch.cat(errors).mean(0).sqrt().tolist(),
                                'mean_command': torch.cat(commands).mean(0).tolist(),
                                'mean_velocity': torch.cat(velocities).mean(0).tolist(),
                                'mean_reward': torch.cat(rewards).mean().item(),
                                'action_std': torch.cat(actions).std(0, unbiased=False).mean().item()})
                if env.foot_friction is not None:
                    results[-1]['foot_ground_friction'] = env.foot_friction[0].reshape(4, 3).tolist()
                if env.base_mass_kg is not None:
                    results[-1]['base_mass_kg'] = env.base_mass_kg.item()
                if env.base_com_offset_m is not None:
                    results[-1]['base_com_offset_m'] = env.base_com_offset_m[0].tolist()
            finally:
                env.close()
    total = sum(item['steps'] for item in results)
    rmse = (torch.tensor([item['squared_error_sum'] for item in results]).sum(0) / total).sqrt().tolist()
    return {'backend': backend, 'seeds': list(seeds), 'steps': total,
            'seconds': sum(item['seconds'] for item in results),
            'falls': sum(item['falls'] for item in results), 'velocity_rmse': rmse,
            'mean_reward': sum(item['mean_reward'] * item['steps'] for item in results) / total,
            'tracking_passed': sum(item['falls'] for item in results) == 0 and all(x < .15 for x in rmse),
            'acceptance': 'zero falls and vx/vy/yaw RMSE each < 0.15 in SI units', 'per_seed': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--backend', choices=('mujoco', 'genesis'), default='mujoco')
    parser.add_argument('--protocol', choices=('random', 'standard'), default='random')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--steps', type=int, help='Number of steps per seed for the random protocol')
    parser.add_argument('--output', type=Path, help='Defaults to <run-dir>/evaluation/tracking-<timestamp>/assessment.json')
    parser.add_argument('--require-tracking', action='store_true')
    args = parser.parse_args()
    standard = args.protocol == 'standard'
    seeds = args.seeds or ([1] if standard else [1, 2, 3, 4, 5])
    if standard and (args.steps is not None or args.require_tracking):
        parser.error('--steps and --require-tracking apply only to the random protocol')
    steps = args.steps if args.steps is not None else 1000
    if steps < 2:
        parser.error('steps must be >= 2')
    if args.policy and args.checkpoint:
        parser.error('Choose --policy or --checkpoint')
    torch.set_num_threads(2)
    config = json.loads((args.run_dir / 'config.json').read_text())
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        policy = Actor(checkpoint['actor']['norm.mean'].numel(), len(config['robot']['joint_names']), checkpoint['trainer']['actor_hidden'],
                       config['robot']['action_clip'])
        policy.load_state_dict(checkpoint['actor'])
    else:
        policy = torch.jit.load(str(args.policy or args.run_dir / 'policy.pt'), map_location='cpu')
    policy.eval()
    if standard:
        report = evaluate_standard(config, policy, seeds, args.backend)
    else:
        report = evaluate(config, policy, seeds, steps, args.backend)
    if args.output is None:
        name = 'tracking-standard' if standard else 'tracking'
        args.output = output_directory(name, args.run_dir / 'evaluation') / 'assessment.json'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Evaluation output: {args.output.resolve()}', file=sys.stderr, flush=True)
    print(json.dumps({key: value for key, value in report.items() if key not in ('per_seed', 'per_case')}, indent=2))
    if args.require_tracking and not report['tracking_passed']:
        raise SystemExit('Speed tracking acceptance failed; see evaluation report')


if __name__ == '__main__':
    main()
