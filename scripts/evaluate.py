"""Evaluate a saved Go2 policy and optionally view or record the same rollout."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from hashlib import file_digest
import json
import math
import socket
import sys
from pathlib import Path

import torch

from rl_go2.env import QuadrupedEnv
from rl_go2.config import FOOT_NAMES
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


def evaluate_standard(config, policy, seeds, backend='mujoco', *, viewer=False, video=None):
    config = deepcopy(config)
    config.update(backend=backend, device='cpu', num_envs=1, curriculum={'name': 'none'})
    config['task'].update(episode_seconds=CASE_SECONDS, command_resample_seconds=CASE_SECONDS)
    results = []
    with torch.inference_mode():
        for seed in seeds:
            config['seed'] = seed
            torch.manual_seed(seed)
            env = QuadrupedEnv(config, viewer=viewer, record_transitions=True)
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
                        done = fell or tr.truncated.item()
                        if video is not None:
                            video.frame(env, tr, step=step + 1, label=f'Seed {seed} | {name}', done=done)
                        if done:
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
                        result['foot_ground_friction'] = tr.context['foot_ground_friction'][0].reshape(4, 3).tolist()
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


def evaluate_rollout(config, policy, seeds, steps, *, viewer=False, video=None):
    results = []
    with torch.inference_mode():
        for seed in seeds:
            config['seed'] = seed
            torch.manual_seed(seed)
            env = QuadrupedEnv(config, viewer=viewer, record_transitions=video is not None)
            total_reward = 0.0
            resets = 0
            try:
                for step in range(1, steps + 1):
                    observation, reward, done, extras = env.step(policy(env.observe()))
                    if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
                        raise RuntimeError('Non-finite simulation output')
                    total_reward += reward.mean().item()
                    resets += int(done.sum())
                    if video is not None:
                        video.frame(env, extras['transition'], step=step, label=f'Seed {seed} | rollout',
                                    done=bool(done[0]) or step == steps)
                results.append({'seed': seed, 'steps': steps, 'seconds': steps * env.dt,
                                'observation_shape': list(observation.shape),
                                'mean_total_reward': total_reward, 'episode_resets': resets,
                                'policy_updates': 0})
            finally:
                env.close()
    return {'protocol': 'rollout', 'backend': config['backend'], 'device': config['device'],
            'num_envs': config['num_envs'], 'seeds': list(seeds), 'per_case': results}


def friction_switch_config(config):
    config['task'].update(episode_seconds=16, command_resample_seconds=16)
    config['domain_randomization'] = {'foot_ground_friction': {
        'enabled': True, 'env_multiplier_range': [1.0, 1.0],
        'base_ranges': {foot: {'sliding': [0.8, 0.8], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]}
                        for foot in FOOT_NAMES},
    }}


def evaluate_friction_switch(config, policy, seeds, *, viewer=False, video=None):
    import mujoco

    results = []
    with torch.inference_mode():
        for seed in seeds:
            config['seed'] = seed
            for case, ratios in (('fixed', (1., 1., 1.)), ('switch', (1., 0.5, 1.))):
                torch.manual_seed(seed)
                env = QuadrupedEnv(config, viewer=viewer, record_transitions=True)
                try:
                    backend = env.backend
                    ground = next(i for i, kind in enumerate(backend.model.geom_type)
                                  if kind == mujoco.mjtGeom.mjGEOM_PLANE)
                    feet = set(backend.foot_ids)
                    command = env.task.commands.new_tensor([0.5, 0., 0.])
                    env.task.commands.copy_(command)
                    segment_steps = round(5 / env.dt)
                    segments = []
                    fell = False
                    step = 0
                    for ratio in ratios:
                        backend.set_foot_friction_multipliers(command.new_tensor([ratio]), torch.arange(1))
                        env.foot_friction.copy_(
                            torch.from_numpy(backend.model.geom_friction[backend.foot_ids]).reshape(1, 12))
                        errors, contact_sliding = [], set()
                        for _ in range(segment_steps):
                            _, _, _, extras = env.step(policy(env.observe()))
                            tr = extras['transition']
                            linear, angular, _ = env.task.body_state(tr.next_state)
                            velocity = torch.cat((linear[0, :2], angular[0, 2:3]))
                            errors.append((velocity - command).square())
                            for contact in backend.data.contact:
                                if (contact.efc_address >= 0 and ground in contact.geom
                                        and any(geom in feet for geom in contact.geom)):
                                    contact_sliding.add(round(float(contact.friction[0]), 6))
                            fell = tr.terminated.item()
                            step += 1
                            if video is not None:
                                video.frame(env, tr, step=step, label=f'Seed {seed} | {case} | friction {0.8 * ratio:.2f}',
                                            done=fell or step == 3 * segment_steps)
                            if fell:
                                break
                        segments.append({'seconds': len(errors) * env.dt, 'sliding': 0.8 * ratio,
                                         'contact_sliding': sorted(contact_sliding),
                                         'contact_matches': contact_sliding == {0.8 * ratio},
                                         'velocity_rmse': torch.stack(errors).mean(0).sqrt().tolist()})
                        if fell:
                            break
                    seconds = sum(segment['seconds'] for segment in segments)
                    results.append({'seed': seed, 'case': case, 'fell': fell, 'seconds': seconds,
                                    'fall_time_seconds': seconds if fell else None, 'segments': segments})
                finally:
                    env.close()
    return {'protocol': 'friction-switch', 'backend': 'mujoco', 'seeds': list(seeds),
            'command': [0.5, 0, 0], 'segment_seconds': 5, 'per_case': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Resolved evaluation config.json')
    parser.add_argument('--condition', default='nominal', help='Name for the condition output directory')
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument('--checkpoint', type=Path, help='SuperMimic model checkpoint')
    weights.add_argument('--policy', type=Path, help='Exported TorchScript policy')
    parser.add_argument('--iteration', type=int, help='Checkpoint iteration for an exported policy')
    parser.add_argument('--backend', choices=('mujoco', 'genesis'), default='mujoco')
    parser.add_argument('--protocol', choices=('standard', 'friction-switch', 'rollout'), default='standard')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--steps', type=int, help='Number of steps per seed for rollout (default: 1000)')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--output', type=Path, help='New directory for this evaluation')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true', help='Record replay.mp4 with MuJoCo')
    args = parser.parse_args()
    if args.steps is not None and (args.protocol != 'rollout' or args.steps < 1):
        parser.error('--steps requires rollout and a positive step count')
    if not args.condition or Path(args.condition).name != args.condition or args.condition in ('.', '..'):
        parser.error('--condition must be a single directory name')
    if args.protocol != 'rollout' and (args.device != 'cpu' or args.num_envs != 1):
        parser.error('Only rollout supports device and environment count overrides')
    if args.num_envs < 1 or (args.iteration is not None and args.iteration < 0):
        parser.error('--num-envs must be positive and --iteration must be nonnegative')
    if (args.video or args.protocol == 'friction-switch') and args.backend != 'mujoco':
        parser.error('Video and friction-switch require MuJoCo')
    if args.backend == 'mujoco' and (args.device != 'cpu' or args.num_envs != 1):
        parser.error('MuJoCo requires CPU and one environment')

    run = args.run_dir.resolve()
    training = run / 'training' if (run / 'training').is_dir() else run
    output_root = run.parent if run.name == 'training' else run
    config_path = args.config or training / 'config.json'
    config = json.loads(config_path.read_text())
    config.update(backend=args.backend, device=args.device, num_envs=args.num_envs, curriculum={'name': 'none'})
    if args.protocol == 'standard':
        config['task'].update(episode_seconds=CASE_SECONDS, command_resample_seconds=CASE_SECONDS)
    elif args.protocol == 'friction-switch':
        friction_switch_config(config)

    torch.set_num_threads(2)
    weight_path = args.checkpoint or args.policy or training / 'policy.pt'
    iteration = args.iteration
    if args.checkpoint:
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
        policy = Actor(checkpoint['actor']['norm.mean'].numel(), len(config['robot']['joint_names']),
                       checkpoint['trainer']['actor_hidden'], config['robot']['action_clip'])
        policy.load_state_dict(checkpoint['actor'])
        if iteration is None:
            iteration = checkpoint.get('iteration')
    else:
        policy = torch.jit.load(str(weight_path), map_location='cpu')
    if iteration is None and args.policy is None:
        candidates = [weight_path] if args.checkpoint else list(training.glob('model_*.pt'))
        iterations = [int(p.stem[6:]) for p in candidates if p.stem.startswith('model_') and p.stem[6:].isdigit()]
        iteration = max(iterations, default=None)
    if args.output is None and iteration is None:
        parser.error('Specify --iteration when the checkpoint iteration cannot be inferred')
    policy = policy.to(args.device).eval()
    seeds = args.seeds or ([config['seed']] if args.protocol == 'rollout' else [1])
    host = socket.gethostname()
    machine = 'icc' if str(output_root).startswith('/home/lky/RL-Go2/') else host.split('.')[0]
    dest = args.output or output_directory(
        f'{args.protocol}-{args.backend}-{machine}', output_root / 'evaluation' / f'iteration-{iteration}')
    dest.mkdir(parents=True, exist_ok=False)
    condition = dest / args.condition
    condition.mkdir()
    with weight_path.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    metadata = {'protocol': args.protocol, 'backend': args.backend, 'machine': machine, 'hostname': host,
                'checkpoint_iteration': iteration, 'weights': str(weight_path.resolve()), 'weights_sha256': digest,
                'seeds': seeds, 'config_source': str(config_path.resolve()), 'config': config}
    if args.protocol == 'rollout':
        metadata['steps'] = args.steps or 1000
    if args.video:
        metadata['video'] = f'{args.condition}/replay.mp4'
        from rl_go2.evaluation_video import EvaluationVideo
        recording = EvaluationVideo(condition / 'replay.mp4')
    else:
        recording = nullcontext(None)
    (dest / 'eval-config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Evaluation output: {dest.resolve()}', file=sys.stderr, flush=True)
    with recording as video:
        options = {'viewer': args.viewer, 'video': video}
        if args.protocol == 'standard':
            report = evaluate_standard(config, policy, seeds, args.backend, **options)
        elif args.protocol == 'friction-switch':
            report = evaluate_friction_switch(config, policy, seeds, **options)
        else:
            report = evaluate_rollout(config, policy, seeds, metadata['steps'], **options)
    (condition / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    summary = {key: value for key, value in report.items() if key != 'per_case'}
    if 'groups' not in report:
        summary['per_case'] = report['per_case']
    summary.update(machine=machine, checkpoint_iteration=iteration, assessment=f'{args.condition}/assessment.json')
    if args.video:
        summary['video'] = metadata['video']
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
