"""Reproducible real-simulator speed tracking evaluation, independent of training loss."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from rl_go2.env import QuadrupedEnv
from rl_go2.trainers.supermimic import Actor


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
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-tracking', action='store_true')
    args = parser.parse_args()
    if args.steps < 2 or not args.seeds:
        parser.error('steps must be >= 2 and seeds must be nonempty')
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
    report = evaluate(config, policy, args.seeds, args.steps, args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'per_seed'}, indent=2))
    if args.require_tracking and not report['tracking_passed']:
        raise SystemExit('Speed tracking acceptance failed; see evaluation report')


if __name__ == '__main__':
    main()
