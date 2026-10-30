"""Small real-simulator learning/export check; this does not establish walking quality."""
import argparse
import json
from pathlib import Path

import torch

from rl_go2.config import load_experiment
from rl_go2.env import QuadrupedEnv
from rl_go2.trainers.supermimic import train
from rl_go2.trainers import output_directory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('genesis', 'mujoco'), default='genesis')
    parser.add_argument('--output', type=Path, help='Defaults to output/supermimic-check-<timestamp>')
    args = parser.parse_args()
    if args.output is None:
        args.output = output_directory('supermimic-check')
    print(f'Check output: {args.output.resolve()}', flush=True)
    torch.set_num_threads(2)
    config = load_experiment(ROOT / 'configs/experiments/flat_supermimic.yaml')
    config.update(backend=args.backend, num_envs=4 if args.backend == 'genesis' else 1)
    config['trainer'].update(iterations=30, warmup_iterations=10, bootstrap_weight=1.0, save_interval=10, log_interval=10,
                             model_hidden=[128, 128], actor_hidden=[64, 64], critic_hidden=[64, 64],
                             model_batch=32, model_updates=2, imagination_batch=32, horizon=8,
                             critic_epochs=2, buffer_steps=64)
    train(config, args.output)
    early = torch.load(args.output / 'model_10.pt', weights_only=True)
    final = torch.load(args.output / 'model_30.pt', weights_only=True)
    changed = lambda key: any(not torch.equal(v, early[key][name]) for name, v in final[key].items() if name.startswith('net.'))
    report = {'training_backend': args.backend, 'iterations': 30, 'policy_updates': 20,
              'actor_parameters_changed': changed('actor'), 'critic_parameters_changed': changed('critic'),
              'world_model_parameters_changed': changed('world_model')}
    assert all(report[key] for key in ('actor_parameters_changed', 'critic_parameters_changed', 'world_model_parameters_changed'))
    metrics = [json.loads(line) for line in (args.output / 'metrics.jsonl').read_text().splitlines()]
    assert all(torch.isfinite(torch.tensor(value)) for item in metrics for value in item.values())
    report['first_metrics'], report['last_metrics'] = metrics[0], metrics[-1]
    policy = torch.jit.load(str(args.output / 'policy.pt')).eval()
    config.update(backend='mujoco', num_envs=1, device='cpu')
    torch.manual_seed(config['seed'])
    env = QuadrupedEnv(config, record_transitions=True)
    rewards, errors, resets = [], [], 0
    try:
        with torch.inference_mode():
            for _ in range(256):
                observation, reward, done, extras = env.step(policy(env.observe()))
                assert torch.isfinite(observation).all() and torch.isfinite(reward).all()
                transition = extras['transition']
                linear, angular, _ = env.task.body_state(transition.next_state)
                velocity = torch.cat((linear[:, :2], angular[:, 2:3]), -1)
                errors.append((velocity - transition.context['commands']).square())
                rewards.append(reward)
                resets += int(done.sum())
        report['mujoco_evaluation'] = {'steps': 256, 'seconds': 256 * env.dt, 'resets': resets,
                                        'mean_reward': torch.cat(rewards).mean().item(),
                                        'velocity_rmse': torch.cat(errors).mean(0).sqrt().tolist()}
    finally:
        env.close()
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
