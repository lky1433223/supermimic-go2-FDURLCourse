"""Smoke-check a saved policy by running it in simulation."""

from ..runner import environment

TASK = 'rollout'
TASK_DESCRIPTION = 'Run a saved policy for a fixed number of simulation steps per seed.'
DEFAULT_STEPS = 1000


def run(config, policy, seeds, *, viewer=False, video=None, steps=DEFAULT_STEPS):
    import torch

    results = []
    with torch.inference_mode():
        for seed in seeds:
            total_reward = 0.0
            resets = 0
            with environment(config, seed, viewer=viewer, record_transitions=video is not None) as env:
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
    return {'device': config['device'], 'num_envs': config['num_envs'], 'per_case': results}
