"""Compare fixed and time-varying foot friction on flat ground."""
from rl_go2.config import FOOT_NAMES

from ..runner import environment

TASK = 'friction-switch'
TASK_DESCRIPTION = 'Compare fixed friction with a timed 0.8 to 0.4 to 0.8 change.'


def configure(config, parameters):
    config['task'].update(episode_seconds=16, command_resample_seconds=16)
    config['domain_randomization'] = {'foot_ground_friction': {
        'enabled': True, 'env_multiplier_range': [1.0, 1.0],
        'base_ranges': {foot: {'sliding': [0.8, 0.8], 'torsional': [0.02, 0.02], 'rolling': [0.01, 0.01]}
                        for foot in FOOT_NAMES},
    }}


def run(config, policy, seeds, *, viewer=False, video=None):
    import mujoco
    import torch

    results = []
    with torch.inference_mode():
        for seed in seeds:
            for case, ratios in (('fixed', (1., 1., 1.)), ('switch', (1., 0.5, 1.))):
                with environment(config, seed, viewer=viewer, record_transitions=True) as env:
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
    return {'command': [0.5, 0, 0], 'segment_seconds': 5, 'per_case': results}
