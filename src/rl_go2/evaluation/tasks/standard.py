"""Standard flat-ground tracking, turning, and stopping."""
import math

from ..runner import environment

TASK = 'standard'
TASK_DESCRIPTION = 'Run 27 velocity tracking, turning, and stopping cases per seed.'

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


def configure(config, parameters):
    config['task'].update(episode_seconds=CASE_SECONDS, command_resample_seconds=CASE_SECONDS)


def run(config, policy, seeds, *, viewer=False, video=None):
    import torch

    results = []
    with torch.inference_mode():
        for seed in seeds:
            with environment(config, seed, viewer=viewer, record_transitions=True) as env:
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
    return {
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
