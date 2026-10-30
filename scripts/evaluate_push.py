"""Reproducible MuJoCo push tests: fixed commands, one finite COM force pulse per trial."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from rl_go2.env import QuadrupedEnv

DIRECTIONS = {'x+': [1, 0, 0], 'x-': [-1, 0, 0], 'y+': [0, 1, 0], 'y-': [0, -1, 0]}
ROOT = Path(__file__).resolve().parents[1]


def load_config(run):
    path = run / 'config.json'
    if not path.exists():
        path = run / 'remote-config.json'
    config = json.loads(path.read_text())
    for entity in config['scene']['entities']:
        if 'file' in entity and not Path(entity['file']).is_file():
            if '/assets/' not in entity['file']:
                raise FileNotFoundError(entity['file'])
            entity['file'] = str(ROOT / 'assets' / entity['file'].split('/assets/', 1)[1])
    config.update(backend='mujoco', device='cpu', num_envs=1, curriculum={'name': 'none'})
    # Fixed commands separate push recovery from random command changes.
    config['task']['name'] = 'flat_walk'
    config['task']['command_resample_seconds'] = 1000
    config['task']['episode_seconds'] = 20
    return config


def record_frame(env, renderer, camera, force, step, dt, speed, magnitude, writer):
    import cv2
    import mujoco
    camera.lookat[:] = [float(env.state.position[0, 0]), float(env.state.position[0, 1]), 0.2]
    renderer.update_scene(env.backend.data, camera=camera)
    for coordinate in range(-20, 21):
        for start, end in (([coordinate, -20, 0.001], [coordinate, 20, 0.001]),
                           ([-20, coordinate, 0.001], [20, coordinate, 0.001])):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                               np.eye(3).reshape(-1), np.array([.25, .25, .25, 1], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1., np.array(start), np.array(end))
            renderer.scene.ngeom += 1
    if np.linalg.norm(force):
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        origin = env.backend.data.xipos[env.backend.base].copy()
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                           np.eye(3).reshape(-1), np.array([1., .1, .1, 1], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .025, origin, origin + force * .008)
        renderer.scene.ngeom += 1
    frame = renderer.render().copy()
    text = f'MuJoCo | target vx={speed:.1f}m/s | pulse={magnitude:.0f}N x 0.2s | t={(step+1)*dt:.2f}s'
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    if np.linalg.norm(force):
        cv2.putText(frame, 'PUSH ACTIVE', (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 80, 80), 2)
    writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forces', nargs='+', type=float, default=[0, 20, 40, 60, 80, 100])
    parser.add_argument('--speeds', nargs='+', type=float, default=[.3, .5, .7])
    parser.add_argument('--directions', nargs='+', choices=list(DIRECTIONS), default=list(DIRECTIONS))
    parser.add_argument('--video', action='store_true', help='Record the 80N y+ 0.5m/s trial')
    args = parser.parse_args()
    if any(not math.isfinite(x) or x < 0 for x in args.forces) or any(not math.isfinite(x) for x in args.speeds):
        parser.error('Forces must be finite and nonnegative; speeds must be finite')
    if args.video and not (80 in args.forces and .5 in args.speeds and 'y+' in args.directions):
        parser.error('--video requires the 80N y+ 0.5m/s condition')
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.run_dir)
    config['task']['command_ranges'] = [[args.speeds[0], args.speeds[0]], [0, 0], [0, 0]]
    env = QuadrupedEnv(config, record_transitions=True)
    renderer = writer = camera = None
    try:
        policy = torch.jit.load(str(args.run_dir / 'policy.pt'), map_location='cpu').eval()
        if args.video:
            import imageio.v2 as imageio
            import mujoco
            renderer = mujoco.Renderer(env.backend.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -20
            env.backend.model.geom('ground').rgba[:] = [.75, .78, .8, 1]
            env.backend.model.vis.headlight.ambient[:] = [.6, .6, .6]
            env.backend.model.vis.headlight.diffuse[:] = [.7, .7, .7]
            writer = imageio.get_writer(str(args.output / 'push-80N-yplus.mp4'), fps=25, codec='libx264', quality=7)
        results = []
        with torch.inference_mode():
            for magnitude in args.forces:
                for direction in (args.directions if magnitude else ['x+']):
                    for speed in args.speeds:
                        env.task.ranges[:] = torch.tensor([[speed, speed], [0., 0.], [0., 0.]])
                        env.reset()
                        force = torch.tensor([DIRECTIONS[direction]], dtype=torch.float32) * magnitude
                        start, end = round(3/env.dt), round(3.2/env.dt)
                        recovered, streak, fallen, max_tilt, max_error = None, 0, False, 0., 0.
                        errors, reward_sum = [], 0.
                        for step in range(round(8/env.dt)):
                            applied = force if start <= step < end else torch.zeros_like(force)
                            observation, reward, done, info = env.step(policy(env.observe()), base_force=applied)
                            state = info['transition'].next_state
                            assert torch.isfinite(observation).all() and torch.isfinite(reward).all()
                            linear, angular, gravity = env.task.body_state(state)
                            error = float(torch.linalg.vector_norm(linear[0, :2] - torch.tensor([speed, 0.])))
                            tilt = math.degrees(math.acos(float((-gravity[0, 2]).clamp(-1, 1))))
                            errors.append(error)
                            reward_sum += float(reward.sum())
                            if step >= start:
                                max_tilt, max_error = max(max_tilt, tilt), max(max_error, error)
                            fallen = bool(info['terminated'].any())
                            if step >= end and magnitude and recovered is None:
                                stable = error < .1 and abs(float(angular[0, 2])) < .15 and tilt < 15
                                streak = streak + 1 if stable else 0
                                if streak >= round(.5/env.dt):
                                    recovered = max(0., (step-end+1)*env.dt - .5)
                            if writer is not None and magnitude == 80 and direction == 'y+' and speed == .5 and step % 2 == 0:
                                record_frame(env, renderer, camera, applied[0].numpy(), step, env.dt, speed, magnitude, writer)
                            if fallen:
                                break
                        result = {'force_n': magnitude, 'duration_s': .2, 'direction_world': direction,
                                  'command_speed_m_s': speed, 'fell': fallen,
                                  'recovery_time_s': recovered if not fallen else None,
                                  'max_tilt_after_push_deg': max_tilt, 'max_velocity_error_after_push_m_s': max_error,
                                  'velocity_rmse_m_s': float(np.sqrt(np.mean(np.square(errors)))),
                                  'simulated_seconds': (step+1)*env.dt, 'return': reward_sum}
                        results.append(result)
                cases = [r for r in results if r['force_n'] == magnitude]
                print(json.dumps({'force_n': magnitude, 'trials': len(cases), 'falls': sum(r['fell'] for r in cases)}), flush=True)
        groups = []
        for magnitude in args.forces:
            cases = [r for r in results if r['force_n'] == magnitude]
            times = [r['recovery_time_s'] for r in cases if r['recovery_time_s'] is not None]
            groups.append({'force_n': magnitude, 'trials': len(cases), 'falls': sum(r['fell'] for r in cases),
                           'recovered_trials': len(times),
                           'mean_recovery_time_s_among_recovered': float(np.mean(times)) if times else None,
                           'maximum_tilt_deg': max(r['max_tilt_after_push_deg'] for r in cases)})
        report = {'policy': str(args.run_dir / 'policy.pt'), 'backend': 'mujoco',
                  'protocol': {'trial_seconds': 8, 'push_start_s': 3, 'push_duration_s': .2,
                               'force_frame': 'world', 'force_application_point': 'base_link_COM',
                               'recovery_velocity_error_xy_m_s': .1, 'recovery_yaw_rate_rad_s': .15,
                               'recovery_tilt_deg': 15, 'recovery_hold_s': .5,
                               'command_speeds_m_s': args.speeds,
                               'note': 'Deterministic fixed test conditions, not an estimated population success probability'},
                  'summary': groups, 'trials': results}
        (args.output / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(groups, indent=2), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == '__main__':
    main()
