"""Record a synchronized Genesis and SuperMimic world-model replay."""

import argparse
from contextlib import closing
from datetime import datetime
import json
import math
import socket
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from rl_go2.backends import LinkPointForces
from rl_go2.backends.mujoco import MujocoBackend
from rl_go2.config import FOOT_NAMES, config_at_iteration, resolve_run_asset
from rl_go2.env import QuadrupedEnv
from rl_go2.mujoco_visuals import brighten_scene
from rl_go2.trainers.model_learning import DynamicsModel, pack
from rl_go2.trainers.supermimic import Actor


def wrench_indices(env: QuadrupedEnv, conditioned: bool) -> list[int]:
    """Return backend indices in the order of the model's force inputs."""
    if not conditioned:
        return []
    names = (env.config['robot']['base_link'], *(f'{foot}_calf' for foot in FOOT_NAMES))
    return [env.backend.link_names.index(name) for name in names]


@torch.no_grad()
def collect(env: QuadrupedEnv, checkpoint: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    """Replay one continuous prediction window with Genesis actions and loads."""
    settings = checkpoint['trainer']
    actor = Actor(env.observe().shape[-1], env.num_actions, settings['actor_hidden'],
                  env.config['robot']['action_clip']).to(env.device)
    actor.load_state_dict(checkpoint['actor'])
    indices = wrench_indices(env, settings.get('condition_on_link_wrenches', False))
    if args.push_n and not indices:
        raise ValueError('The checkpoint must support link wrenches to replay an explicit push')
    features = env.task.model_features(env.state, env.model_context())
    model = DynamicsModel(features.shape[-1], settings['model_hidden'], wrench_links=len(indices),
                          terrain_scan=env.terrain_scan,
                          predict_contacts='contact_head.weight' in checkpoint['world_model']).to(env.device)
    model.load_state_dict(checkpoint['world_model'])
    actor.eval()
    model.eval()
    if args.terrain is not None:
        env.select_terrain(args.terrain)
    command = env.task.commands.new_tensor(args.command) if args.command is not None else None
    for _ in range(args.start):
        if command is not None:
            env.task.commands.copy_(command)
        env.step(actor(env.observe()))

    state = pack(env.state)
    actual = [state]
    predicted = [state]
    points = state.new_zeros((1, len(env.backend.link_names), 3))
    forces = torch.zeros_like(points)
    stop_reason = 'horizon_reached'
    for step in range(args.horizon):
        if command is not None:
            env.task.commands.copy_(command)
        loads = None
        if args.push_n and args.push_at <= step < args.push_at + args.push_steps:
            forces[:, indices[0], 1] = args.push_n
            loads = LinkPointForces(points, forces)
        _, _, done, extras = env.step(actor(env.observe()), link_point_forces=loads)
        transition = extras['transition']
        features = env.task.model_features(transition.state, transition.context)
        wrenches = None
        if indices:
            recorded = transition.link_wrenches_world
            wrenches = points.new_zeros((1, len(indices), 6)) if recorded is None else recorded[:, indices]
        state, _, _ = model.predict_transition(state, transition.action, features, wrenches)
        if not torch.isfinite(state).all():
            stop_reason = f'non_finite_prediction_at_step_{step + 1}'
            break
        actual.append(pack(transition.next_state))
        predicted.append(state)
        if done.item():
            stop_reason = 'reference_episode_ended'
            break
    return torch.cat(actual).cpu().numpy(), torch.cat(predicted).cpu().numpy(), stop_reason


def render(output: Path, config: dict, actual: np.ndarray, predicted: np.ndarray, dt: float,
           *, realtime: bool = False) -> None:
    """Place both poses in the same scene and render without advancing physics."""
    import cv2
    import imageio.v2 as imageio
    import mujoco

    backend = MujocoBackend({**config, 'device': 'cpu', 'num_envs': 1})
    model, data = backend.model, backend.data
    brighten_scene(model)
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    # MuJoCo expresses clip distances in units of the whole scene's extent.
    # The training map spans many distant patches, but this camera is 2 m away.
    model.vis.map.znear = 0.025 / model.stat.extent
    model.vis.map.zfar = 20.0 / model.stat.extent
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 115, -14, 1.4
    with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(
        str(output), fps=20, codec='libx264',
    ) as writer:
        frame_count = max(1, math.ceil((len(actual) - 1) * dt * 20))
        steps = (min(round(frame / (20 * dt)), len(actual) - 1) for frame in range(frame_count))
        for step in steps if realtime else range(len(actual)):
            truth, estimate = actual[step], predicted[step]
            images = []
            tracks = (('Genesis', truth, actual, (0.2, 0.8, 1.0, 1.0)),
                      ('World model', estimate, predicted, (1.0, 0.6, 0.2, 1.0)))
            for label, state, track, color in tracks:
                camera.lookat[:] = (*state[:2], 0.2)
                data.qpos[:7] = state[:7]
                data.qpos[backend.q_indices] = state[13:25]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                # Bound trail geometry and rendering work for long predictions.
                trail = track[:step + 1:max(1, math.ceil((step + 1) / 512)), :3]
                draw_track(mujoco, renderer.scene, trail, color)
                image = renderer.render().copy()
                cv2.rectangle(image, (0, 0), (640, 35), (22, 22, 22), -1)
                title = f'{label} | step {step}/{len(actual) - 1} | {1000 * dt * step:.0f} ms'
                cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (245, 245, 245), 1, cv2.LINE_AA)
                images.append(image)
            frame = np.concatenate(images, axis=1)
            for _ in range(12 if not realtime and step in (0, len(actual) - 1) else 1):
                writer.append_data(frame)


def draw_track(mujoco, scene, positions: np.ndarray, color: tuple[float, ...]) -> None:
    """Trace the base in world coordinates from the shared start pose."""
    line = mujoco.mjtGeom.mjGEOM_LINE
    for start, end in zip(positions[:-1], positions[1:], strict=True):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, line, np.zeros(3), start, np.eye(3).ravel(), np.array(color, dtype=np.float32))
        mujoco.mjv_connector(geom, line, 3.0, start, end)
        scene.ngeom += 1


def main() -> None:
    """Load the checkpoint's effective course, collect, and export the replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--iteration', required=True, type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=201)
    parser.add_argument('--terrain', type=int, help='Index in the saved scene terrain_patches')
    parser.add_argument('--command', nargs=3, type=float, metavar=('VX', 'VY', 'YAW_RATE'))
    parser.add_argument('--start', type=int, default=50, help='Genesis warmup steps before the window')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--horizon', type=int, default=32)
    duration.add_argument('--seconds', type=float,
                          help='Continuous prediction duration in simulation seconds; render at real-time speed')
    parser.add_argument('--push-n', type=float, default=0., help='Optional lateral base push along world +Y')
    parser.add_argument('--push-at', type=int, default=0, help='Zero-based prediction step to start pushing')
    parser.add_argument('--push-steps', type=int, default=5)
    args = parser.parse_args()
    if args.start < 0 or args.horizon < 1 or args.iteration < 1:
        parser.error('Require a nonnegative start and positive horizon and iteration')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be finite and positive')
    if not math.isfinite(args.push_n) or (args.command and not all(map(math.isfinite, args.command))):
        parser.error('Push and command must be finite')
    if args.push_n and not (0 <= args.push_at < args.horizon and args.push_steps > 0):
        parser.error('Require a push start inside the prediction window and positive push steps')

    training = args.run_dir / 'training'
    weights = training / f'model_{args.iteration}.pt'
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != 'rl-go2.supermimic/v2' or checkpoint['iteration'] != args.iteration:
        parser.error('Expected the matching SuperMimic v2 checkpoint')
    config = config_at_iteration(json.loads((training / 'config.json').read_text()), args.iteration)
    for entity in config['scene']['entities']:
        if 'file' in entity:
            entity['file'] = resolve_run_asset(entity['file'])
    if config['task']['name'] != 'flat_walk':
        parser.error('This replay supports flat_walk checkpoints')
    patches = config['scene'].get('terrain_patches', [])
    if args.terrain is not None and not 0 <= args.terrain < len(patches):
        parser.error(f'Expected a terrain index in [0, {len(patches)})')

    config.update(backend='genesis', device=args.device, num_envs=1, seed=args.seed)
    if args.seconds is not None:
        control_dt = config['scene']['sim_options']['dt'] * config['robot']['control_decimation']
        args.horizon = math.ceil(args.seconds / control_dt)
        config['task']['episode_seconds'] = max(
            config['task']['episode_seconds'], (args.start + args.horizon + 1) * control_dt,
        )
    torch.manual_seed(args.seed)
    with closing(QuadrupedEnv(config, record_transitions=True)) as env:
        actual, predicted, stop_reason = collect(env, checkpoint, args)
        dt = env.dt

    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    machine = 'icc' if weights.resolve().is_relative_to('/home/lky/RL-Go2') else socket.gethostname()
    output = args.output or args.run_dir / 'evaluation' / f'iteration-{args.iteration}' / (
        f'world-model-genesis-{machine}-{timestamp}'
    )
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**vars(args), 'run_dir': str(args.run_dir.resolve()), 'output': str(output.resolve()),
                'checkpoint': str(weights.resolve()), 'control_dt_seconds': dt, 'machine': machine,
                'reference_backend': 'genesis', 'render_backend': 'mujoco',
                'recorded_steps': len(actual) - 1,
                'recorded_seconds': (len(actual) - 1) * dt,
                'stop_reason': stop_reason, 'prediction_state_resets': 0,
                'reference_episode_seconds': config['task']['episode_seconds']}
    (output / 'eval-config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez_compressed(output / 'trajectory.npz', genesis=actual, world_model=predicted)
    render(output / 'replay.mp4', config, actual, predicted, dt, realtime=args.seconds is not None)
    print(output.resolve())


if __name__ == '__main__':
    main()
