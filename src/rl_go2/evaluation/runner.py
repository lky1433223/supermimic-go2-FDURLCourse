"""Load the evaluation inputs, run a task, and save its artifacts."""
from contextlib import contextmanager, nullcontext
from hashlib import file_digest
import json
from pathlib import Path
import socket
import sys


@contextmanager
def environment(config, seed, *, viewer, record_transitions):
    import torch

    from rl_go2.env import QuadrupedEnv

    torch.manual_seed(seed)
    env = QuadrupedEnv({**config, 'seed': seed}, viewer=viewer, record_transitions=record_transitions)
    try:
        yield env
    finally:
        env.close()


def load_policy(args, training, config):
    import torch

    path = (args.checkpoint or args.policy or training / 'policy.pt').resolve()
    iteration = args.iteration
    if args.checkpoint:
        from rl_go2.trainers.supermimic import Actor

        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        policy = Actor(checkpoint['actor']['norm.mean'].numel(), len(config['robot']['joint_names']),
                       checkpoint['trainer']['actor_hidden'], config['robot']['action_clip'])
        policy.load_state_dict(checkpoint['actor'])
        if iteration is None:
            iteration = checkpoint.get('iteration')
    else:
        policy = torch.jit.load(str(path), map_location='cpu')

    if iteration is None and args.policy is None:
        candidates = [path] if args.checkpoint else training.glob('model_*.pt')
        iteration = max((int(p.stem[6:]) for p in candidates
                         if p.stem.startswith('model_') and p.stem[6:].isdigit()), default=None)
    if iteration is None:
        raise ValueError('Specify --iteration when the checkpoint iteration cannot be inferred')
    return policy.to(args.device).eval(), path, iteration


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run_evaluation(args, task, condition_name, parameters):
    import torch

    from rl_go2.trainers import output_directory

    run = args.run_dir.resolve()
    training = run / 'training' if (run / 'training').is_dir() else run
    output_root = run.parent if run.name == 'training' else run
    config_path = training / 'config.json'
    config = json.loads(config_path.read_text())
    config.update(backend=args.backend, device=args.device, num_envs=args.num_envs, curriculum={'name': 'none'})
    configure = getattr(task, 'configure', None)
    if configure is not None:
        configure(config, parameters)

    torch.set_num_threads(2)
    policy, weight_path, iteration = load_policy(args, training, config)
    with weight_path.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    seeds = args.seeds or ([config['seed']] if task.TASK == 'rollout' else [1])
    host = socket.gethostname()
    machine = 'icc' if output_root.is_relative_to(Path('/home/lky/RL-Go2')) else host.split('.')[0]
    common = {
        'task': task.TASK, 'condition': condition_name, 'backend': args.backend, 'machine': machine,
        'checkpoint_iteration': iteration, 'seeds': seeds,
    }
    dest = args.output or output_directory(
        f'{task.TASK}-{args.backend}-{machine}', output_root / 'evaluation' / f'iteration-{iteration}')
    dest.mkdir(parents=True, exist_ok=False)
    condition = dest / condition_name
    condition.mkdir()

    metadata = {
        **common, 'hostname': host, 'weights': str(weight_path), 'weights_sha256': digest,
        'config_source': str(config_path), 'task_config_source': str(args.config.resolve()) if args.config else None,
        'parameters': parameters, 'config': config,
    }
    run_options = {}
    if args.steps is not None:
        metadata['steps'] = args.steps
        run_options['steps'] = args.steps
    recording = nullcontext()
    video_metadata = {}
    if args.video:
        from .video import EvaluationVideo

        views = args.video_view or ['default']
        video_metadata = {'video': f'{condition_name}/replay.mp4', 'video_views': views}
        if len(views) == 1:
            video_metadata['video_view'] = views[0]
        recording = EvaluationVideo(condition / 'replay.mp4', views=views)
    metadata.update(video_metadata)
    write_json(dest / 'eval-config.json', metadata)
    print(f'Evaluation output: {dest.resolve()}', file=sys.stderr, flush=True)
    with recording as video:
        report = task.run(config, policy, seeds, viewer=args.viewer, video=video, **run_options)

    assessment = {**common, **report}
    write_json(condition / 'assessment.json', assessment)
    summary = {key: value for key, value in assessment.items() if key != 'per_case' or 'groups' not in assessment}
    summary['assessment'] = f'{condition_name}/assessment.json'
    summary.update(video_metadata)
    write_json(dest / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
