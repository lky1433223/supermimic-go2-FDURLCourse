"""Load the evaluation inputs, run a task, and save its artifacts."""
from contextlib import nullcontext
from dataclasses import asdict, replace
from hashlib import file_digest
import json
from pathlib import Path
import socket
import sys

from rl_go2.config import FLAT_SCENE_PATH, load_scene, validate_scene_robot

from .executor import execute_plan, summarize_plan
from .push_events import build_push_events
from .task import EvaluationPlan
from .video import EvaluationVideo


def load_policy(args, training, config, *, require_iteration: bool = True):
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
    if iteration is None and require_iteration:
        raise ValueError('Specify --iteration when the checkpoint iteration cannot be inferred')
    return policy.to(args.device).eval(), path, iteration


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def output_records(plan: EvaluationPlan, condition_name: str) -> dict[Path, EvaluationPlan]:
    """Choose output directories for the plan and check names used in paths."""
    if not plan.scenes:
        return {Path(condition_name): plan}
    cases = {}
    for trial in plan.trials:
        scene, condition = trial.scene, trial.condition.name
        if any(not name or name in ('.', '..') or Path(name).name != name for name in (scene, condition)):
            raise ValueError('Scene and condition names must be single directory names')
        cases.setdefault((scene, condition), []).append(trial)
    return {
        Path(scene) / condition: replace(plan, trials=tuple(trials), scenes={scene: plan.scenes[scene]})
        for (scene, condition), trials in cases.items()
    }


def configure_evaluation_scene(config: dict, plan: EvaluationPlan) -> dict:
    """Use the flat baseline for legacy suites, with the run's robot and simulator settings."""
    if plan.scenes:
        for scene in plan.scenes.values():
            validate_scene_robot(scene, config['robot'])
        return config

    training_scene = config['scene']
    validate_scene_robot(training_scene, config['robot'])
    training_robot = next(entity for entity in training_scene['entities'] if entity.get('robot'))
    flat_scene = load_scene(FLAT_SCENE_PATH)
    flat_robot = next(entity for entity in flat_scene['entities'] if entity.get('robot'))
    flat_robot.update(id=training_robot['id'], file=training_robot['file'])
    flat_scene['sim_options'] = training_scene['sim_options'].copy()
    return {**config, 'scene': flat_scene}


def run_evaluation(args, task):
    import torch

    from rl_go2.trainers import output_directory

    run = args.run_dir.resolve()
    training = run / 'training' if (run / 'training').is_dir() else run
    output_root = run.parent if run.name == 'training' else run
    config_path = training / 'config.json'
    config = json.loads(config_path.read_text())
    config.update(backend=args.backend, device=args.device, num_envs=args.num_envs, curriculum={'name': 'none'})
    plan = task.build_plan(args.steps)
    plan = plan.select_trials(conditions=args.condition, motions=args.trial)
    config = configure_evaluation_scene(config, plan)
    conditions = tuple({trial.condition.name: trial.condition for trial in plan.trials}.values())
    condition_name = conditions[0].name if len(conditions) == 1 else 'comparison'
    records = output_records(plan, condition_name)
    if task.environment_task is not None:
        config['task']['name'] = task.environment_task
    if plan.episode_seconds is not None:
        config['task'].update(episode_seconds=plan.episode_seconds, command_resample_seconds=plan.episode_seconds)

    torch.set_num_threads(2)
    policy, weight_path, iteration = load_policy(args, training, config)
    with weight_path.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    seeds = args.seeds or list(task.default_seeds)
    push_events = build_push_events(config, plan, seeds)
    host = socket.gethostname()
    machine = 'icc' if output_root.is_relative_to(Path('/home/lky/RL-Go2')) else host.split('.')[0]
    common = {
        'task': task.TASK, 'condition': condition_name, 'backend': args.backend, 'machine': machine,
        'checkpoint_iteration': iteration, 'seeds': seeds,
    }
    dest = args.output or output_directory(
        f'{task.TASK}-{args.backend}-{machine}', output_root / 'evaluation' / f'iteration-{iteration}')
    dest.mkdir(parents=True, exist_ok=False)

    metadata = {
        **common, 'hostname': host, 'weights': str(weight_path), 'weights_sha256': digest,
        'config_source': str(config_path), 'config': conditions[0].apply(plan.scene_config(config, plan.trials[0])),
        'physical_conditions': [asdict(condition) for condition in conditions],
    }
    metadata['trials'] = [asdict(trial) for trial in plan.trials]
    if plan.scenes:
        metadata['scenes'] = plan.scenes
        metadata['cases'] = [
            {'scene': case.trials[0].scene, 'condition': case.trials[0].condition.name,
             'assessment': str(directory / 'assessment.json'),
             **({'video': str(directory / 'replay.mp4')} if args.video else {})}
            for directory, case in records.items()
        ]
    if push_events:
        metadata['point_push_events'] = [
            {'seed': seed, 'condition': condition, 'case': case,
             **({'scene': scene, 'group': group} if scene is not None else {}), **asdict(event)}
            for (seed, scene, group, condition, case), event in push_events.items()
        ]
    motions = {}
    for trial in plan.trials:
        if trial.motion is not None:
            group = motions.setdefault(trial.group or 'straight', [])
            motion = asdict(trial.motion)
            if motion not in group:
                group.append(motion)
    if motions:
        metadata['motion_tasks'] = motions
    if args.steps is not None:
        metadata['steps'] = args.steps
    video_metadata = {}
    views = args.video_view or ['default']
    if args.video:
        video_metadata['video_views'] = views
        if len(views) == 1:
            video_metadata['video_view'] = views[0]
        first_video = next(iter(records)) / 'replay.mp4'
        video_metadata['video_encoding'] = EvaluationVideo(dest / first_video, views=views).encoding
        if not plan.scenes:
            video_metadata['video'] = str(first_video)
    metadata.update(video_metadata)
    write_json(dest / 'eval-config.json', metadata)
    print(f'Evaluation output: {dest.resolve()}', file=sys.stderr, flush=True)
    results = []
    for directory, case in records.items():
        case_dir = dest / directory
        case_dir.mkdir(parents=True)
        recording = EvaluationVideo(case_dir / 'replay.mp4', views=views) if args.video else nullcontext()
        with recording as video:
            report = execute_plan(config, policy, seeds, case, viewer=args.viewer, video=video,
                                  push_events=push_events)
        trial = case.trials[0]
        case_metadata = {**common, 'condition': trial.condition.name, 'scene': trial.scene} if plan.scenes else common
        height_maps = case.scene_config(config, trial)['scene'].get('height_maps')
        if height_maps:
            case_metadata['height_maps'] = height_maps
        write_json(case_dir / 'assessment.json', {**case_metadata, **report})
        results.extend(report['per_case'])

    assessment = {**common, **summarize_plan(config, plan, results)}
    summary = {key: value for key, value in assessment.items() if key != 'per_case' or 'groups' not in assessment}
    if plan.scenes:
        summary['cases'] = metadata['cases']
    else:
        summary['assessment'] = f'{condition_name}/assessment.json'
    summary.update(video_metadata)
    write_json(dest / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
