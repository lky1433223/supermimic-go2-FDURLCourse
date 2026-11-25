"""Execute all evaluation trials through one simulation loop."""
from contextlib import ExitStack, contextmanager
import math

import torch

from rl_go2.env import QuadrupedEnv

from .metrics import METRICS, PushRecoveryMetrics
from .push_events import PushEvent
from .task import EvaluationPlan, Trial


@contextmanager
def environment(config: dict, seed: int, condition, *, viewer: bool):
    """Create and close a simulator with this seed and physical condition."""
    torch.manual_seed(seed)
    env_config = condition.apply({**config, 'seed': seed})
    env = QuadrupedEnv(env_config, viewer=viewer, record_transitions=True)
    try:
        yield env
    finally:
        env.close()


def execute_trial(env: QuadrupedEnv, policy, seed: int, trial: Trial, video,
                  *, push_event: PushEvent | None = None) -> dict:
    """Run one trial, applying commands and physical phases before inference."""
    if trial.reset_before_start:
        env.reset()
    schedule = trial.motion.command_schedule(env.dt) if trial.motion is not None else {}
    commands = {step: env.task.commands.new_tensor(values) for step, values in schedule.items()}
    budget = trial.steps
    if budget is None:
        budget = math.ceil(trial.motion.duration_seconds / env.dt)
    phases = trial.condition.phase_schedule(env.dt)
    metrics = METRICS[trial.metrics](env, trial)
    loads = None
    recovery = None
    onset_step = None
    push_end_step = None
    applied_steps = 0
    for step in range(budget):
        if step in commands:
            command = commands[step]
        if schedule:
            env.task.commands.copy_(command)
        if step in phases:
            trial.condition.apply_phase(env, phases[step])
        action = policy(env.observe())
        if push_event is not None and onset_step is None and push_event.ready(step, env.state.position[0]):
            onset_step = step
            push_end_step = step + push_event.duration_steps
            loads = push_event.loads(env)
            recovery = PushRecoveryMetrics(push_end_step, env.dt)
        if push_end_step is not None and step < push_end_step:
            observation, reward, done, extras = env.step(action, link_point_forces=loads)
            applied_steps += 1
        else:
            observation, reward, done, extras = env.step(action)
        transition = extras['transition']
        if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
            raise RuntimeError(f'Non-finite simulation output in {trial.name} at step {step + 1}')
        metrics.update(env, transition, reward, done, observation, step)
        if recovery is not None:
            recovery.update(env, transition, step)
        finished = step + 1 == budget or (bool(done.any()) and not trial.continue_after_reset)
        if video is not None:
            video.frame(env, transition, step=step + 1,
                        label=metrics.label(seed), done=finished or bool(done[0]))
        if finished:
            break
    result = metrics.finish(env)
    if push_event is not None:
        if recovery is None:
            recovery_result = {
                'status': 'fell_before_push' if result['fell'] else 'not_triggered',
                'recovery_seconds': None,
            }
        else:
            recovery_result = recovery.finish(result['fell'])
            recovery_result.update(start_step=onset_step, start_seconds=onset_step * env.dt,
                                   end_step=push_end_step, applied_steps=applied_steps,
                                   force_world_n=loads.force_world_n[0].sum(dim=0).tolist())
        result['push_recovery'] = recovery_result
        if 'success' in result and recovery_result['status'] != 'recovered':
            result['success'] = False
            result['failure_reasons'].append(f"push_{recovery_result['status']}")
    if trial.scene is not None:
        result['scene'] = trial.scene
    return {'seed': seed, 'condition': trial.condition.name, **result}


def summarize_plan(config: dict, plan: EvaluationPlan, results: list[dict]) -> dict:
    """Aggregate trial results using the plan's declared groups."""
    report = {**plan.report, 'per_case': results}
    groups = tuple(dict.fromkeys(trial.group for trial in plan.trials if trial.group))
    if groups:
        report['groups'] = {
            group: {'trials': sum(row['group'] == group for row in results),
                    'falls': sum(row['group'] == group and row['fell'] for row in results)}
            for group in groups
        }
        by_condition = {}
        for row in results:
            counts = by_condition.setdefault(row['condition'], {'trials': 0, 'falls': 0})
            counts['trials'] += 1
            counts['falls'] += int(row['fell'])
        if len(by_condition) > 1:
            report['by_condition'] = by_condition
    push_outcomes = {}
    for row in results:
        if 'push_recovery' in row:
            counts = push_outcomes.setdefault(row['condition'], {})
            status = row['push_recovery']['status']
            counts[status] = counts.get(status, 0) + 1
    if push_outcomes:
        report['push_outcomes'] = push_outcomes
    if plan.scenes:
        report['by_scene'] = {name: {'trials': 0, 'falls': 0} for name in plan.scenes}
        for row in results:
            counts = report['by_scene'][row['scene']]
            counts['trials'] += 1
            counts['falls'] += int(row['fell'])
            if 'success' in row:
                counts['successes'] = counts.get('successes', 0) + int(row['success'])
                counts['failures'] = counts.get('failures', 0) + int(not row['success'])
    if all(trial.continue_after_reset for trial in plan.trials):
        report.update(device=config['device'], num_envs=config['num_envs'])
    return report


def execute_plan(config: dict, policy, seeds: list[int], plan: EvaluationPlan,
                 *, viewer: bool = False, video=None,
                 push_events: dict[tuple, PushEvent] | None = None) -> dict:
    """Run the plan, sharing an environment while scene and condition stay the same."""
    results = []
    events = push_events or {}
    with torch.inference_mode(), ExitStack() as resources:
        for seed in seeds:
            active_environment = None
            for trial in plan.trials:
                environment_key = (trial.scene, trial.condition.name)
                if environment_key != active_environment:
                    resources.close()
                    trial_config = plan.scene_config(config, trial)
                    env = resources.enter_context(environment(trial_config, seed, trial.condition, viewer=viewer))
                    active_environment = environment_key
                event = events[trial.event_key(seed)] if trial.condition.point_push is not None else None
                results.append(execute_trial(env, policy, seed, trial, video, push_event=event))
    return summarize_plan(config, plan, results)
