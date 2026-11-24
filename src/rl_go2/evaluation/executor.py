"""Execute all evaluation trials through one simulation loop."""
from contextlib import ExitStack, contextmanager
import math

import torch

from rl_go2.env import QuadrupedEnv

from .metrics import METRICS
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


def execute_trial(env: QuadrupedEnv, policy, seed: int, trial: Trial, video) -> dict:
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
    for step in range(budget):
        if step in commands:
            command = commands[step]
        if schedule:
            env.task.commands.copy_(command)
        if step in phases:
            trial.condition.apply_phase(env, phases[step])
        observation, reward, done, extras = env.step(policy(env.observe()))
        transition = extras['transition']
        if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
            raise RuntimeError(f'Non-finite simulation output in {trial.name} at step {step + 1}')
        metrics.update(env, transition, reward, done, observation, step)
        finished = step + 1 == budget or (bool(done.any()) and not trial.continue_after_reset)
        if video is not None:
            video.frame(env, transition, step=step + 1,
                        label=metrics.label(seed), done=finished or bool(done[0]))
        if finished:
            break
    return {'seed': seed, 'condition': trial.condition.name, **metrics.finish(env)}


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
    if all(trial.continue_after_reset for trial in plan.trials):
        report.update(device=config['device'], num_envs=config['num_envs'])
    return report


def execute_plan(config: dict, policy, seeds: list[int], plan: EvaluationPlan,
                 *, viewer: bool = False, video=None) -> dict:
    """Run the plan, sharing an environment across consecutive trials of a condition."""
    results = []
    with torch.inference_mode(), ExitStack() as resources:
        for seed in seeds:
            active_condition = None
            for trial in plan.trials:
                if trial.condition is not active_condition:
                    resources.close()
                    env = resources.enter_context(environment(config, seed, trial.condition, viewer=viewer))
                    active_condition = trial.condition
                results.append(execute_trial(env, policy, seed, trial, video))
    return summarize_plan(config, plan, results)
