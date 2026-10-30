"""SuperMimic-style analytic policy gradients for goal-conditioned Go2 control.

Independent reproduction inspired by Zhengtianjin/go2-supermimic (MIT) and
Liao et al., arXiv:2610.28480. No mjlab code or task-specific assets are required.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn

from rl_go2.env import QuadrupedEnv
from rl_go2.trainers.model_learning import DynamicsModel, ReplayBuffer, RunningNorm, heading, local_state, mlp, pack, unpack


@dataclass
class TrainConfig:
    name: str = 'supermimic'
    iterations: int = 10000
    collect_steps: int = 4
    buffer_steps: int = 256
    warmup_iterations: int = 200
    model_hidden: tuple = (512, 512, 512)
    actor_hidden: tuple = (256, 128, 64)
    critic_hidden: tuple = (256, 128)
    model_window: int = 8
    model_batch: int = 256
    model_updates: int = 4
    imagination_batch: int = 256
    horizon: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    learning_rate: float = 0.0002
    actor_learning_rate: float = 0.0002
    bootstrap_weight: float = 0.0
    driver_policy: str | None = None
    driver_fraction: float = 0.5
    behavior_updates: int = 0
    behavior_weight: float = 0.0
    warmup_learning_rate: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    action_noise: float = 0.1
    grad_norm: float = 1.0
    actor_grad_value: float = 0.1
    height_softness: float = 0.01
    upright_softness: float = 0.03
    save_interval: int = 100
    log_interval: int = 10

    def validate(self):
        integer_fields = ('iterations', 'collect_steps', 'buffer_steps', 'model_window', 'model_batch',
                          'model_updates', 'imagination_batch', 'horizon', 'critic_epochs',
                          'critic_minibatches', 'save_interval', 'log_interval')
        for key in integer_fields:
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.behavior_updates) is not int or self.behavior_updates < 0:
            raise ValueError('behavior_updates must be a nonnegative integer')
        if not math.isfinite(self.behavior_weight) or self.behavior_weight < 0:
            raise ValueError('behavior_weight must be finite and nonnegative')
        if (self.behavior_updates or self.behavior_weight) and self.driver_policy is None:
            raise ValueError('Behavior guidance requires an explicit driver_policy')
        if type(self.warmup_iterations) is not int or self.warmup_iterations < 0:
            raise ValueError('warmup_iterations must be a nonnegative integer')
        if self.buffer_steps < self.model_window:
            raise ValueError('buffer_steps must be >= model_window')
        for key in ('model_hidden', 'actor_hidden', 'critic_hidden'):
            if not getattr(self, key) or any(type(x) is not int or x < 1 for x in getattr(self, key)):
                raise ValueError(f'{key} must contain positive integer widths')
        for key in ('learning_rate', 'actor_learning_rate', 'warmup_learning_rate', 'grad_norm', 'actor_grad_value', 'height_softness', 'upright_softness'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.bootstrap_weight) or not 0 <= self.bootstrap_weight <= 1:
            raise ValueError('bootstrap_weight must be in [0, 1]')
        if not math.isfinite(self.driver_fraction) or not 0 < self.driver_fraction < 1:
            raise ValueError('driver_fraction must be in (0, 1)')
        if not math.isfinite(self.action_noise) or self.action_noise < 0:
            raise ValueError('action_noise must be finite and nonnegative')
        if not 0 <= self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('Expected 0 <= gamma < 1 and 0 <= gae_lambda <= 1')
        if self.critic_minibatches > self.imagination_batch * self.horizon:
            raise ValueError('Too many critic minibatches')


class Actor(nn.Module):
    def __init__(self, obs_dim, actions, hidden, action_clip):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = mlp(obs_dim, hidden, actions)
        self.action_clip = float(action_clip)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observation):
        return self.net(self.norm(observation)).tanh() * self.action_clip


class Critic(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.norm = RunningNorm(inputs)
        self.net = mlp(inputs, hidden, 1)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)


@contextmanager
def frozen(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    previous = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, previous):
            p.requires_grad_(flag)


def optimize(loss, optimizer, parameters, norm, value=None):
    parameters = list(parameters)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite SuperMimic loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = nn.utils.clip_grad_norm_(parameters, norm, error_if_nonfinite=True)
    if value is not None:
        nn.utils.clip_grad_value_(parameters, value)
    optimizer.step()
    return float(grad)


def context_from(record):
    return {key[8:]: value for key, value in record.items() if key.startswith('context/')}


def critic_input(state, previous_action, context, features):
    return torch.cat((local_state(state, heading(state[:, 3:7])), previous_action, features,
                      *(context[key].reshape(state.shape[0], -1).float() for key in sorted(context))), -1)


def imagined_rollout(env, model, actor, critic, seed, cfg):
    state, previous_action, context = seed['state'], seed['previous_action'], context_from(seed)
    survival = torch.ones(state.shape[0], device=state.device)
    objective = torch.zeros_like(survival)
    rewards, continuations, inputs = [], [], []
    for t in range(cfg.horizon):
        robot = unpack(state)
        features = env.task.model_features(robot, context)
        inputs.append(critic_input(state, previous_action, context, features))
        action = actor(env.task.observe(robot, previous_action, context))
        next_state, torque = model(state, action, previous_action, features)
        next_robot = unpack(next_state)
        torque = torque.clamp(-env.controller.limits, env.controller.limits)
        reward = env.task.reward(next_robot, action, previous_action, torque, context)
        continuation = 1 - env.task.termination_probability(next_robot, context, cfg.height_softness, cfg.upright_softness)
        objective = objective + cfg.gamma**t * survival * reward
        survival = survival * continuation
        rewards.append(reward.detach())
        continuations.append(continuation.detach())
        context = env.task.advance_context(context, next_robot)
        state, previous_action = next_state, action
    features = env.task.model_features(unpack(state), context)
    inputs.append(critic_input(state, previous_action, context, features))
    if cfg.bootstrap_weight:
        objective = objective + cfg.bootstrap_weight * cfg.gamma**cfg.horizon * survival * critic(inputs[-1])
    return objective.mean(), {'inputs': torch.stack(inputs, 1).detach(),
                              'rewards': torch.stack(rewards, 1),
                              'continuations': torch.stack(continuations, 1)}, survival.detach().mean()


@torch.no_grad()
def lambda_returns(critic, data, gamma, lam):
    values = critic(data['inputs'])
    target = values[:, -1]
    targets = []
    for t in reversed(range(data['rewards'].shape[1])):
        target = data['rewards'][:, t] + gamma * data['continuations'][:, t] * (
            (1 - lam) * values[:, t + 1] + lam * target)
        targets.append(target)
    return torch.stack(list(reversed(targets)), 1)


class Trainer:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        required = ('model_features', 'advance_context', 'termination_probability')
        if any(not callable(getattr(env.task, key, None)) for key in required):
            raise ValueError(f'Task must implement model-learning hooks: {required}')
        if not env.record_transitions:
            raise ValueError('SuperMimic requires record_transitions=True')
        features = env.task.model_features(env.state, env.task.context())
        state = pack(env.state)
        ci = critic_input(state, env.previous_action, env.task.context(), features)
        self.actor = Actor(env.observe().shape[-1], env.num_actions, cfg.actor_hidden,
                           env.config['robot']['action_clip']).to(env.device)
        self.model = DynamicsModel(features.shape[-1], cfg.model_hidden).to(env.device)
        self.critic = Critic(ci.shape[-1], cfg.critic_hidden).to(env.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_learning_rate, betas=(0.9, 0.95))
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.buffer_steps, env.num_envs, env.device)
        self.driver = None
        self.driver_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if cfg.driver_policy is not None:
            if env.num_envs < 2:
                raise ValueError('A data driver requires at least two environments')
            self.driver = torch.jit.load(str(cfg.driver_policy), map_location=env.device).eval()
            for parameter in self.driver.parameters():
                parameter.requires_grad_(False)
            count = min(env.num_envs - 1, max(1, round(env.num_envs * cfg.driver_fraction)))
            self.driver_mask[-count:] = True

    @torch.no_grad()
    def collect(self):
        rewards, dones, driver_rewards = [], [], []
        task_metrics = {}
        for _ in range(self.cfg.collect_steps):
            obs = self.env.observe()
            self.actor.norm.update(obs)
            features = self.env.task.model_features(self.env.state, self.env.task.context()).detach().clone()
            action = self.actor(obs)
            if self.driver is not None:
                driver_action = self.driver(obs)
                if driver_action.shape != action.shape or not torch.isfinite(driver_action).all():
                    raise ValueError('Data driver must return finite [num_envs, num_actions] actions')
                action = torch.where(self.driver_mask[:, None], driver_action, action)
            action = action + self.cfg.action_noise * torch.randn_like(action)
            _, reward, done, extras = self.env.step(action)
            transition = extras['transition']
            record = {'state': pack(transition.state), 'next_state': pack(transition.next_state),
                      'action': transition.action, 'previous_action': transition.previous_action,
                      'torque': transition.torque, 'terminated': transition.terminated,
                      'truncated': transition.truncated, 'features': features,
                      'reward': reward, 'is_driver': self.driver_mask.clone()}
            record.update({f'context/{key}': value for key, value in transition.context.items()})
            if any(not torch.isfinite(x).all() for x in record.values()) or not torch.isfinite(reward).all():
                raise RuntimeError('Non-finite simulation transition')
            if callable(getattr(self.env.task, 'metrics', None)):
                for name, value in self.env.task.metrics(transition.next_state, transition.context).items():
                    task_metrics.setdefault(name, []).append(value[~self.driver_mask].mean())
            self.replay.add(record)
            self.model.update_stats(record)
            self.critic.norm.update(critic_input(record['state'], record['previous_action'],
                                                 transition.context, features))
            rewards.append(reward[~self.driver_mask].mean())
            dones.append(done[~self.driver_mask].float().mean())
            if self.driver is not None:
                driver_rewards.append(reward[self.driver_mask].mean())
        result = {'real_reward': torch.stack(rewards).mean().item(),
                  'reset_fraction': torch.stack(dones).mean().item()}
        result.update({f'real/{name}': torch.stack(values).mean().item() for name, values in task_metrics.items()})
        if driver_rewards:
            result['driver_reward'] = torch.stack(driver_rewards).mean().item()
        return result

    def fit_model(self):
        losses = []
        for _ in range(self.cfg.model_updates):
            window = self.replay.sample_windows(self.cfg.model_batch, self.cfg.model_window)
            loss = self.model.loss(window)
            optimize(loss, self.model_optimizer, self.model.parameters(), self.cfg.grad_norm)
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def behavior_loss(self, seed):
        if self.driver is None:
            raise ValueError('Behavior guidance requires a data driver')
        observation = self.env.task.observe(unpack(seed['state']), seed['previous_action'], context_from(seed))
        with torch.no_grad():
            target = self.driver(observation).clamp(-self.actor.action_clip, self.actor.action_clip)
        return (self.actor(observation) - target).square().mean()

    def warmup_actor(self):
        losses = []
        for group in self.actor_optimizer.param_groups:
            group['lr'] = self.cfg.warmup_learning_rate
        try:
            for _ in range(self.cfg.behavior_updates):
                loss = self.behavior_loss(self.replay.sample_starts(self.cfg.imagination_batch))
                optimize(loss, self.actor_optimizer, self.actor.parameters(), self.cfg.grad_norm)
                losses.append(loss.item())
        finally:
            for group in self.actor_optimizer.param_groups:
                group['lr'] = self.cfg.actor_learning_rate
        return sum(losses) / len(losses)

    def improve_policy(self):
        cfg = self.cfg
        seed = self.replay.sample_starts(cfg.imagination_batch)
        with frozen(self.model, self.critic):
            objective, data, survival = imagined_rollout(self.env, self.model, self.actor, self.critic, seed, cfg)
            behavior = self.behavior_loss(seed) if cfg.behavior_weight else objective.new_zeros(())
            grad = optimize(-objective + cfg.behavior_weight * behavior, self.actor_optimizer,
                            self.actor.parameters(), cfg.grad_norm, cfg.actor_grad_value)
        stats = {'imagined_return': objective.item(), 'imagined_survival': survival.item(),
                 'actor_grad_norm': grad, 'behavior_loss': behavior.item(), 'critic_loss': 0.0}
        if cfg.bootstrap_weight == 0:
            return stats
        targets = lambda_returns(self.critic, data, cfg.gamma, cfg.gae_lambda).flatten()
        x = data['inputs'][:, :-1].reshape(-1, data['inputs'].shape[-1])
        losses = []
        for _ in range(cfg.critic_epochs):
            for ids in torch.randperm(x.shape[0], device=x.device).tensor_split(cfg.critic_minibatches):
                loss = (self.critic(x[ids]) - targets[ids]).square().mean()
                optimize(loss, self.critic_optimizer, self.critic.parameters(), cfg.grad_norm)
                losses.append(loss.item())
        stats['critic_loss'] = sum(losses) / len(losses)
        return stats

    def save(self, output, iteration):
        checkpoint = {'schema': 'rl-go2.supermimic/v2', 'iteration': iteration, 'trainer': asdict(self.cfg),
                      'actor': self.actor.state_dict(), 'world_model': self.model.state_dict(),
                      'critic': self.critic.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(),
                      'model_optimizer': self.model_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict()}
        torch.save(checkpoint, output / f'model_{iteration}.pt')
        # Export exactly the same normalized actor used during data collection.
        torch.jit.script(self.actor).save(str(output / 'policy.pt'))


def train(config, output):
    from rl_go2.trainers import save_run_spec
    cfg = TrainConfig(**config['trainer'])
    cfg.validate()
    torch.manual_seed(config['seed'])
    # Metadata creation rejects existing directories before creating a simulator.
    save_run_spec(config, output)
    output = Path(output)
    env = QuadrupedEnv(config, record_transitions=True)
    try:
        trainer = Trainer(env, cfg)
        while trainer.replay.size < cfg.model_window:
            trainer.collect()
        with (output / 'metrics.jsonl').open('w') as log:
            for iteration in range(1, cfg.iterations + 1):
                stats = trainer.collect()
                stats['model_loss'] = trainer.fit_model()
                stats['policy_updates'] = max(0, iteration - cfg.warmup_iterations)
                if iteration > cfg.warmup_iterations:
                    stats.update(trainer.improve_policy())
                elif cfg.behavior_updates:
                    stats['behavior_loss'] = trainer.warmup_actor()
                stats.update(iteration=iteration, environment_steps=env.environment_steps)
                log.write(json.dumps(stats, allow_nan=False) + '\n')
                log.flush()
                if iteration % cfg.log_interval == 0 or iteration == cfg.iterations:
                    print(json.dumps(stats, allow_nan=False), flush=True)
                if iteration % cfg.save_interval == 0 or iteration == cfg.iterations:
                    trainer.save(output, iteration)
    finally:
        env.close()
