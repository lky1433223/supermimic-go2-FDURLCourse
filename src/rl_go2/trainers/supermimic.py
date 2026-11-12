"""SuperMimic-style analytic policy gradients for goal-conditioned Go2 control.

Independent reproduction inspired by Zhengtianjin/go2-supermimic (MIT) and
Liao et al., arXiv:2610.28480. No mjlab code or task-specific assets are required.
"""
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path
from time import perf_counter

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
    # Enables the official replay schedule, complete frame windows and critic shuffling.
    official_replay_ratio: int | None = None
    imagination_batch: int = 256
    horizon: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    learning_rate: float = 0.0002
    actor_learning_rate: float = 0.0002
    tf32: bool = False
    fused_adam: bool = False
    compile_training: bool = False
    compile_policy_losses: bool = False
    cuda_graph_model: bool = False
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
        if type(self.tf32) is not bool:
            raise ValueError('tf32 must be a boolean')
        if type(self.fused_adam) is not bool:
            raise ValueError('fused_adam must be a boolean')
        if type(self.compile_training) is not bool:
            raise ValueError('compile_training must be a boolean')
        if type(self.compile_policy_losses) is not bool:
            raise ValueError('compile_policy_losses must be a boolean')
        if type(self.cuda_graph_model) is not bool:
            raise ValueError('cuda_graph_model must be a boolean')
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
        if self.official_replay_ratio is not None:
            if type(self.official_replay_ratio) is not int or self.official_replay_ratio < 1:
                raise ValueError('official_replay_ratio must be a positive integer')
            if self.model_window < 2 or self.warmup_iterations or self.behavior_updates:
                raise ValueError('Official replay requires at least two frames and no separate warmup')
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


def optimize(loss, optimizer, parameters, norm, value=None, *, backward=True):
    parameters = list(parameters)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite SuperMimic loss')
    if backward:
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


def imagined_step(env, model, actor, cfg, state, previous_action, context):
    robot = unpack(state)
    features = env.task.model_features(robot, context)
    inputs = None
    if cfg.bootstrap_weight:
        with torch.no_grad():
            inputs = critic_input(state, previous_action, context, features)
    action = actor(env.task.observe(robot, previous_action, context))
    _, action = env.controller.target(action)
    next_state, torque = model(state, action, previous_action, features)
    next_robot = unpack(next_state)
    torque = torque.clamp(-env.controller.limits, env.controller.limits)
    reward = env.task.reward(next_robot, action, previous_action, torque, context)
    continuation = 1 - env.task.termination_probability(next_robot, context, cfg.height_softness, cfg.upright_softness)
    return next_state, action, reward, continuation, inputs


def imagined_return(rewards, continuations, gamma):
    survival = torch.ones_like(rewards[0])
    objective = torch.zeros_like(survival)
    for t, (reward, continuation) in enumerate(zip(rewards, continuations)):
        objective = objective + gamma**t * survival * reward
        survival = survival * continuation
    return objective, survival


def critic_minibatch_loss(critic, inputs, targets, ids):
    return (critic(inputs[ids]) - targets[ids]).square().mean()


def imagined_rollout(env, model, actor, critic, seed, cfg, *, step=None, returns=imagined_return):
    if step is None:
        step = partial(imagined_step, env, model, actor, cfg)
    state, previous_action, context = seed['state'], seed['previous_action'], context_from(seed)
    rewards, continuations, inputs = [], [], []
    for _ in range(cfg.horizon):
        state, action, reward, continuation, critic_obs = step(state, previous_action, context)
        rewards.append(reward)
        continuations.append(continuation)
        if cfg.bootstrap_weight:
            inputs.append(critic_obs)
        context = env.task.advance_context(context, unpack(state))
        previous_action = action
    objective, survival = returns(rewards, continuations, cfg.gamma)
    if cfg.bootstrap_weight:
        features = env.task.model_features(unpack(state), context)
        final_input = critic_input(state, previous_action, context, features)
        objective = objective + cfg.bootstrap_weight * cfg.gamma**cfg.horizon * survival * critic(final_input)
        inputs.append(final_input.detach())
        data = {'inputs': torch.stack(inputs, 1),
                'rewards': torch.stack([reward.detach() for reward in rewards], 1),
                'continuations': torch.stack([value.detach() for value in continuations], 1)}
    else:
        data = {}
    return objective.mean(), data, survival.detach().mean()


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


class GraphedModelBackward:
    def __init__(self, loss_fn, optimizer, window, device):
        self.window = {key: value.clone() for key, value in window.items()}
        self.stream = torch.cuda.Stream(device=device)
        self.graph = torch.cuda.CUDAGraph()
        caller = torch.cuda.current_stream(device)
        self.stream.wait_stream(caller)
        # Keep AccumulateGrad on the same stream during warmup and capture.
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss_fn(self.window).backward()
            # Allocate gradients inside the graph so replay overwrites them.
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph, stream=self.stream):
                loss = loss_fn(self.window)
                loss.backward()
                self.loss = loss.detach()
        caller.wait_stream(self.stream)

    def __call__(self, window):
        caller = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            for key, value in window.items():
                self.window[key].copy_(value)
            self.graph.replay()
        caller.wait_stream(self.stream)
        return self.loss


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
        self.model_loss = self.model.loss
        self.graphed_model_backward = None
        self.imagined_step = partial(imagined_step, env, self.model, self.actor, cfg)
        if cfg.compile_training and env.device.type == 'cuda':
            self.model_loss = torch.compile(self.model_loss, fullgraph=True)
            self.imagined_step = torch.compile(self.imagined_step, fullgraph=True)
        self.imagined_return = imagined_return
        self.critic_loss = partial(critic_minibatch_loss, self.critic)
        if cfg.compile_policy_losses and env.device.type == 'cuda':
            self.imagined_return = torch.compile(self.imagined_return, fullgraph=True)
            self.critic_loss = torch.compile(self.critic_loss, fullgraph=True)
        fused = True if cfg.fused_adam and env.device.type == 'cuda' else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_learning_rate,
                                                betas=(0.9, 0.95), fused=fused)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, fused=fused)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate, fused=fused)
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
        use_graph = self.cfg.cuda_graph_model and self.env.device.type == 'cuda'
        for _ in range(self.cfg.model_updates):
            window = self.replay.sample_windows(self.cfg.model_batch, self.cfg.model_window,
                                                complete_frames=self.cfg.official_replay_ratio is not None)
            if use_graph:
                if self.graphed_model_backward is None:
                    with torch.cuda.device(self.env.device):
                        self.graphed_model_backward = GraphedModelBackward(
                            self.model_loss, self.model_optimizer, window, self.env.device)
                loss = self.graphed_model_backward(window)
            else:
                loss = self.model_loss(window)
            optimize(loss, self.model_optimizer, self.model.parameters(), self.cfg.grad_norm, backward=not use_graph)
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def behavior_loss(self, seed):
        if self.driver is None:
            raise ValueError('Behavior guidance requires a data driver')
        observation = self.env.task.observe(unpack(seed['state']), seed['previous_action'], context_from(seed))
        with torch.no_grad():
            _, target = self.env.controller.target(self.driver(observation))
        _, action = self.env.controller.target(self.actor(observation))
        return (action - target).square().mean()

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
        start = perf_counter()
        seed = self.replay.sample_starts(cfg.imagination_batch)
        with frozen(self.model, self.critic):
            objective, data, survival = imagined_rollout(
                self.env, self.model, self.actor, self.critic, seed, cfg,
                step=self.imagined_step, returns=self.imagined_return)
            behavior = self.behavior_loss(seed) if cfg.behavior_weight else objective.new_zeros(())
            grad = optimize(-objective + cfg.behavior_weight * behavior, self.actor_optimizer,
                            self.actor.parameters(), cfg.grad_norm, cfg.actor_grad_value)
        stats = {'imagined_return': objective.item(), 'imagined_survival': survival.item(),
                 'actor_grad_norm': grad, 'behavior_loss': behavior.item(), 'critic_loss': 0.0}
        critic_start = perf_counter()
        stats['actor_seconds'] = critic_start - start
        stats['critic_seconds'] = 0.0
        if cfg.bootstrap_weight == 0:
            return stats
        targets = lambda_returns(self.critic, data, cfg.gamma, cfg.gae_lambda).flatten()
        x = data['inputs'][:, :-1].reshape(-1, data['inputs'].shape[-1])
        losses = []
        ids = None
        for _ in range(cfg.critic_epochs):
            if ids is None or cfg.official_replay_ratio is None:
                ids = torch.randperm(x.shape[0], device=x.device).tensor_split(cfg.critic_minibatches)
            for minibatch in ids:
                loss = self.critic_loss(x, targets, minibatch)
                optimize(loss, self.critic_optimizer, self.critic.parameters(), cfg.grad_norm)
                losses.append(loss.item())
        stats['critic_loss'] = sum(losses) / len(losses)
        stats['critic_seconds'] = perf_counter() - critic_start
        return stats

    def save(self, output, iteration):
        checkpoint = {'schema': 'rl-go2.supermimic/v2', 'iteration': iteration, 'trainer': asdict(self.cfg),
                      'actor': self.actor.state_dict(), 'world_model': self.model.state_dict(),
                      'critic': self.critic.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(),
                      'model_optimizer': self.model_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict()}
        torch.save(checkpoint, output / f'model_{iteration}.pt')
        # Export exactly the same normalized actor used during data collection.
        torch.jit.script(self.actor).save(str(output / 'policy.pt'))


@contextmanager
def cuda_matmul_tf32(enabled):
    if not enabled:
        yield
        return
    matmul = torch.backends.cuda.matmul
    # PyTorch < 2.9 exposes the legacy flag instead.
    setting, value = ('fp32_precision', 'tf32') if hasattr(matmul, 'fp32_precision') else ('allow_tf32', True)
    previous = getattr(matmul, setting)
    setattr(matmul, setting, value)
    try:
        yield
    finally:
        setattr(matmul, setting, previous)


def train(config, output):
    from rl_go2.trainers import save_run_spec
    cfg = TrainConfig(**config['trainer'])
    cfg.validate()
    torch.manual_seed(config['seed'])
    # Metadata creation rejects existing directories before creating a simulator.
    save_run_spec(config, output)
    output = Path(output)
    env = QuadrupedEnv(config, record_transitions=True)
    with closing(env), cuda_matmul_tf32(cfg.tf32 and env.device.type == 'cuda'):
        trainer = Trainer(env, cfg)
        if cfg.official_replay_ratio is None:
            while trainer.replay.size < cfg.model_window:
                trainer.collect()
        replay_budget = env.num_envs * cfg.collect_steps * (cfg.official_replay_ratio or 0)
        update_cost = cfg.model_updates * cfg.model_batch * cfg.model_window
        policy_updates = 0
        with (output / 'metrics.jsonl').open('w') as log:
            for iteration in range(1, cfg.iterations + 1):
                start = perf_counter()
                stats = trainer.collect()
                stats.update(collect_seconds=perf_counter() - start, model_seconds=0.0,
                             actor_seconds=0.0, critic_seconds=0.0)
                updates = 1
                if cfg.official_replay_ratio is not None:
                    step = iteration - 1
                    # Official scheduling starts with a tick at step zero. Subsequent
                    # ticks track the accumulated update budget, even before replay is ready.
                    updates = (1 if step == 0 else
                               step * replay_budget // update_cost - (step - 1) * replay_budget // update_cost)
                    if trainer.replay.size < cfg.model_window:
                        updates = 0
                for _ in range(updates):
                    model_start = perf_counter()
                    stats['model_loss'] = trainer.fit_model()
                    stats['model_seconds'] += perf_counter() - model_start
                    if iteration > cfg.warmup_iterations:
                        policy = trainer.improve_policy()
                        policy['actor_seconds'] += stats['actor_seconds']
                        policy['critic_seconds'] += stats['critic_seconds']
                        stats.update(policy)
                        policy_updates += 1
                    elif cfg.behavior_updates:
                        actor_start = perf_counter()
                        stats['behavior_loss'] = trainer.warmup_actor()
                        stats['actor_seconds'] += perf_counter() - actor_start
                stats.update(iteration=iteration, environment_steps=env.environment_steps,
                             policy_updates=policy_updates)
                log.write(json.dumps(stats, allow_nan=False) + '\n')
                log.flush()
                if iteration % cfg.log_interval == 0 or iteration == cfg.iterations:
                    print(json.dumps(stats, allow_nan=False), flush=True)
                if iteration % cfg.save_interval == 0 or iteration == cfg.iterations:
                    trainer.save(output, iteration)
