"""SuperMimic-style analytic policy gradients for goal-conditioned Go2 control.

Independent reproduction inspired by Zhengtianjin/go2-supermimic (MIT) and
Liao et al., arXiv:2610.28480. No mjlab code or task-specific assets are required.
"""
from collections.abc import Callable, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from torch import nn

from rl_go2.backends import LinkPointForces, make_backend
from rl_go2.config import FOOT_NAMES, MODEL_PARAMETER_KEYS
from rl_go2.env import QuadrupedEnv
from rl_go2.observations import SensorReadings
from rl_go2.point_push import Go2PushGeometry, PointPushConfig, PointPushScheduler, PushState
from rl_go2.step_delay import StepDelayState
from rl_go2.trainers.model_learning import DynamicsModel, ReplayBuffer, RunningNorm, heading, local_state, mlp, pack, unpack
from rl_go2.trainers.terrain_features import TerrainHeightScan

OFFICIAL_REPLAY_RATIO = 16


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
    tf32: bool = True
    fused_adam: bool = True
    compile_training: bool = True
    cuda_graph: bool = True
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
    condition_on_link_wrenches: bool = False
    init_checkpoint: str | None = None
    world_model_warmup_iterations: int = 0

    @property
    def uses_official_schedule(self):
        return self.warmup_iterations == 0

    def validate(self):
        for key in ('tf32', 'fused_adam', 'compile_training', 'cuda_graph'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(self.condition_on_link_wrenches) is not bool:
            raise ValueError('condition_on_link_wrenches must be a boolean')
        if type(self.world_model_warmup_iterations) is not int or self.world_model_warmup_iterations < 0:
            raise ValueError('world_model_warmup_iterations must be a nonnegative integer')
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
        if self.uses_official_schedule and self.model_window < 2:
            raise ValueError('Official replay requires at least two frames')
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


def optimize(loss, optimizer, parameters, norm, value=None, *, backward=True, step=None):
    parameters = list(parameters)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite SuperMimic loss')
    if backward:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    grad = nn.utils.clip_grad_norm_(parameters, norm, error_if_nonfinite=True)
    if value is not None:
        nn.utils.clip_grad_value_(parameters, value)
    (optimizer.step if step is None else step)()
    return float(grad)


def context_from(record):
    return {key[8:]: value for key, value in record.items() if key.startswith('context/')}


def critic_input(state, previous_action, context, features):
    return torch.cat((local_state(state, heading(state[:, 3:7])), previous_action, features,
                      *(context[key].reshape(state.shape[0], -1).float() for key in sorted(context)
                        if key not in MODEL_PARAMETER_KEYS)), -1)


def imagined_step(env, model, actor, cfg, state, previous_action, context,
                  wrenches_world: torch.Tensor | None = None, *,
                  observation: torch.Tensor | None = None, action_delay_state: StepDelayState | None = None):
    robot = unpack(state)
    features = env.task.model_features(robot, context)
    inputs = None
    if cfg.bootstrap_weight:
        with torch.no_grad():
            inputs = critic_input(state, previous_action, context, features)
    if observation is None:
        observation = env.task.observe(robot, previous_action, context)
    issued_action = env.controller.clip_action(actor(observation))
    applied_action = issued_action
    if action_delay_state is not None:
        applied_action = env.action_delay.select(issued_action, action_delay_state)
    next_state, torque = model(state, applied_action, previous_action, features, wrenches_world)
    next_robot = unpack(next_state)
    torque = torque.clamp(-env.controller.limits, env.controller.limits)
    reward = env.task.reward(next_robot, issued_action, previous_action, torque, context)
    continuation = 1 - env.task.termination_probability(next_robot, context, cfg.height_softness, cfg.upright_softness)
    return next_state, issued_action, reward, continuation, inputs


def imagined_return(rewards, continuations, gamma):
    survival = torch.ones_like(rewards[0])
    objective = torch.zeros_like(survival)
    for t, (reward, continuation) in enumerate(zip(rewards, continuations)):
        objective = objective + gamma**t * survival * reward
        survival = survival * continuation
    return objective, survival


def critic_minibatch_loss(critic, inputs, targets):
    return (critic(inputs) - targets).square().mean()


def imagined_rollout(env, model, actor, critic, seed, cfg, *, step=None, returns=imagined_return,
                     wrench_plan_world: torch.Tensor | None = None,
                     push_scheduler: PointPushScheduler | None = None):
    """Roll out a (batch, horizon, links, 6) world-frame wrench plan if supplied.

    A push scheduler continues the event state stored in the seed. Without
    either a plan or a scheduler, only the initial recorded wrench is applied.
    If enabled, action delay continues the sampled episode's queue and lag.
    """
    if step is None:
        step = partial(imagined_step, env, model, actor, cfg)
    state, previous_action, context = seed['state'], seed['previous_action'], context_from(seed)
    action_delay = env.action_delay
    delay_state = None if action_delay is None else StepDelayState(seed['action_delay_steps'], seed['action_queue'])
    observation_noise = env.observation_noise
    noise_history = {key.removeprefix('sensor_history/'): value for key, value in seed.items()
                     if key.startswith('sensor_history/')}
    initial_wrenches = seed.get('link_wrenches_world')
    zero_wrenches = None if initial_wrenches is None else torch.zeros_like(initial_wrenches)
    push_state = None if push_scheduler is None else PushState.from_replay(seed)
    if wrench_plan_world is not None:
        expected_shape = (state.shape[0], cfg.horizon, model.wrench_links, 6)
        if not model.wrench_links or wrench_plan_world.shape != expected_shape:
            raise ValueError(f'Expected a wrench-conditioned model and a wrench plan of shape {expected_shape}')
    rewards, continuations, inputs = [], [], []
    for index in range(cfg.horizon):
        if wrench_plan_world is not None:
            wrenches_world = wrench_plan_world[:, index]
        elif push_state is not None and index > 0:
            link_quat = push_scheduler.geometry.link_quaternions(unpack(state))
            # Event sampling is exogenous. The moment arm of an ongoing push
            # still depends on the differentiable predicted joint positions.
            push_state = push_scheduler.prepare(push_scheduler.advance(push_state), link_quat.detach())
            wrenches_world = push_scheduler.wrenches(push_state, link_quat)
        else:
            wrenches_world = initial_wrenches if index == 0 else zero_wrenches
        step_args = (state, previous_action, context)
        if wrenches_world is not None:
            step_args += (wrenches_world,)
        step_options = {}
        if delay_state is not None:
            step_options['action_delay_state'] = delay_state
        if observation_noise is not None:
            if index == 0:
                observation = seed['actor_observation']
            else:
                robot = unpack(state)
                readings, noise_history = observation_noise.apply(SensorReadings.from_state(robot), noise_history)
                observation = env.task.observe(robot, previous_action, context, readings=readings)
            step_options['observation'] = observation
        state, action, reward, continuation, critic_obs = step(*step_args, **step_options)
        if delay_state is not None:
            # Keep queued commands differentiable until the dynamics apply them.
            delay_state = action_delay.advance(action, delay_state)
        rewards.append(reward)
        continuations.append(continuation)
        if cfg.bootstrap_weight:
            inputs.append(critic_obs)
        context = env.advance_context(context, unpack(state))
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


class GraphedBackward:
    def __init__(self, loss_fn, optimizer, window, device):
        self.window = {key: value.clone() for key, value in window.items()}
        self.stream = torch.cuda.Stream(device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.optimizer = optimizer
        self.step_graph = None
        caller = torch.cuda.current_stream(device)
        self.stream.wait_stream(caller)
        # Keep AccumulateGrad on the same stream during warmup and capture.
        with torch.random.fork_rng(devices=[device]), torch.cuda.stream(self.stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss_fn(self.window)[0].backward()
            # Allocate gradients inside the graph so replay overwrites them.
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph, stream=self.stream):
                loss, self.outputs = loss_fn(self.window)
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
        return self.loss, self.outputs

    def step(self):
        if not self.optimizer.state:
            # Initialize Adam state once on a real, validated minibatch.
            self.optimizer.step()
            return
        caller = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            if self.step_graph is None:
                self.step_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.step_graph, stream=self.stream):
                    self.optimizer.step()
            self.step_graph.replay()
        caller.wait_stream(self.stream)


class Trainer:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        required = ('model_features', 'advance_context', 'termination_probability')
        if any(not callable(getattr(env.task, key, None)) for key in required):
            raise ValueError(f'Task must implement model-learning hooks: {required}')
        if not env.record_transitions:
            raise ValueError('SuperMimic requires record_transitions=True')
        self.wrench_link_indices: tuple[int, ...] = ()
        self.other_link_indices: tuple[int, ...] = ()
        if cfg.condition_on_link_wrenches:
            names = (env.config['robot']['base_link'], *(f'{foot}_calf' for foot in FOOT_NAMES))
            self.wrench_link_indices = tuple(env.backend.link_names.index(name) for name in names)
            self.other_link_indices = tuple(index for index in range(len(env.backend.link_names))
                                            if index not in self.wrench_link_indices)
        context = env.model_context()
        features = env.task.model_features(env.state, context)
        state = pack(env.state)
        ci = critic_input(state, env.previous_action, context, features)
        self.actor = Actor(env.observe().shape[-1], env.num_actions, cfg.actor_hidden,
                           env.config['robot']['action_clip']).to(env.device)
        scene = env.config['scene']
        terrain_scan = TerrainHeightScan(scene) if scene.get('height_maps') or scene.get('terrain_patches') or any(
            entity['type'] == 'box' for entity in scene['entities']
        ) else None
        self.model = DynamicsModel(features.shape[-1], cfg.model_hidden,
                                   wrench_links=len(self.wrench_link_indices), terrain_scan=terrain_scan).to(env.device)
        self.critic = Critic(ci.shape[-1], cfg.critic_hidden).to(env.device)
        push_spec = PointPushConfig.from_config(env.config)
        self.push_scheduler: PointPushScheduler | None = None
        self.push_state: PushState | None = None
        if push_spec is not None:
            if not self.wrench_link_indices or env.config['task']['name'] != 'flat_walk':
                raise ValueError('point_push requires flat_walk and condition_on_link_wrenches')
            geometry = Go2PushGeometry(env.config, FOOT_NAMES, env.device)
            self.push_scheduler = PointPushScheduler(push_spec, env.dt, geometry)
            self.push_state = self.push_scheduler.reset(env.state.position)
            self.push_link_indices = torch.tensor(self.wrench_link_indices, device=env.device)
        self.model_loss = self.model.loss
        self.graphed_model_backward = None
        self.graphed_actor_backward = None
        self.graphed_critic_backward = None
        self.use_cuda_graph = cfg.cuda_graph and env.device.type == 'cuda'
        self.imagined_step = partial(imagined_step, env, self.model, self.actor, cfg)
        self.imagined_return = imagined_return
        self.critic_loss = partial(critic_minibatch_loss, self.critic)
        if cfg.compile_training and env.device.type == 'cuda':
            self.model_loss = torch.compile(self.model_loss, fullgraph=True)
            self.imagined_step = torch.compile(self.imagined_step, fullgraph=True)
            self.imagined_return = torch.compile(self.imagined_return, fullgraph=True)
            self.critic_loss = torch.compile(self.critic_loss, fullgraph=True)
        fused = True if cfg.fused_adam and env.device.type == 'cuda' else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_learning_rate,
                                                betas=(0.9, 0.95), fused=fused, capturable=self.use_cuda_graph)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, fused=fused)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate,
                                                 fused=fused, capturable=self.use_cuda_graph)
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

    def _step_for_collection(self, action: torch.Tensor, loads: LinkPointForces | None) -> tuple:
        if loads is None:
            return self.env.step(action)
        loads.validate(self.env.num_envs, len(self.env.backend.link_names))
        if loads.force_world_n[:, self.other_link_indices].any():
            raise ValueError('World model point loads must target the base or one of the four calf links')
        return self.env.step(action, link_point_forces=loads)

    @torch.no_grad()
    def collect(
        self, *, point_force_plan: Sequence[LinkPointForces | None] | None = None,
        update_actor_norm: bool = True,
    ) -> dict[str, float]:
        """Collect transitions with an optional point-load batch for each control step."""
        if point_force_plan is not None and (
            self.push_scheduler is not None or not self.wrench_link_indices
            or len(point_force_plan) != self.cfg.collect_steps
        ):
            raise ValueError('A point-force plan requires scheduler-free wrench conditioning and one entry per step')
        if self.env.foot_friction is not None:
            self.env.resample_foot_friction()
        rewards, dones, driver_rewards = [], [], []
        push_counts = []
        task_metrics = {}
        for index in range(self.cfg.collect_steps):
            obs = self.env.observe()
            sensor_record = {}
            if self.env.observation_noise is not None:
                sensor_record['actor_observation'] = obs
                sensor_record.update({f'sensor_history/{key}': value for key, value in self.env.noise_history.items()})
            if update_actor_norm:
                self.actor.norm.update(obs)
            action = self.actor(obs)
            if self.driver is not None:
                driver_obs = obs if self.env.observation_noise is None else self.env.observe(clean=True)
                driver_action = self.driver(driver_obs)
                if driver_action.shape != action.shape or not torch.isfinite(driver_action).all():
                    raise ValueError('Data driver must return finite [num_envs, num_actions] actions')
                action = torch.where(self.driver_mask[:, None], driver_action, action)
            action = action + self.cfg.action_noise * torch.randn_like(action)
            loads = None if point_force_plan is None else point_force_plan[index]
            if self.push_scheduler is None:
                _, reward, done, extras = self._step_for_collection(action, loads)
                push_record = {}
            else:
                (_, reward, done, extras), push_record = self._collect_push_step(action)
                push_counts.append(extras['transition'].link_wrenches_world[..., :3].ne(0).any(dim=-1).sum(dim=-1))
            transition = extras['transition']
            features = self.env.task.model_features(transition.state, transition.context)
            record = {'state': pack(transition.state), 'next_state': pack(transition.next_state),
                      'action': transition.action, 'previous_action': transition.previous_action,
                      'torque': transition.torque, 'terminated': transition.terminated,
                      'truncated': transition.truncated, 'features': features,
                      'reward': reward, 'is_driver': self.driver_mask.clone()}
            if transition.action_delay_state is not None:
                record['action_delay_steps'] = transition.action_delay_state.steps
                record['action_queue'] = transition.action_delay_state.history
            if self.wrench_link_indices:
                wrenches = transition.link_wrenches_world
                shape = (self.env.num_envs, len(self.wrench_link_indices), 6)
                record['link_wrenches_world'] = (
                    features.new_zeros(shape) if wrenches is None else wrenches[:, self.wrench_link_indices]
                )
            record.update({f'context/{key}': value for key, value in transition.context.items()})
            record.update(push_record)
            record.update(sensor_record)
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
        if push_counts:
            counts = torch.stack(push_counts)
            result['push_active_fraction'] = (counts > 0).float().mean().item()
            result['push_overlap_fraction'] = (counts > 1).float().mean().item()
        return result

    def _collect_push_step(self, action: torch.Tensor) -> tuple[tuple, dict[str, torch.Tensor]]:
        scheduler = self.push_scheduler
        link_quat = self.env.backend.read_link_quaternions()[:, self.wrench_link_indices]
        prepared = scheduler.prepare(self.push_state, link_quat)
        loads = scheduler.loads(prepared)
        shape = (self.env.num_envs, len(self.env.backend.link_names), 3)
        points = loads.point_local_m.new_zeros(shape).index_copy(1, self.push_link_indices, loads.point_local_m)
        forces = loads.force_world_n.new_zeros(shape).index_copy(1, self.push_link_indices, loads.force_world_n)
        result = self.env.step(action, link_point_forces=LinkPointForces(points, forces))
        reset_ids = result[3].get('final_env_ids')
        advanced = scheduler.advance(prepared)
        if reset_ids is not None:
            reset_state = scheduler.reset(self.env.state.position[reset_ids])
            advanced = PushState(*(value.index_copy(0, reset_ids, reset)
                                   for value, reset in zip(advanced, reset_state)))
        self.push_state = advanced
        return result, prepared.replay_fields()

    def fit_model(self):
        losses = []
        use_graph = self.use_cuda_graph
        for _ in range(self.cfg.model_updates):
            window = self.replay.sample_windows(self.cfg.model_batch, self.cfg.model_window,
                                                complete_frames=self.cfg.uses_official_schedule)
            if use_graph:
                if self.graphed_model_backward is None:
                    with torch.cuda.device(self.env.device):
                        self.graphed_model_backward = GraphedBackward(
                            lambda batch: (self.model_loss(batch), ()), self.model_optimizer, window, self.env.device)
                loss, _ = self.graphed_model_backward(window)
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
        _, action = self.env.controller.target(self.actor(seed.get('actor_observation', observation)))
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

    def actor_loss(self, seed):
        cfg = self.cfg
        objective, data, survival = imagined_rollout(
            self.env, self.model, self.actor, self.critic, seed, cfg,
            step=self.imagined_step, returns=self.imagined_return, push_scheduler=self.push_scheduler)
        behavior = self.behavior_loss(seed) if cfg.behavior_weight else objective.new_zeros(())
        return -objective + cfg.behavior_weight * behavior, (objective.detach(), data, survival, behavior.detach())

    def improve_policy(self):
        cfg = self.cfg
        start = perf_counter()
        seed = self.replay.sample_starts(cfg.imagination_batch)
        with frozen(self.model, self.critic):
            if self.use_cuda_graph:
                if self.graphed_actor_backward is None:
                    with torch.cuda.device(self.env.device):
                        self.graphed_actor_backward = GraphedBackward(
                            self.actor_loss, self.actor_optimizer, seed, self.env.device)
                loss, outputs = self.graphed_actor_backward(seed)
            else:
                loss, outputs = self.actor_loss(seed)
            objective, data, survival, behavior = outputs
            grad = optimize(loss, self.actor_optimizer, self.actor.parameters(), cfg.grad_norm,
                            cfg.actor_grad_value, backward=not self.use_cuda_graph,
                            step=self.graphed_actor_backward.step if self.use_cuda_graph else None)
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
        graph_critic = self.use_cuda_graph and x.shape[0] % cfg.critic_minibatches == 0
        for _ in range(cfg.critic_epochs):
            if ids is None or not cfg.uses_official_schedule:
                ids = torch.randperm(x.shape[0], device=x.device).tensor_split(cfg.critic_minibatches)
            for minibatch in ids:
                batch = {'inputs': x[minibatch], 'targets': targets[minibatch]}
                if graph_critic:
                    if self.graphed_critic_backward is None:
                        with torch.cuda.device(self.env.device):
                            self.graphed_critic_backward = GraphedBackward(
                                lambda data: (self.critic_loss(**data), ()),
                                self.critic_optimizer, batch, self.env.device)
                    loss, _ = self.graphed_critic_backward(batch)
                else:
                    loss = self.critic_loss(**batch)
                optimize(loss, self.critic_optimizer, self.critic.parameters(), cfg.grad_norm,
                         backward=not graph_critic, step=self.graphed_critic_backward.step if graph_critic else None)
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
    from rl_go2.trainers.cuda_startup import reserve_startup_memory

    cfg = TrainConfig(**config['trainer'])
    cfg.validate()
    courses = config.get('courses')
    if courses and cfg.iterations > courses[-1]['end_iteration']:
        raise ValueError('Training iterations exceed the total course budget')
    torch.manual_seed(config['seed'])
    reserve_gib = 0
    if (config['backend'], config['device']) == ('genesis', 'cuda'):
        reserve_gib = config.get('startup_gpu_reserve_gib', 0)
    with reserve_startup_memory(reserve_gib) as release_startup_memory:
        save_run_spec(config, output)
        output = Path(output)
        with (output / 'metrics.jsonl').open('w') as log:
            if courses:
                _train_courses(config, cfg, output, log, release_startup_memory)
            else:
                _train_stage(config, cfg, output, log, 1, cfg.iterations,
                             release_startup_memory=release_startup_memory)


def _train_courses(config, cfg: TrainConfig, output: Path, log,
                   release_startup_memory: Callable[[], None] | None) -> None:
    first = 1
    progress = (0, 0)
    previous = None
    for index, course in enumerate(config['courses'], start=1):
        last = min(course['end_iteration'], cfg.iterations)
        print(f"Course {index}: {course['name']} (iterations {first}–{last})", flush=True)
        progress, previous = _train_stage(
            {**config, **course['settings']}, cfg, output, log, first, last,
            progress=progress, previous=previous, course_index=index,
            release_startup_memory=release_startup_memory,
        )
        release_startup_memory = None
        if last == cfg.iterations:
            break
        first = last + 1


def _train_stage(config, cfg: TrainConfig, output: Path, log, first: int, last: int, *,
                 progress=(0, 0), previous=None, course_index: int | None = None,
                 release_startup_memory: Callable[[], None] | None = None):
    from rl_go2.trainers.supermimic_courses import continue_from_course, input_layouts
    from rl_go2.trainers.supermimic_init import initialize_weights

    environment_steps, policy_updates = progress
    backend = None
    if release_startup_memory is not None:
        backend = make_backend(config, before_genesis_build=release_startup_memory)
    env = QuadrupedEnv(config, record_transitions=True, backend=backend)
    with closing(env), cuda_matmul_tf32(cfg.tf32 and env.device.type == 'cuda'):
        env.environment_steps = environment_steps
        trainer = Trainer(env, cfg)
        if previous is not None:
            continue_from_course(trainer, output / f'model_{first - 1}.pt', previous)
        elif cfg.init_checkpoint is not None:
            initialize_weights(trainer, Path(cfg.init_checkpoint))
        if not cfg.uses_official_schedule:
            while trainer.replay.size < cfg.model_window:
                trainer.collect(update_actor_norm=cfg.world_model_warmup_iterations == 0)
        replay_budget = env.num_envs * cfg.collect_steps * OFFICIAL_REPLAY_RATIO
        update_cost = cfg.model_updates * cfg.model_batch * cfg.model_window
        for iteration in range(first, last + 1):
            start = perf_counter()
            model_only = iteration <= cfg.world_model_warmup_iterations
            stats = trainer.collect(update_actor_norm=not model_only)
            stats.update(collect_seconds=perf_counter() - start, model_seconds=0.0,
                         actor_seconds=0.0, critic_seconds=0.0)
            updates = 1
            if cfg.uses_official_schedule:
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
                if model_only:
                    continue
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
            if course_index is not None:
                stats['course_stage'] = course_index
            log.write(json.dumps(stats, allow_nan=False) + '\n')
            log.flush()
            if iteration % cfg.log_interval == 0 or iteration == last:
                print(json.dumps(stats, allow_nan=False), flush=True)
            if iteration % cfg.save_interval == 0 or iteration == last:
                trainer.save(output, iteration)

        layouts = input_layouts(trainer) if last < cfg.iterations else None
        return (env.environment_steps, policy_updates), layouts
