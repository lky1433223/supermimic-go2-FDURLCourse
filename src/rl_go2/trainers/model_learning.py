"""Tensor replay and differentiable dynamics, independent of the physics backend.

State tensors retain world pose. The MLP sees height and heading-local quantities,
then predicts a local pose/velocity increment plus the final physics-substep torque.
"""
from dataclasses import fields

import torch
from torch import nn
from torch.nn import functional as F

from rl_go2.state import RobotState, rotate_inverse

STATE_DIM = 37


def pack(state):
    return torch.cat([getattr(state, f.name) for f in fields(RobotState)], -1)


def unpack(state):
    return RobotState(*state.split([3, 4, 3, 3, 12, 12], dim=-1))


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1)), -1)


def canonical_quat(q):
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    q = torch.where(q.norm(dim=-1, keepdim=True) > 1e-6, F.normalize(q, dim=-1), identity)
    return torch.where(q[..., :1] < 0, -q, q)


def heading(q):
    w, x, y, z = canonical_quat(q).unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack(((yaw / 2).cos(), zero, zero, (yaw / 2).sin()), -1)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def local_state(state, frame):
    """Keep orientation signs consistent before computing residual targets."""
    s = unpack(state)
    q = canonical_quat(quat_mul(conjugate(frame), s.quaternion))
    return torch.cat((s.position[:, 2:3], q, rotate_inverse(frame, s.linear_velocity),
                      rotate_inverse(frame, s.angular_velocity), s.joint_position, s.joint_velocity), -1)


def mlp(inputs, hidden, outputs):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(inputs, width), nn.SiLU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel())
        n = x.shape[0]
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.var.copy_((self.var * self.count + x.var(0, unbiased=False) * n
                        + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / self.scale()).clamp(-10, 10)

    def scale(self):
        return self.var.clamp(min=1e-6).sqrt()


class DynamicsModel(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.input_norm = RunningNorm(35 + 12 + features)
        self.output_norm = RunningNorm(STATE_DIM + 12)
        self.net = mlp(35 + 12 + features, hidden, STATE_DIM + 12)
        # Begin near the empirical average increment after statistics warmup.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def inputs(self, state, action, previous_action, features):
        frame = heading(state[:, 3:7])
        # RobotState is Markov for the external PD controller (there is no actuator
        # delay). Feeding previous_action creates a noncausal shortcut on gait data.
        return torch.cat((local_state(state, frame), action, features), -1)

    def targets(self, state, next_state, torque):
        frame = heading(state[:, 3:7])
        before, after = local_state(state, frame), local_state(next_state, frame)
        # q and -q describe the same rotation. Align near the current orientation.
        q = after[:, 1:5]
        q = torch.where((q * before[:, 1:5]).sum(-1, keepdim=True) < 0, -q, q)
        after = torch.cat((after[:, :1], q, after[:, 5:]), -1)
        displacement = rotate_inverse(frame, next_state[:, :3] - state[:, :3])
        return torch.cat((displacement, after[:, 1:] - before[:, 1:], torque), -1)

    @torch.no_grad()
    def update_stats(self, record):
        self.input_norm.update(self.inputs(record['state'], record['action'], record['previous_action'], record['features']))
        self.output_norm.update(self.targets(record['state'], record['next_state'], record['torque']))

    def forward(self, state, action, previous_action, features):
        raw = self.net(self.input_norm(self.inputs(state, action, previous_action, features)))
        delta = raw * self.output_norm.scale() + self.output_norm.mean
        frame = heading(state[:, 3:7])
        before = local_state(state, frame)
        after = before[:, 1:] + delta[:, 3:STATE_DIM]
        q = canonical_quat(quat_mul(frame, canonical_quat(after[:, :4])))
        rotate = lambda v: rotate_inverse(conjugate(frame), v)
        next_state = torch.cat((state[:, :3] + rotate(delta[:, :3]), q,
                                rotate(after[:, 4:7]), rotate(after[:, 7:10]), after[:, 10:]), -1)
        return next_state, delta[:, STATE_DIM:]

    def loss(self, window):
        state = window['state'][:, 0]
        total = state.new_zeros(())
        for j in range(window['action'].shape[1]):
            predicted, torque = self(state, window['action'][:, j], window['previous_action'][:, j], window['features'][:, j])
            # Compare full autoregressive state, not only the current step's increment.
            error = self.targets(window['next_state'][:, j], predicted,
                                 torque - window['torque'][:, j]) / self.output_norm.scale()
            total = total + (error.square().mean(-1) * window['valid'][:, j]).sum()
            state = predicted
        return total / window['valid'].sum().clamp(min=1)


class ReplayBuffer:
    def __init__(self, length, num_envs, device):
        self.length, self.num_envs, self.device = length, num_envs, device
        self.data = {}
        self.size = self.ptr = 0

    def add(self, record):
        if not self.data:
            self.data = {k: torch.empty((self.length, *v.shape), device=self.device, dtype=v.dtype)
                         for k, v in record.items()}
        if record.keys() != self.data.keys():
            raise ValueError('Replay schema changed during training')
        for key, value in record.items():
            self.data[key][self.ptr].copy_(value.detach())
        self.ptr = (self.ptr + 1) % self.length
        self.size = min(self.size + 1, self.length)

    def _slots(self, logical):
        return (self.ptr - self.size + logical) % self.length

    def sample_starts(self, batch):
        t = self._slots(torch.randint(self.size, (batch,), device=self.device))
        env = torch.randint(self.num_envs, (batch,), device=self.device)
        return {k: v[t, env] for k, v in self.data.items()}

    def sample_windows(self, batch, steps):
        if self.size < steps:
            raise ValueError('Not enough replay steps')
        t = torch.randint(self.size - steps + 1, (batch, 1), device=self.device)
        t = self._slots(t + torch.arange(steps, device=self.device))
        env = torch.randint(self.num_envs, (batch, 1), device=self.device)
        result = {k: v[t, env] for k, v in self.data.items()}
        # The final transition itself is valid. Only following transitions cross reset.
        done = (result['terminated'] | result['truncated']).float()
        result['valid'] = torch.cat((torch.ones_like(done[:, :1]), (1 - done[:, :-1]).cumprod(1)), 1)
        return result
