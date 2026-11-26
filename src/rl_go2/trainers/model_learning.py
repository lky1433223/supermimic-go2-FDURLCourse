"""Tensor replay and differentiable dynamics, independent of the physics backend.

State tensors retain world pose. The MLP sees height and heading-local quantities,
then predicts a local pose/velocity increment plus the final physics-substep torque.
"""
from dataclasses import fields

import torch
from torch import nn
from torch.nn import functional as F

from rl_go2.state import RobotState, quat_mul, rotate_inverse
from rl_go2.terrain_features import TerrainHeightScan

STATE_DIM = 37


def pack(state):
    return torch.cat([getattr(state, f.name) for f in fields(RobotState)], -1)


def unpack(state):
    return RobotState(*state.split([3, 4, 3, 3, 12, 12], dim=-1))


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
    """Predict the next robot state from the current state, action and physics.

    Flat Go2 input layout before normalization (zero-based indices, slices
    exclude their stop index). The heading frame shares world Z and rotates
    its horizontal axes by the current base yaw.

    - ``0``                      Base-link origin height in world coordinates (m).
    - ``1:5``                    Base orientation relative to heading (unit wxyz).
    - ``5:8``                    Base-origin linear velocity (heading xyz, m/s).
    - ``8:11``                   Base angular velocity (heading xyz, rad/s).
    - ``11:23``                  Joint positions (rad).
    - ``23:35``                  Joint velocities (rad/s).
    - ``35:47``                  Current applied action after clipping (unitless).
    - ``47:59``                  Foot friction, ordered FR, FL, RR, RL. Each foot
                                 contributes sliding (unitless), torsional (m),
                                 and rolling (m), in that order.
    - ``59``                     Base-link mass (kg).
    - ``60:63``                  Base COM offset from nominal (base-link xyz, m).
    - ``63:69``, ``69:75``, ``75:81``, ``81:87``, ``87:93``
                                 Point-load features for base, FR_calf, FL_calf,
                                 RR_calf and RL_calf, respectively. Each block
                                 is [Fx, Fy, Fz, Mx, My, Mz] in heading axes
                                 (N and N·m), with moment about the link origin.
                                 Unloaded links have zero-valued blocks.

    Joint blocks follow ``robot.joint_names``. The Go2 default is FR, FL, RR,
    RL, each with hip, thigh and calf in that order. The controller converts
    the input action to an angle target using the default angles and action
    scale. The indices above show all randomization features enabled. Disabled
    physics groups are omitted, so subsequent slices shift. The final 30
    values shown above are included only when link-wrench conditioning is
    enabled. On rough ground, 54 relative heights follow those inputs.
    """

    def __init__(self, features: int, hidden: tuple[int, ...], *, wrench_links: int = 0,
                 terrain_scan: TerrainHeightScan | None = None, predict_contacts: bool = False) -> None:
        super().__init__()
        self.wrench_links = wrench_links
        base_width = 35 + 12 + features
        self.wrench_end = base_width + 6 * wrench_links
        self.terrain_scan = terrain_scan
        terrain_width = terrain_scan.size if terrain_scan is not None else 0
        self.input_norm = RunningNorm(base_width)
        # Newly introduced load features have no samples in a flat-policy checkpoint.
        self.wrench_norm = RunningNorm(6 * wrench_links) if wrench_links else None
        self.terrain_norm = RunningNorm(terrain_width) if terrain_width else None
        self.output_norm = RunningNorm(STATE_DIM + 12)
        self.net = mlp(self.wrench_end + terrain_width, hidden, STATE_DIM + 12)
        # Begin near the empirical average increment after statistics warmup.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.contact_head = nn.Linear(self.net[-1].in_features, 5) if predict_contacts else None
        if self.contact_head is not None:
            nn.init.zeros_(self.contact_head.weight)
            nn.init.zeros_(self.contact_head.bias)
            # Unsafe contact is rare. Avoid making most imagined steps terminal at initialization.
            with torch.no_grad():
                self.contact_head.bias[-1] = -4.0

    def _heading_features(
        self, features: torch.Tensor, frame: torch.Tensor, wrenches_world: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.wrench_links:
            return features
        shape = (features.shape[0], self.wrench_links, 6)
        if wrenches_world is None or wrenches_world.shape != shape:
            raise ValueError(f"Expected world-frame link wrenches with shape {shape}")
        # Rotate force and moment about each link origin into the same frame as the state.
        vectors = wrenches_world.reshape(shape[0], self.wrench_links, 2, 3)
        local = rotate_inverse(frame[:, None, None, :], vectors)
        return torch.cat((features, local.flatten(1)), dim=-1)

    def _assemble_inputs(self, state: torch.Tensor, local: torch.Tensor, action: torch.Tensor,
                         frame: torch.Tensor, features: torch.Tensor,
                         wrenches_world: torch.Tensor | None) -> torch.Tensor:
        parts = [local, action, self._heading_features(features, frame, wrenches_world)]
        if self.terrain_scan is not None:
            parts.append(self.terrain_scan(state[:, :3], conjugate(frame)))
        return torch.cat(parts, dim=-1)

    def inputs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        features: torch.Tensor,
        wrenches_world: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the network input in the heading frame of the supplied state."""
        frame = heading(state[:, 3:7])
        # RobotState is Markov for the external PD controller (there is no actuator
        # delay). Feeding previous_action creates a noncausal shortcut on gait data.
        return self._assemble_inputs(state, local_state(state, frame), action, frame, features, wrenches_world)

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
        inputs = self.inputs(record['state'], record['action'], record['previous_action'],
                             record['features'], record.get('link_wrenches_world'))
        base_width = self.input_norm.mean.numel()
        self.input_norm.update(inputs[:, :base_width])
        if self.wrench_norm is not None:
            self.wrench_norm.update(inputs[:, base_width:self.wrench_end])
        if self.terrain_norm is not None:
            self.terrain_norm.update(inputs[:, self.wrench_end:])
        self.output_norm.update(self.targets(record['state'], record['next_state'], record['torque']))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        features: torch.Tensor,
        wrenches_world: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next state and motor torque for optional per-link force/moment (N, N·m)."""
        next_state, torque, _ = self.predict_transition(state, action, features, wrenches_world)
        return next_state, torque

    def predict_transition(
        self, state: torch.Tensor, action: torch.Tensor, features: torch.Tensor,
        wrenches_world: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Also predict foot contact and prohibited-contact logits when enabled."""
        frame = heading(state[:, 3:7])
        before = local_state(state, frame)
        inputs = self._assemble_inputs(state, before, action, frame, features, wrenches_world)
        base_width = self.input_norm.mean.numel()
        normalized = [self.input_norm(inputs[:, :base_width])]
        if self.wrench_norm is not None:
            normalized.append(self.wrench_norm(inputs[:, base_width:self.wrench_end]))
        if self.terrain_norm is not None:
            normalized.append(self.terrain_norm(inputs[:, self.wrench_end:]))
        hidden = normalized[0] if len(normalized) == 1 else torch.cat(normalized, dim=-1)
        contact_logits = None
        if self.contact_head is not None:
            for index in range(len(self.net) - 1):
                hidden = self.net[index](hidden)
            contact_logits = self.contact_head(hidden)
            raw = self.net[-1](hidden)
        else:
            raw = self.net(hidden)
        delta = raw * self.output_norm.scale() + self.output_norm.mean
        after = before[:, 1:] + delta[:, 3:STATE_DIM]
        q = canonical_quat(quat_mul(frame, canonical_quat(after[:, :4])))
        rotate = lambda v: rotate_inverse(conjugate(frame), v)
        next_state = torch.cat((state[:, :3] + rotate(delta[:, :3]), q,
                                rotate(after[:, 4:7]), rotate(after[:, 7:10]), after[:, 10:]), -1)
        return next_state, delta[:, STATE_DIM:], contact_logits

    def loss(self, window, *, error_trace: list[tuple[torch.Tensor, torch.Tensor]] | None = None):
        state = window['state'][:, 0]
        wrenches = window.get('link_wrenches_world')
        total = state.new_zeros(())
        for j in range(window['action'].shape[1]):
            wrenches_world = None if wrenches is None else wrenches[:, j]
            predicted, torque = self(
                state, window['action'][:, j], window['previous_action'][:, j], window['features'][:, j], wrenches_world
            )
            # Compare full autoregressive state, not only the current step's increment.
            raw_error = self.targets(window['next_state'][:, j], predicted, torque - window['torque'][:, j])
            error = raw_error / self.output_norm.scale()
            if error_trace is not None:
                error_trace.append((raw_error.detach(), error.detach()))
            total = total + (error.square().mean(-1) * window['valid'][:, j]).sum()
            state = predicted
        loss = total / window['valid'].sum().clamp(min=1)
        if self.contact_head is not None:
            _, _, logits = self.predict_transition(window['contact/state'], window['contact/action'],
                                                   window['contact/features'], window.get('contact/link_wrenches_world'))
            assert logits is not None
            loss = loss + F.binary_cross_entropy_with_logits(logits, window['contact/contact_labels'])
        return loss


class ReplayUnavailable(ValueError):
    """No eligible data are currently available for the requested sampling mode."""


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

    def _sample_transitions(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one valid frame per sample, replacing only draws flagged invalid."""
        t = self._slots(torch.randint(self.size, (batch,), device=self.device))
        env = torch.randint(self.num_envs, (batch,), device=self.device)
        invalid = self.data.get('invalid')
        if invalid is None:
            return t, env
        replace = invalid[t, env].nonzero(as_tuple=True)[0]
        if not replace.numel():
            return t, env
        slots = self._slots(torch.arange(self.size, device=self.device))
        valid = (~invalid[slots]).flatten().float()
        if not valid.any():
            raise ReplayUnavailable('No valid replay transitions')
        draws = torch.multinomial(valid, replace.numel(), replacement=True)
        t[replace] = slots[draws // self.num_envs]
        env[replace] = draws % self.num_envs
        return t, env

    def sample_starts(self, batch):
        t, env = self._sample_transitions(batch)
        return {k: v[t, env] for k, v in self.data.items() if k != 'invalid'}

    def sample_contacts(self, batch: int) -> dict[str, torch.Tensor]:
        """Draw single transitions, including failures excluded by complete dynamics windows."""
        slots, env_ids = self._sample_transitions(batch)
        keys = ['state', 'action', 'features', 'contact_labels']
        if 'link_wrenches_world' in self.data:
            keys.append('link_wrenches_world')
        return {f'contact/{key}': self.data[key][slots, env_ids] for key in keys}

    @staticmethod
    def _valid_window_starts(blocked: torch.Tensor, transitions: int) -> torch.Tensor:
        """Return the starts of contiguous windows with no blocked transitions."""
        if transitions == 0:
            return blocked.new_ones((blocked.shape[1], blocked.shape[0] + 1))
        counts = torch.cat((blocked.new_zeros((1, blocked.shape[1])), blocked.cumsum(0)))
        return (counts[transitions:] == counts[:-transitions]).T

    def _sample_window_starts(self, starts: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        cumulative = starts.cumsum(1)
        totals = cumulative[:, -1]
        if not (totals > 0).any():
            raise ReplayUnavailable('No eligible replay windows')
        env = torch.multinomial((totals > 0).float(), batch, replacement=True)
        choice = (torch.rand(batch, device=self.device) * totals[env]).long()
        start = torch.searchsorted(cumulative[env], choice[:, None], right=True)
        return start, env

    def sample_windows(self, batch, steps, *, complete_frames=False):
        if self.size < steps:
            raise ValueError('Not enough replay steps')
        invalid = self.data.get('invalid')
        if complete_frames:
            slots = self._slots(torch.arange(self.size, device=self.device))
            done = (self.data['terminated'] | self.data['truncated'])[slots]
            # A reset after the last frame does not break the window.
            blocked = done[:-1]
            if invalid is not None:
                blocked = blocked | invalid[slots[:-1]]
            starts = self._valid_window_starts(blocked, steps - 1)
            start, env = self._sample_window_starts(starts, batch)
            t = self._slots(start + torch.arange(steps - 1, device=self.device))
            window = {key: self.data[key][t, env[:, None]]
                      for key in ('state', 'action', 'previous_action', 'torque', 'features')}
            if 'link_wrenches_world' in self.data:
                window['link_wrenches_world'] = self.data['link_wrenches_world'][t, env[:, None]]
            window['next_state'] = self.data['state'][(t + 1) % self.length, env[:, None]]
            window['valid'] = torch.ones_like(t, dtype=torch.bool)
            return window

        if invalid is None:
            start = torch.randint(self.size - steps + 1, (batch, 1), device=self.device)
            env = torch.randint(self.num_envs, (batch,), device=self.device)
        else:
            slots = self._slots(torch.arange(self.size, device=self.device))
            starts = self._valid_window_starts(invalid[slots], steps)
            start, env = self._sample_window_starts(starts, batch)
        t = self._slots(start + torch.arange(steps, device=self.device))
        env = env[:, None]
        result = {k: v[t, env] for k, v in self.data.items() if k != 'invalid'}
        # The final transition itself is valid. Only following transitions cross reset.
        done = (result['terminated'] | result['truncated']).float()
        result['valid'] = torch.cat((torch.ones_like(done[:, :1]), (1 - done[:, :-1]).cumprod(1)), 1)
        return result
