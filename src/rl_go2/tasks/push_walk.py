"""Velocity tracking with finite-duration, unpredictable horizontal force pulses."""

import math

import torch

from rl_go2.tasks.flat_walk import FlatWalkTask

DIRECTIONS = {'x+': [1, 0, 0], 'x-': [-1, 0, 0], 'y+': [0, 1, 0], 'y-': [0, -1, 0]}


def validate_force_range(value):
    if len(value) != 2 or not all(math.isfinite(x) for x in value) or value[0] < 0 or value[1] < value[0]:
        raise ValueError('push_force_range must be a finite, nonnegative ordered pair')


class PushWalkTask(FlatWalkTask):
    @staticmethod
    def validate_config(config, robot):
        FlatWalkTask.validate_config(config, robot)
        push = config['push']
        validate_force_range(push['force_range_n'])
        interval = push['interval_seconds']
        if len(interval) != 2 or not all(math.isfinite(x) for x in interval) or not 0 < interval[0] <= interval[1]:
            raise ValueError('push interval must be two finite positive ordered seconds')
        if not math.isfinite(push['duration_seconds']) or not 0 < push['duration_seconds'] < interval[0]:
            raise ValueError('Push duration must be positive and shorter than its minimum interval')
        directions = push['directions']
        if not directions or len(set(directions)) != len(directions) or any(d not in DIRECTIONS for d in directions):
            raise ValueError('Push directions must be unique selections of x+, x-, y+, y-')

    def __init__(self, config, default_positions, num_envs, device, dt, seed):
        super().__init__(config, default_positions, num_envs, device, dt, seed)
        push = config['push']
        self.duration_steps = round(push['duration_seconds'] / dt)
        if self.duration_steps < 1 or not math.isclose(self.duration_steps * dt, push['duration_seconds'], abs_tol=1e-6):
            raise ValueError('Push duration must be an integer number of control steps')
        self.force_ranges = torch.tensor(push['force_range_n'], device=device, dtype=torch.float32).expand(num_envs, -1).clone()
        self.directions = torch.tensor([DIRECTIONS[d] for d in push['directions']], device=device, dtype=torch.float32)
        self.next_push = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.remaining = torch.zeros_like(self.next_push)
        self.force = torch.zeros(num_envs, 3, device=device)
        self.applied_force = torch.zeros_like(self.force)

    def _interval(self, count):
        lo, hi = self.config['push']['interval_seconds']
        seconds = lo + torch.rand(count, generator=self.generator) * (hi - lo)
        return torch.ceil(seconds / self.dt).to(device=self.device, dtype=torch.long)

    def reset(self, env_ids, state, updates):
        updates = dict(updates)
        force_range = updates.pop('push_force_range', self.config['push']['force_range_n'])
        validate_force_range(force_range)
        super().reset(env_ids, state, updates)
        self.force_ranges[env_ids] = self.force_ranges.new_tensor(force_range)
        self.next_push[env_ids] = self._interval(len(env_ids))
        self.remaining[env_ids] = 0
        self.force[env_ids] = 0
        self.applied_force[env_ids] = 0

    def external_force(self, state, episode_steps):
        ids = ((episode_steps >= self.next_push) & (self.remaining == 0)).nonzero(as_tuple=False).flatten()
        if ids.numel():
            sample = torch.rand(len(ids), generator=self.generator).to(self.device)
            ranges = self.force_ranges[ids]
            magnitude = ranges[:, 0] + sample * (ranges[:, 1] - ranges[:, 0])
            direction = torch.randint(len(self.directions), (len(ids),), generator=self.generator).to(self.device)
            self.force[ids] = self.directions[direction] * magnitude.unsqueeze(-1)
            self.remaining[ids] = self.duration_steps
            self.next_push[ids] = episode_steps[ids] + self._interval(len(ids))
        active = self.remaining > 0
        self.applied_force.copy_(self.force * active.unsqueeze(-1))
        self.remaining.sub_(active.long())
        return self.applied_force

    def context(self):
        return {**super().context(), 'external_force_world': self.applied_force}
