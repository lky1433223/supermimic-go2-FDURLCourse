"""Choose surrounding terrain patches from per-environment crossing results."""

import math
from collections.abc import Mapping, Sequence

import torch


class TerrainCurriculum:
    """Hold terrain type, difficulty and crossing history for parallel environments."""

    def __init__(
        self,
        patches: Sequence[dict],
        settings: Mapping[str, float],
        num_envs: int,
        device: torch.device,
        generator: torch.Generator,
        dt: float,
        episode_steps: int,
    ) -> None:
        self._dt = dt
        self._episode_steps = episode_steps
        self._hold_steps = math.ceil(settings['hold_seconds'] / dt)
        self._pass_streak = settings['pass_streak']
        self._fail_streak = settings['fail_streak']
        self._review_probability = settings['top_review_probability']
        self._generator = generator
        self._device = device

        kinds = list(dict.fromkeys(patch['type'] for patch in patches))
        self._max_level = max(patch['level'] for patch in patches)
        by_level = self._index_patches(patches, kinds)
        self._variants = torch.tensor([[len(options) for options in levels] for levels in by_level], device=device)
        max_variants = max(len(options) for levels in by_level for options in levels)
        self._patch_lookup = torch.tensor([
            [options + [-1] * (max_variants - len(options)) for options in levels]
            for levels in by_level
        ], device=device)
        self._patch_centers = torch.tensor([p['position'][:2] for p in patches], device=device)
        self._patch_inner = torch.tensor([p['inner_half_width_m'] for p in patches], device=device)
        self._patch_outer = torch.tensor([p['outer_half_width_m'] for p in patches], device=device)

        kind_weights = torch.tensor([sum(p['weight'] for p in patches if p['type'] == kind) for kind in kinds],
                                    dtype=torch.float64)
        self.kind_ids = self._assign_kinds(kind_weights, num_envs)
        self.levels = torch.randint(0, settings['initial_max_level'] + 1, (num_envs,),
                                    generator=generator).to(device)
        self._passes = torch.zeros_like(self.levels)
        self._failures = torch.zeros_like(self.levels)
        self._center = torch.zeros((num_envs, 2), device=device)
        self._inner = torch.zeros(num_envs, device=device)
        self._outer = torch.zeros_like(self._inner)
        self._ideal_xy = torch.zeros_like(self._center)
        self._ideal_yaw = torch.zeros_like(self._inner)
        self._entered = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._ideal_reached = torch.zeros_like(self._entered)
        self._passed = torch.zeros_like(self._entered)
        self._steps_since_exit = torch.zeros_like(self.levels)

    def _index_patches(self, patches: Sequence[dict], kinds: list[str]) -> list[list[list[int]]]:
        """Group patch indices by type and difficulty, including the initial flat patch."""
        kind_lookup = {kind: index for index, kind in enumerate(kinds)}
        grouped: list[list[list[int]]] = [[[] for _ in range(self._max_level + 1)] for _ in kinds]
        for index, patch in enumerate(patches):
            grouped[kind_lookup[patch['type']]][patch['level']].append(index)
        if any(not options for levels in grouped for options in levels):
            raise ValueError('Each curriculum terrain type must provide a patch at every level from zero')
        return grouped

    def _assign_kinds(self, weights: torch.Tensor, count: int) -> torch.Tensor:
        """Stratify by weight, then shuffle so assignments do not cluster by environment ID."""
        cumulative = weights.cumsum(0) / weights.sum()
        offset = torch.rand((), generator=self._generator)
        quantiles = (torch.arange(count, dtype=weights.dtype) + offset) / count
        selected = torch.searchsorted(cumulative, quantiles)
        return selected[torch.randperm(count, generator=self._generator)].to(self._device)

    def sample_patches(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Choose a map variant for each selected environment's assigned type and level."""
        kinds, levels = self.kind_ids[env_ids], self.levels[env_ids]
        draw = torch.rand(len(env_ids), generator=self._generator).to(self._device)
        variant = (draw * self._variants[kinds, levels]).long()
        return self._patch_lookup[kinds, levels, variant]

    def start_episodes(self, env_ids: torch.Tensor, patch_ids: torch.Tensor,
                       position: torch.Tensor, quaternion: torch.Tensor) -> None:
        """Reset crossing history without changing the selected environments' level or streaks."""
        center = self._patch_centers[patch_ids]
        self._center[env_ids] = center
        self._inner[env_ids] = self._patch_inner[patch_ids]
        self._outer[env_ids] = self._patch_outer[patch_ids]
        self._ideal_xy[env_ids] = position[env_ids, :2] - center
        w, x, y, z = quaternion[env_ids].unbind(dim=-1)
        self._ideal_yaw[env_ids] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        self._entered[env_ids] = False
        self._ideal_reached[env_ids] = False
        self._passed[env_ids] = False
        self._steps_since_exit[env_ids] = 0

    def step(self, position: torch.Tensor, commands: torch.Tensor,
             terminated: torch.Tensor, episode_steps: torch.Tensor) -> None:
        """Record the actual crossing and integrate a reference trajectory from issued commands."""
        self._advance_ideal(commands, episode_steps)
        distance = (position[:, :2] - self._center).abs().amax(dim=-1)
        self._entered |= distance >= self._inner
        exited = distance >= self._outer
        self._steps_since_exit += exited | (self._steps_since_exit > 0)
        # Include a full hold interval after the sample that first crossed the boundary.
        self._passed |= (self._steps_since_exit > self._hold_steps) & ~terminated

    def _advance_ideal(self, commands: torch.Tensor, episode_steps: torch.Tensor) -> None:
        vx, vy, yaw_rate = commands.unbind(dim=-1)
        yaw_delta = yaw_rate * self._dt
        # Rotate local velocity at the midpoint of the control step.
        mid_yaw = self._ideal_yaw + 0.5 * yaw_delta
        cos_yaw, sin_yaw = mid_yaw.cos(), mid_yaw.sin()
        world_velocity = torch.stack((vx * cos_yaw - vy * sin_yaw, vx * sin_yaw + vy * cos_yaw), dim=-1)
        self._ideal_xy.add_(world_velocity, alpha=self._dt)
        self._ideal_yaw += yaw_delta
        can_hold = episode_steps + self._hold_steps <= self._episode_steps
        self._ideal_reached |= can_hold & (self._ideal_xy.abs().amax(dim=-1) >= self._outer)

    def finish_episodes(self, env_ids: torch.Tensor, terminated: torch.Tensor) -> dict[str, float]:
        """Apply outcomes at automatic reset and return counts for that reset batch."""
        passed = self._passed[env_ids]
        failed = ~passed & (self._ideal_reached[env_ids] | (terminated[env_ids] & self._entered[env_ids]))
        self._adjust_levels(env_ids, passed, failed)
        mean_level, top_fraction, pass_count, fail_count = torch.stack((
            self.levels.float().mean(), (self.levels == self._max_level).float().mean(),
            passed.float().sum(), failed.float().sum(),
        )).tolist()
        return {'mean_level': mean_level, 'level_9_fraction': top_fraction, 'passed': pass_count, 'failed': fail_count,
                'skipped': len(env_ids) - pass_count - fail_count}

    def _adjust_levels(self, env_ids: torch.Tensor, passed: torch.Tensor, failed: torch.Tensor) -> None:
        levels = self.levels[env_ids]
        passes = torch.where(failed, 0, self._passes[env_ids] + passed)
        failures = torch.where(passed, 0, self._failures[env_ids] + failed)
        promoted = passed & (levels < self._max_level) & ((levels == 0) | (passes >= self._pass_streak))
        demoted = failed & (levels > 1) & (failures >= self._fail_streak)

        draw = torch.rand((len(env_ids), 2), generator=self._generator).to(self._device)
        top_passed = passed & (levels == self._max_level)
        reviewing = top_passed & (draw[:, 0] < self._review_probability)
        review_level = (draw[:, 1] * (self._max_level - 1)).long() + 1
        self.levels[env_ids] = torch.where(reviewing, review_level, levels + promoted.long() - demoted.long())
        reset_passes = promoted | top_passed
        reset_failures = failures >= self._fail_streak
        self._passes[env_ids] = torch.where(reset_passes | demoted, 0, passes)
        self._failures[env_ids] = torch.where(reset_failures, 0, failures)
