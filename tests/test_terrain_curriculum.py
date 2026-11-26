"""Geometric crossing and per-environment progression on terraced terrain."""

import unittest

import torch

from rl_go2.terrain_curriculum import TerrainCurriculum


class TerrainCurriculumTests(unittest.TestCase):
    def make_curriculum(self, count: int, *, review_probability: float = 0.1) -> TerrainCurriculum:
        kinds = ('step', 'ramp', 'stairs', 'rough_ground')
        all_patches = [
            {'type': kind, 'level': level, 'weight': 0.1, 'position': [0.0, 0.0, 0.42],
             'inner_half_width_m': 1.0, 'outer_half_width_m': 1.6}
            for kind in kinds for level in range(10)
        ]
        settings = {'initial_max_level': 3, 'pass_streak': 2, 'fail_streak': 2, 'hold_seconds': 0.5,
                    'top_review_probability': review_probability}
        return TerrainCurriculum(all_patches, settings, count, torch.device('cpu'),
                                 torch.Generator().manual_seed(25), 0.25, 8)

    def start(self, course: TerrainCurriculum, env_ids: torch.Tensor) -> None:
        count = len(course.levels)
        course.start_episodes(env_ids, course.sample_patches(env_ids), torch.zeros(count, 3),
                              torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1))

    def test_balanced_types_and_random_initial_levels(self):
        course = self.make_curriculum(4096)
        env_ids = torch.arange(4096)
        self.assertEqual(course.kind_ids.bincount().tolist(), [1024] * 4)
        self.assertEqual(course.levels.unique().tolist(), [0, 1, 2, 3])
        self.assertEqual(course.sample_patches(env_ids).tolist(), (course.kind_ids * 10 + course.levels).tolist())

    def test_full_hold_and_true_command_opportunity_at_timeout(self):
        course = self.make_curriculum(5)
        env_ids = torch.arange(5)
        self.start(course, env_ids)
        positions = torch.zeros(5, 3)
        commands = torch.zeros(5, 3)
        done = torch.zeros(5, dtype=torch.bool)
        for step in range(1, 9):
            if step == 6:
                positions[:2, 0] = 1.61
            if step == 8:
                done[1] = True
            commands[:, 0] = torch.tensor([0., 0., 1.1, 1.0, 1.4 if step <= 4 else -1.4])
            course.step(positions, commands, done, torch.full((5,), step))
        outcome = course.finish_episodes(env_ids, done)
        self.assertEqual((outcome['passed'], outcome['failed'], outcome['skipped']), (1, 2, 2))

    def test_progression_review_and_floor(self):
        course = self.make_curriculum(4, review_probability=1.0)
        course.levels[:] = torch.tensor([0, 2, 1, 9])
        env_ids = torch.arange(4)
        position = torch.zeros(4, 3)
        commands = torch.zeros(4, 3)
        terminated = torch.tensor([False, False, True, False])
        self.start(course, env_ids)
        position[:, 0] = torch.tensor([1.7, 1.7, 1.1, 1.7])
        for step in range(1, 4):
            course.step(position, commands, terminated, torch.full((4,), step))
        course.finish_episodes(env_ids, terminated)
        levels = course.levels.tolist()
        self.assertEqual(levels[:3], [1, 2, 1])
        self.assertIn(levels[3], range(1, 9))

        # A skipped episode between successes preserves the prior success streak.
        chosen = torch.tensor([1])
        self.start(course, chosen)
        course.finish_episodes(chosen, torch.zeros_like(terminated))
        self.start(course, chosen)
        for step in range(1, 4):
            course.step(position, commands, terminated, torch.full((4,), step))
        course.finish_episodes(chosen, terminated)
        self.assertEqual(course.levels[1].item(), 3)
        self.assertEqual(course.levels[[0, 2, 3]].tolist(), [1, 1, levels[3]])

        # Two failed attempts lower level 3, but level 1 cannot return to flat.
        chosen = torch.tensor([1, 2])
        position[chosen, 0] = 1.1
        failures = torch.tensor([False, True, True, False])
        for _ in range(2):
            self.start(course, chosen)
            course.step(position, commands, failures, torch.ones(4, dtype=torch.long))
            course.finish_episodes(chosen, failures)
        self.assertEqual(course.levels[chosen].tolist(), [2, 1])


if __name__ == '__main__':
    unittest.main()
