"""Geometry checks for the graded surround terrain scene."""

from math import radians, tan
from pathlib import Path
import unittest

import torch

from rl_go2.config import load_experiment, read_yaml
from rl_go2.surround_terrain import build_curriculum_patches
from rl_go2.terrain_features import TerrainHeightScan


ROOT = Path(__file__).resolve().parents[1]


class SurroundTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_experiment(ROOT / 'configs/experiments/curriculum_supermimic_rewards_graded.yaml')
        cls.scene = config['courses'][-1]['settings']['scene']

    def test_full_course_fits_around_origin_with_stairs(self):
        course = read_yaml(ROOT / 'configs/courses/terrain_graded.yaml')
        course['terrains']['stairs'] = {
            'step_height_levels_m': [0.02 + 0.005 * level for level in range(9)],
            'tread_depth_m': 0.30,
            'num_steps': 10,
        }
        _, patches, height_maps = build_curriculum_patches(course['terrains'], course['terrain_curriculum'], 1)
        coordinates = {(x, y) for x, y, _ in (patch['position'] for patch in patches)}
        self.assertEqual(coordinates, {(x, y) for x in range(-60, 61, 20) for y in range(-60, 61, 20)})
        self.assertEqual(len(coordinates), len(patches))
        for patch in patches:
            if patch['name'] in height_maps:
                x, y, _ = patch['position']
                height_map = height_maps[patch['name']]
                self.assertAlmostEqual(height_map['start_x_m'], x - patch['outer_half_width_m'])
                self.assertEqual(height_map['center_y_m'], y)

    def test_flat_spawn_and_continuous_obstacles_at_sides_and_corners(self):
        scan = TerrainHeightScan(self.scene)
        directions = torch.tensor([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y])
        for patch in (patch for patch in self.scene['terrain_patches'] if patch['level'] == 9):
            with self.subTest(patch=patch['name']):
                center = torch.tensor(patch['position'][:2])
                inner = patch['inner_half_width_m']
                flat_heights = scan.heights(center + directions * (inner - 0.1))
                self.assertTrue((flat_heights == 0).all())

                heights = scan.heights(center + directions * (inner + 0.15))
                if patch['type'] == 'rough_ground':
                    self.assertTrue((heights >= 0.005).all())
                else:
                    expected = {'ramp': 0.15 * tan(radians(20)), 'step': 0.10, 'stairs': 0.06}[patch['type']]
                    torch.testing.assert_close(heights, torch.full((8,), expected), atol=5e-5, rtol=0)

                if patch['type'] in ('ramp', 'stairs'):
                    landing = scan.heights(center + directions * (patch['outer_half_width_m'] + 0.8))
                    top = 0.8 * tan(radians(20)) if patch['type'] == 'ramp' else 0.6
                    torch.testing.assert_close(landing, torch.full((8,), top), atol=5e-5, rtol=0)

    def test_rough_maps_reach_their_configured_maximum_with_bounded_tile_deltas(self):
        maps = self.scene['height_maps']
        for level in range(1, 10):
            variants = [grid for name, grid in maps.items() if name.startswith(f'rough_ground-level-{level}-')]
            self.assertEqual(len(variants), 2)
            self.assertNotEqual(variants[0]['height_map_m'], variants[1]['height_map_m'])
            for grid in variants:
                heights = torch.tensor(grid['height_map_m'])
                max_delta = max(heights.diff(dim=0).abs().max(), heights.diff(dim=1).abs().max()).item()
                self.assertAlmostEqual(heights.max().item(), grid['max_height_m'])
                self.assertLessEqual(max_delta, grid['max_adjacent_delta_m'] + 1e-7)


if __name__ == '__main__':
    unittest.main()
