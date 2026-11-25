"""Spatial height lookup agrees with simulator geometry across terrain patches."""

from copy import deepcopy
from pathlib import Path
import unittest

import mujoco
import numpy as np
import torch

from rl_go2.config import load_experiment
from rl_go2.env import QuadrupedEnv
from rl_go2.trainers.terrain_features import TerrainHeightScan

ROOT = Path(__file__).resolve().parents[1]


class TerrainLookupTests(unittest.TestCase):
    def test_local_candidates_match_simulator_across_patches_and_boundaries(self):
        config = load_experiment(ROOT / 'configs/experiments/terrain_supermimic_push.yaml')
        config.update(backend='mujoco', device='cpu', num_envs=1, domain_randomization={})
        scene = config['scene']
        points = [[-10., -100.], [100., 1000.], [0., 0.]]
        generator = np.random.default_rng(42)
        boxes = [entity for entity in scene['entities'] if entity['type'] == 'box']
        for box in boxes:
            center = np.array(box['position'][:2])
            # Avoid exact discontinuities in the simulator comparison.
            for offset in generator.uniform(-0.35, 0.35, (3, 2)):
                points.append((center + offset).tolist())
        points.extend([[value, 0.] for value in (0.3 - 1e-5, 0.3, 0.3 + 1e-5)])
        xy = torch.tensor(points, dtype=torch.float32, requires_grad=True)
        lookup = TerrainHeightScan(scene)
        # The candidate count stays local even when all terrain patches are present.
        self.assertLess(lookup.surface_index.shape[-1], len(boxes) // 2)
        env = QuadrupedEnv(config)
        try:
            expected = []
            for x, y in xy.detach().tolist():
                distance = mujoco.mj_ray(env.backend.model, env.backend.data, np.array([x, y, 10.]),
                                         np.array([0., 0., -1.]), None, True, -1,
                                         np.zeros(1, dtype=np.int32))
                expected.append(10 - distance)
            values = lookup.heights(xy)
            torch.testing.assert_close(values, torch.tensor(expected), atol=2e-5, rtol=1e-5)
            values.sum().backward()
            self.assertTrue(torch.isfinite(xy.grad).all())
        finally:
            env.close()

    def test_slope_gradient_and_overlapping_surfaces(self):
        config = load_experiment(ROOT / 'configs/experiments/flat_supermimic.yaml')
        scene = deepcopy(config['scene'])
        scene['entities'].append({'id': 'incline', 'type': 'box', 'fixed': True,
                                  'size': [2., 1., 0.1], 'position': [2., 0., 0.3],
                                  'rotation': [0.995004165, 0., -0.099833417, 0.]})
        scene['entities'].append({'id': 'platform', 'type': 'box', 'fixed': True,
                                  'size': [0.2, 0.2, 1.], 'position': [2., 0., 0.5],
                                  'rotation': [1., 0., 0., 0.]})
        lookup = TerrainHeightScan(scene)
        xy = torch.tensor([[1.7, 0.], [2., 0.]], requires_grad=True)
        heights = lookup.heights(xy)
        self.assertAlmostEqual(heights[1].item(), 1.)
        heights.sum().backward()
        self.assertAlmostEqual(xy.grad[0, 0].item(), float(np.tan(0.2)), places=5)
        torch.testing.assert_close(xy.grad[1], torch.zeros(2))


if __name__ == '__main__':
    unittest.main()
