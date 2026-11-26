"""Continuous stairs are configurable in authored, training and evaluation scenes."""

from pathlib import Path
import unittest

import torch

from rl_go2.config import load_experiment, load_scene
from rl_go2.evaluation.tasks import TASKS
from rl_go2.trainers.terrain_features import TerrainHeightScan

ROOT = Path(__file__).resolve().parents[1]


class StairsTests(unittest.TestCase):
    def test_overrides_define_treads_and_model_heights(self):
        scene = load_scene(ROOT / 'configs/scenes/stairs.yaml', entity_overrides={'stairs': {
            'step_height_m': 0.04, 'tread_depth_m': 0.25, 'num_steps': 4, 'width_m': 1.5,
        }})
        boxes = [entity for entity in scene['entities'] if entity['type'] == 'box']
        self.assertEqual(len(boxes), 4)
        points = []
        for index, box in enumerate(boxes):
            self.assertEqual(box['size'][:2], [0.25, 1.5])
            self.assertAlmostEqual(box['position'][0], 1 + (index + 0.5) * 0.25)
            self.assertAlmostEqual(box['size'][2], (index + 1) * 0.04)
            self.assertAlmostEqual(box['position'][2] - box['size'][2] / 2, 0)
            points.append(box['position'][:2])
        heights = TerrainHeightScan(scene).heights(torch.tensor(points))
        torch.testing.assert_close(heights, torch.tensor([0.04, 0.08, 0.12, 0.16]))
        for invalid in (0, -1, 2.5, True):
            with self.subTest(count=invalid), self.assertRaises(ValueError):
                load_scene(ROOT / 'configs/scenes/stairs.yaml', entity_overrides={'stairs': {'num_steps': invalid}})

    def test_training_and_evaluation_select_stairs(self):
        config = load_experiment(ROOT / 'configs/experiments/terrain_supermimic_push.yaml')
        patches = [patch for patch in config['scene']['terrain_patches'] if patch['type'] == 'stairs']
        self.assertEqual(len(patches), 1)
        boxes = [entity for entity in config['scene']['entities'] if entity['id'].startswith('stairs-')]
        self.assertEqual(len(boxes), 10)
        plan = TASKS['stairs'].build_plan()
        self.assertEqual(len(plan.trials), 4)
        self.assertTrue(all(trial.scene.startswith('stairs-') for trial in plan.trials))
        self.assertEqual({trial.motion.name for trial in plan.trials}, {'forward_0.5', 'lateral_0.5'})


if __name__ == '__main__':
    unittest.main()
