"""Regression checks for held input, focus safety, and cross-host playback."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from rl_go2.play.controls import Controls
from rl_go2.play.panel import serve_panel
from rl_go2.play.runner import load_inputs


class PlayTests(unittest.TestCase):
    def test_combination_release_and_speed_bounds(self):
        controls = Controls(0.8, 0.5)
        controls.receive(['w', 'a'])
        self.assertEqual(controls.sample()[0], [0.5, 0.0, 0.5])
        controls.receive(['s', 'd'])
        self.assertEqual(controls.sample()[0], [-0.5, 0.0, -0.5])
        for _ in range(20):
            controls.receive([], 'up')
            controls.receive([], 'right')
        self.assertEqual((controls.speed, controls.yaw), (0.8, 0.5))
        for _ in range(20):
            controls.receive([], 'down')
            controls.receive([], 'left')
        self.assertEqual((controls.speed, controls.yaw), (0.0, 0.0))

    def test_timeout_pause_stop_and_reset_clear_input(self):
        controls = Controls(0.8, 0.5)
        controls.receive(['w'])
        with patch('rl_go2.play.controls.monotonic', return_value=controls.last_input + 1):
            self.assertEqual(controls.sample()[0], [0.0, 0.0, 0.0])
        for action in ('pause', 'reset', 'stop', 'quit'):
            controls.receive(['w'], action)
            self.assertEqual(controls.sample()[0], [0.0, 0.0, 0.0])
        controls.receive([], 'reset')
        self.assertTrue(controls.sample()[3])
        self.assertFalse(controls.sample()[3])

    def test_panel_serves_and_accepts_keyboard_updates(self):
        controls = Controls(0.8, 0.5)
        with serve_panel(controls) as url:
            with urlopen(url) as response:
                self.assertIn(b'Go2', response.read())
            request = Request(url + '/input', data=b'{"keys":["w","d"]}',
                              headers={'Content-Type': 'application/json'})
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(controls.sample()[0], [0.5, 0.0, -0.5])
            with urlopen(url + '/status') as response:
                self.assertEqual(json.load(response)['speed'], 0.5)

    def test_relocates_icc_assets_without_changing_source(self):
        from rl_go2.config import load_experiment

        repo = Path(__file__).resolve().parents[1]
        config = load_experiment(repo / 'configs/experiments/flat_supermimic.yaml')
        robot = next(e for e in config['scene']['entities'] if e.get('robot'))
        robot['file'] = '/home/lky/RL-Go2/assets/robots/go2/go2.xml'
        original = deepcopy(config)
        with tempfile.TemporaryDirectory() as directory:
            training = Path(directory) / 'training'
            training.mkdir()
            source = training / 'config.json'
            source.write_text(json.dumps(config))
            args = argparse.Namespace(run_dir=Path(directory), backend='mujoco', device='cpu',
                                      seed=1, iteration=None, checkpoint=None, policy=None)
            with patch('rl_go2.play.runner.load_policy', return_value=(None, training / 'policy.pt', 0)):
                loaded, _, _ = load_inputs(args)
            local = next(e for e in loaded['scene']['entities'] if e.get('robot'))
            self.assertEqual(local['file'], str(repo / 'assets/robots/go2/go2.xml'))
            self.assertEqual(json.loads(source.read_text()), original)


if __name__ == '__main__':
    unittest.main()
