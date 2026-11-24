"""Regression checks for held input, focus safety, and cross-host playback."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_go2.play.controls import Controls
from rl_go2.play.runner import load_inputs


class PlayTests(unittest.TestCase):
    def test_combination_release_and_speed_bounds(self):
        controls = Controls(0.8, 0.5)
        controls.key_event('w', True)
        controls.key_event('a', True)
        self.assertEqual(controls.sample()[0], [0.5, 0.0, 0.5])
        controls.clear_keys()
        controls.key_event('s', True)
        controls.key_event('d', True)
        self.assertEqual(controls.sample()[0], [-0.5, 0.0, -0.5])
        for _ in range(20):
            controls.key_event('up', True)
            controls.key_event('right', True)
        self.assertEqual((controls.speed, controls.yaw), (0.8, 0.5))
        for _ in range(20):
            controls.key_event('down', True)
            controls.key_event('left', True)
        self.assertEqual((controls.speed, controls.yaw), (0.0, 0.0))

    def test_release_focus_and_discrete_repeat(self):
        controls = Controls(0.8, 0.5)
        controls.key_event('w', True)
        controls.key_event('a', True)
        controls.key_event('w', False)
        self.assertEqual(controls.sample()[0], [0.0, 0.0, 0.5])
        controls.clear_keys()
        self.assertEqual(controls.sample()[0], [0.0, 0.0, 0.0])
        controls.key_event('pause', True)
        controls.key_event('pause', True, repeat=True)
        self.assertTrue(controls.sample()[1])
        controls.key_event('pause', False)
        controls.key_event('pause', True)
        self.assertFalse(controls.sample()[1])
        controls.key_event('reset', True)
        self.assertTrue(controls.sample()[3])
        self.assertFalse(controls.sample()[3])
        controls.key_event('quit', True)
        self.assertTrue(controls.sample()[2])

    @unittest.skipUnless(os.environ.get("RL_GO2_TEST_VIEWER") == "1", "requires native display")
    def test_genesis_plugin_consumes_keys_and_clears_focus(self):
        from pyglet.window import key
        from rl_go2.play.genesis_viewer import PlaybackPlugin

        controls = Controls(0.8, 0.5)
        plugin = PlaybackPlugin(controls)
        self.assertTrue(plugin.on_key_press(key.W, 0))
        self.assertEqual(controls.sample()[0][0], 0.5)
        plugin.on_key_release(key.W, 0)
        self.assertEqual(controls.sample()[0][0], 0.0)
        plugin.on_key_press(key.P, 0)
        plugin.on_key_press(key.P, 0)
        self.assertTrue(controls.sample()[1])
        plugin.on_key_press(key.W, 0)
        plugin.on_deactivate()
        self.assertEqual(controls.sample()[0][0], 0.0)

    def test_mujoco_native_key_and_focus_callbacks(self):
        import glfw
        from types import SimpleNamespace
        from rl_go2.play.mujoco_viewer import MujocoPlayViewer

        controls = Controls(0.8, 0.5)
        adapter = SimpleNamespace(controls=controls)
        MujocoPlayViewer._on_key(adapter, None, glfw.KEY_W, 0, glfw.PRESS, 0)
        self.assertEqual(controls.sample()[0][0], 0.5)
        MujocoPlayViewer._on_key(adapter, None, glfw.KEY_W, 0, glfw.RELEASE, 0)
        self.assertEqual(controls.sample()[0][0], 0.0)
        MujocoPlayViewer._on_key(adapter, None, glfw.KEY_P, 0, glfw.PRESS, 0)
        MujocoPlayViewer._on_key(adapter, None, glfw.KEY_P, 0, glfw.REPEAT, 0)
        self.assertTrue(controls.sample()[1])
        MujocoPlayViewer._on_key(adapter, None, glfw.KEY_W, 0, glfw.PRESS, 0)
        MujocoPlayViewer._on_focus(adapter, None, 0)
        self.assertEqual(controls.sample()[0][0], 0.0)

    def test_mujoco_camera_drag_and_zoom(self):
        import glfw
        import mujoco
        from types import SimpleNamespace
        from rl_go2.play.mujoco_viewer import MujocoPlayViewer

        camera = mujoco.MjvCamera()
        camera.distance = 3.0
        adapter = SimpleNamespace(camera=camera, last_cursor=(0.0, 0.0))
        with patch('glfw.get_mouse_button', return_value=glfw.PRESS), \
                patch('glfw.get_window_size', return_value=(1280, 800)):
            MujocoPlayViewer._on_cursor(adapter, None, 20, 10)
        self.assertGreater(float(abs(camera.lookat).sum()), 0.0)
        with patch('glfw.get_mouse_button', side_effect=[glfw.PRESS, glfw.RELEASE]), \
                patch('glfw.get_window_size', return_value=(1280, 800)):
            MujocoPlayViewer._on_cursor(adapter, None, 50, 20)
        self.assertNotEqual(camera.azimuth, 0.0)
        MujocoPlayViewer._on_scroll(adapter, None, 0.0, 1.0)
        self.assertLess(camera.distance, 3.0)

    def test_playback_lighting_uses_a_separate_model(self):
        import mujoco
        import numpy as np
        from rl_go2.play.mujoco_viewer import playback_model

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
        ambient = model.vis.headlight.ambient.copy()
        colors = model.geom_rgba.copy()
        visual = playback_model(model)
        self.assertTrue((visual.vis.headlight.ambient > ambient).all())
        np.testing.assert_array_equal(model.vis.headlight.ambient, ambient)
        np.testing.assert_array_equal(model.geom_rgba, colors)
        np.testing.assert_array_equal(visual.geom_friction, model.geom_friction)
        np.testing.assert_array_equal(visual.body_mass, model.body_mass)

    def test_hud_exposes_speed_command_and_pause(self):
        from rl_go2.play.controls import hud_text

        controls = Controls(0.8, 0.5)
        controls.publish(command=[0.4, -0.2], actual=[0.3, -0.1], seconds=2.0)
        text = hud_text(controls.status())
        self.assertIn('Command vx/yaw +0.40 / -0.20', text)
        self.assertIn('Actual  vx/yaw +0.30 / -0.10', text)
        controls.key_event('pause', True)
        self.assertIn('PAUSED', hud_text(controls.status()))

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
