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
    def test_next_terrain_key_is_consumed_once(self):
        controls = Controls(0.8, 0.5)
        controls.key_event('next_terrain', True)
        self.assertTrue(controls.consume_terrain_switch())
        controls.key_event('next_terrain', True, repeat=True)
        self.assertFalse(controls.consume_terrain_switch())

    def test_playback_terrains_keep_selected_spawn_after_reset(self):
        from unittest.mock import Mock

        import torch

        from rl_go2.config import load_experiment
        from rl_go2.env import QuadrupedEnv
        from rl_go2.play.runner import select_playback_terrain
        from rl_go2.play.terrain import playback_scene, terrain_order

        repo = Path(__file__).resolve().parents[1]
        for experiment in ('flat_supermimic', 'terrain_supermimic_push'):
            with self.subTest(experiment=experiment):
                config = load_experiment(repo / 'configs/experiments' / f'{experiment}.yaml')
                config.update(backend='mujoco', device='cpu', num_envs=1)
                scene = playback_scene(config['scene'], config['robot'], config['seed'], repo)
                config['scene'] = scene
                patches = scene['terrain_patches']
                order = terrain_order(patches)
                self.assertCountEqual(order, range(len(patches)))
                self.assertEqual([patches[index]['type'] for index in order[:4]],
                                 ['flat', 'step', 'ramp', 'rough_ground'])
                env = QuadrupedEnv(config)
                viewer = Mock()
                controls = Controls(0.8, 0.5)
                try:
                    for cursor, index in enumerate(order[:4]):
                        select_playback_terrain(env, viewer, controls, order, cursor)
                        expected = torch.tensor([patches[index]['position']])
                        torch.testing.assert_close(env.state.position, expected)
                        viewer.center_camera.assert_called_with(tuple(patches[index]['position']))
                        self.assertEqual(controls.status()['terrain'], patches[index]['name'])
                        env.reset()
                        torch.testing.assert_close(env.state.position, expected)
                finally:
                    env.close()

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
        adapter = SimpleNamespace(controls=controls, release_forces=lambda: None)
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
        adapter = SimpleNamespace(camera=camera, last_cursor=(0.0, 0.0), perturb=mujoco.MjvPerturb())
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

    def test_mujoco_perturbation_survives_each_physics_step_and_adds_task_force(self):
        import mujoco
        import numpy as np
        import torch
        from rl_go2.backends import make_backend
        from rl_go2.config import load_experiment
        from rl_go2.play.mujoco_viewer import MujocoPlayViewer
        from types import SimpleNamespace

        repo = Path(__file__).resolve().parents[1]
        config = load_experiment(repo / 'configs/experiments/flat_supermimic.yaml')
        config.update(backend='mujoco', device='cpu', num_envs=1)
        backend = make_backend(config)
        try:
            perturb = mujoco.MjvPerturb()
            perturb.select = backend.base
            scene = mujoco.MjvScene(backend.model, maxgeom=1000)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(backend.model, camera)
            mujoco.mjv_updateScene(backend.model, backend.data, mujoco.MjvOption(), None,
                                  camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjv_initPerturb(backend.model, backend.data, scene, perturb)
            perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
            mujoco.mjv_movePerturb(backend.model, backend.data, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                                  0.1, 0.0, scene, perturb)
            adapter = SimpleNamespace(perturb=perturb, controls=Controls(0.8, 0.5))
            backend.set_interactive_force_callback(
                lambda model, data: MujocoPlayViewer.apply_forces(adapter, model, data))
            for _ in range(3):
                backend.data.xfrc_applied[:] = 0
                mujoco.mjv_applyPerturbForce(backend.model, backend.data, perturb)
                expected = backend.data.xfrc_applied[backend.base].copy()
                self.assertGreater(np.linalg.norm(expected[:3]), 0)
                expected[0] += 1.0
                backend.step(torch.zeros(1, 12), base_force=torch.tensor([[1.0, 0.0, 0.0]]))
                np.testing.assert_allclose(backend.data.xfrc_applied[backend.base], expected)
            MujocoPlayViewer.release_forces(adapter)
            backend.step(torch.zeros(1, 12))
            self.assertFalse(backend.data.xfrc_applied.any())
            backend.set_interactive_force_callback(None)
        finally:
            backend.close()

    def test_mujoco_mouse_selection_and_release(self):
        import glfw
        import mujoco
        import numpy as np
        from types import SimpleNamespace
        from rl_go2.play.mujoco_viewer import MujocoPlayViewer

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="box"><freejoint/>'
            '<geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        controls = Controls(0.8, 0.5)
        adapter = SimpleNamespace(model=model, data=data, controls=controls,
                                  perturb=mujoco.MjvPerturb(), options=mujoco.MjvOption(),
                                  scene=mujoco.MjvScene(model, maxgeom=100))
        adapter.release_forces = lambda: MujocoPlayViewer.release_forces(adapter)
        with patch('glfw.get_cursor_pos', return_value=(640, 400)), \
                patch('glfw.get_window_size', return_value=(1280, 800)), \
                patch('mujoco.mjv_select', return_value=1), patch('mujoco.mjv_initPerturb'):
            MujocoPlayViewer._on_mouse_button(adapter, None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
        self.assertEqual(adapter.perturb.select, 1)
        self.assertTrue(adapter.perturb.active)
        self.assertEqual(controls.status()['mouse_body'], 'box')
        MujocoPlayViewer._on_mouse_button(adapter, None, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
        self.assertFalse(adapter.perturb.active)
        np.testing.assert_array_equal(adapter.perturb.localpos, [0, 0, 0])

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
        from rl_go2.play.hud import layout_hud

        controls = Controls(0.8, 0.5)
        controls.publish(backend='mujoco', policy='path/to/policy.pt',
                         command=[0.4, -0.2], actual=[0.3, -0.1], seconds=2.0)
        compact = layout_hud(controls.status())
        values = [label.value for label in compact.labels]
        self.assertIn('+0.40', values)
        self.assertIn('+0.30', values)
        self.assertIn('No grab', values)
        self.assertNotIn('path/to/policy.pt', values)

        controls.key_event('pause', True)
        controls.key_event('details', True)
        controls.key_event('details', True, repeat=True)
        details = layout_hud(controls.status())
        values = [label.value for label in details.labels]
        self.assertIn('PAUSED', values)
        self.assertIn('path/to/policy.pt', values)
        self.assertGreater(details.height, compact.height)

        controls.publish(mouse_body='base', mouse_force_n=80.0)
        active = layout_hud(controls.status())
        self.assertIn('80.0 N', [label.value for label in active.labels])
        self.assertGreater(active.height, details.height)

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
                loaded, _, _, _ = load_inputs(args)
            local = next(e for e in loaded['scene']['entities'] if e.get('robot'))
            self.assertEqual(local['file'], str(repo / 'assets/robots/go2/go2.xml'))
            self.assertEqual(json.loads(source.read_text()), original)


if __name__ == '__main__':
    unittest.main()
