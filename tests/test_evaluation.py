"""Evaluation plan composition and automatic-reset execution boundaries."""
from contextlib import contextmanager
from dataclasses import replace
from math import radians, sin, sqrt
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rl_go2.evaluation.push_events import sample_event
from rl_go2.evaluation.conditions import EvaluationCondition, PHYSICAL_SAMPLES, PhysicalSample
from rl_go2.evaluation.executor import execute_plan, execute_trial
from rl_go2.evaluation.metrics import TerrainMetrics
from rl_go2.evaluation.motion_tasks import MotionTask
from rl_go2.evaluation.runner import configure_evaluation_scene, output_records
from rl_go2.evaluation.task import Trial
from rl_go2.evaluation.tasks import TASKS
from rl_go2.evaluation.tasks.friction_switch import FrictionSwitchTask
from rl_go2.observation_noise import ObservationNoise
from rl_go2.step_delay import action_delay_from_config


class ResettingEnvironment:
    dt = 0.02

    def __init__(self):
        self.steps = 0
        self.task = SimpleNamespace(commands=torch.zeros(1, 3))
        self.commands = []

    def observe(self):
        return torch.zeros(1, 2)

    def step(self, action):
        self.steps += 1
        self.commands.append(self.task.commands.clone())
        self.task.commands.zero_()
        return self.observe(), torch.ones(1), torch.tensor([True]), {'transition': object()}


class EvaluationTests(unittest.TestCase):
    def test_terrain_tracking_failure_is_distinct_from_falling(self):
        plan = TASKS['terrain'].build_plan()
        for scene, final_x, velocity, reasons in (
            ('step-5cm', 0.5, (0.5, 0, 0), []),
            ('step-5cm', 2.0, (0, 0, 0), ['tracking_mse']),
            ('step-5cm-side', 0.5, (0, 0.5, 0), []),
            ('step-5cm-side', 0.0, (0, 0, 0), ['tracking_mse']),
            ('step-5cm-side', 0.5, (0, 0.5, 0.5), ['tracking_mse']),
        ):
            with self.subTest(scene=scene, velocity=velocity):
                trial = next(trial for trial in plan.trials if trial.scene == scene)
                command = trial.motion.command
                env = SimpleNamespace(
                    dt=1.0,
                    state=SimpleNamespace(position=torch.zeros(1, 3)),
                    task=SimpleNamespace(body_state=lambda state: (state.linear, state.angular, None)),
                    foot_friction=None, base_mass_kg=None, base_com_offset_m=None,
                )
                metrics = TerrainMetrics(env, trial)
                for step in range(8):
                    state = SimpleNamespace(
                        position=torch.tensor([[final_x * (step + 1) / 8, 0, 0.3]]),
                        quaternion=torch.tensor([[1., 0., 0., 0.]]),
                        linear=torch.tensor([[velocity[0], velocity[1], 0.0]]),
                        angular=torch.tensor([[0.0, 0.0, velocity[2]]]),
                    )
                    transition = SimpleNamespace(
                        next_state=state, terminated=torch.tensor([False]),
                        context={'commands': torch.tensor([command])},
                    )
                    metrics.update(env, transition, None, None, None, step)
                result = metrics.finish(env)
                self.assertFalse(result['fell'])
                self.assertEqual(result['success'], not reasons)
                self.assertEqual(result['failure_reasons'], reasons)
                self.assertAlmostEqual(result['forward_velocity_mse_after_2s_m2_s2'], (command[0] - velocity[0]) ** 2)

    def test_select_point_push_conditions_and_trial(self):
        task = TASKS['point-push']
        plan = task.build_plan()
        pushes = plan.select_trials(conditions=['base-200n-100ms', 'base-200n-300ms'])
        self.assertEqual(len(pushes.trials) * len(task.default_seeds), 16)
        one_trial = plan.select_trials(conditions=['base-200n-100ms'], motions=['vx0.5_yaw0'])
        self.assertEqual(
            [(trial.condition.name, trial.name) for trial in one_trial.trials],
            [('base-200n-100ms', 'vx0.5_yaw0')],
        )
        self.assertEqual(output_records(one_trial, 'base-200n-100ms'), {Path('base-200n-100ms'): one_trial})
        with self.assertRaisesRegex(ValueError, 'Unknown condition names: missing'):
            plan.select_trials(conditions=['base-200n-300ms', 'missing'])
        with self.assertRaisesRegex(ValueError, 'Unknown trial motion names: missing'):
            plan.select_trials(motions=['missing'])

    def test_legacy_suites_default_to_flat_for_a_step_training_run(self):
        terrain_plan = TASKS['terrain'].build_plan()
        step_scene = terrain_plan.scenes['step-10cm']
        step_scene['sim_options']['dt'] = 0.004
        robot = next(entity for entity in step_scene['entities'] if entity.get('robot'))
        remote_robot = '/training/assets/robots/go2/go2.xml'
        local_robot = Path(__file__).resolve().parents[1] / 'assets/robots/go2/go2.xml'
        robot.update(file=remote_robot, position=[1.3, 0, 0.48])
        config = {'scene': step_scene, 'robot': {'entity_id': robot['id']}}
        for name, task in TASKS.items():
            with self.subTest(task=name):
                plan = task.build_plan()
                scene = configure_evaluation_scene(config, plan)['scene']
                if plan.scenes:
                    self.assertIs(scene, step_scene)
                    continue
                ground, flat_robot = scene['entities']
                self.assertEqual(ground['type'], 'plane')
                self.assertEqual(flat_robot['position'], [0, 0, 0.42])
                self.assertEqual(flat_robot['file'], str(local_robot))
                self.assertEqual(scene['sim_options']['dt'], 0.004)
        self.assertEqual(robot['position'], [1.3, 0, 0.48])
        self.assertEqual(robot['file'], remote_robot)

    def test_sensor_action_task_isolates_and_combines_perturbations(self):
        config = {'domain_randomization': {
            'action_delay_steps': [0, 2], 'observation_noise': {'frame_delay_steps': [0, 1]},
        }}
        for name, task in TASKS.items():
            if name in ('sensor-action', 'combined'):
                continue
            for trial in task.build_plan().trials:
                domain = trial.condition.apply(config)['domain_randomization']
                self.assertNotIn('observation_noise', domain)
                self.assertNotIn('action_delay_steps', domain)
        plan = TASKS['sensor-action'].build_plan()
        cases = output_records(plan, 'comparison')
        self.assertEqual(list(map(str, cases)), ['sensor-only', 'action-only', 'combined'])
        self.assertEqual(len(plan.trials), 24)
        expected_noise = {'angular_velocity': 0.20, 'projected_gravity': 0.05,
                          'joint_position': 0.01, 'joint_velocity': 1.50}
        for path, case in cases.items():
            self.assertEqual(len(case.trials), 8)
            effective = case.trials[0].condition.apply(config)
            delay = action_delay_from_config(effective)
            if path.name == 'sensor-only':
                self.assertIsNone(delay)
            else:
                self.assertEqual((delay.min_steps, delay.max_steps), (1, 1))
            noise = ObservationNoise.from_config(effective)
            if path.name == 'action-only':
                self.assertIsNone(noise)
            else:
                frame_delay = noise.frame_delay.delay
                self.assertEqual((frame_delay.min_steps, frame_delay.max_steps), (1, 1))
                self.assertFalse(noise.chains['attitude'])
                for channel, amplitude in expected_noise.items():
                    perturbation, = noise.chains[channel]
                    torch.testing.assert_close(perturbation.low, torch.full_like(perturbation.low, -amplitude))
                    torch.testing.assert_close(perturbation.high, torch.full_like(perturbation.high, amplitude))

    def test_terrain_plan_resolves_one_box_per_step_and_partitions_outputs(self):
        task = TASKS['terrain']
        plan = task.build_plan()
        obstacles = ['step-5cm', 'step-10cm', 'stairs-5cm', 'stairs-10cm',
                     'ramp-10deg', 'ramp-20deg', 'rough-101', 'rough-102']
        scenes = [*obstacles, *(f'{name}-side' for name in obstacles)]
        self.assertEqual([trial.scene for trial in plan.trials], scenes)
        self.assertEqual(len({trial.event_key(1) for trial in plan.trials}), 16)
        self.assertEqual({trial.motion.duration_seconds for trial in plan.trials}, {8})
        for trial in plan.trials:
            lateral = trial.scene.endswith('-side')
            self.assertEqual(trial.motion.command, (0, 0.5, 0) if lateral else (0.5, 0, 0))
            robot = next(entity for entity in plan.scenes[trial.scene]['entities'] if entity.get('robot'))
            self.assertEqual(robot['position'], [0, 0, 0.42])
            self.assertEqual(robot['rotation'], [sqrt(0.5), 0, 0, -sqrt(0.5)] if lateral else [1, 0, 0, 0])
        for name, height in (('step-5cm', 0.05), ('step-10cm', 0.10)):
            box, = (entity for entity in plan.scenes[name]['entities'] if entity['type'] == 'box')
            self.assertTrue(box['fixed'])
            self.assertAlmostEqual(box['position'][0] - box['size'][0] / 2, 1.0)
            self.assertAlmostEqual(box['position'][2] - box['size'][2] / 2, 0.0)
            self.assertAlmostEqual(box['position'][2] + box['size'][2] / 2, height)
        for angle in (10, 20):
            ramp = next(entity for entity in plan.scenes[f'ramp-{angle}deg']['entities'] if entity['id'] == 'ramp-up')
            self.assertAlmostEqual(abs(ramp['rotation'][2]), sin(radians(angle) / 2))
        records = output_records(plan, 'nominal')
        self.assertEqual([str(path) for path in records], [f'{name}/nominal' for name in scenes])
        for case in records.values():
            for trial in case.trials:
                self.assertEqual(case.scene_config({}, trial)['scene'], plan.scenes[trial.scene])
        for seed in (101, 102):
            rough = plan.scenes[f'rough-{seed}']['height_maps']['rough']
            self.assertEqual(rough['seed'], seed)
            self.assertAlmostEqual(rough['max_height_m'], 0.05)
            self.assertAlmostEqual(rough['max_adjacent_delta_m'], 0.05)
            self.assertEqual(rough, plan.scenes[f'rough-{seed}-side']['height_maps']['rough'])

    def test_execute_plan_reuses_environment_until_scene_changes(self):
        plan = TASKS['terrain'].build_plan()
        first_trial = plan.trials[0]
        second_motion = replace(first_trial.motion, name='second_motion')
        plan = replace(plan, trials=(first_trial, replace(first_trial, motion=second_motion), *plan.trials[1:]))
        opened = []
        closed = []

        @contextmanager
        def environment(config, seed, condition, *, viewer):
            scene = config['scene']
            opened.append(scene)
            yield scene
            closed.append(scene)

        def execute_trial(env, policy, seed, trial, video, *, push_event):
            self.assertIs(env, plan.scenes[trial.scene])
            return {'group': trial.group, 'scene': trial.scene, 'condition': trial.condition.name, 'fell': False}

        with patch('rl_go2.evaluation.executor.environment', environment), patch(
            'rl_go2.evaluation.executor.execute_trial', execute_trial
        ):
            report = execute_plan({}, None, [1], plan)
        self.assertEqual(opened, list(plan.scenes.values()))
        self.assertEqual(closed, opened)
        self.assertEqual(report['by_scene'][first_trial.scene], {'trials': 2, 'falls': 0})

    def test_explicit_conditions_and_motion_lists(self):
        rich_motion = TASKS['rich-motion'].build_plan()
        low = TASKS['low-friction'].build_plan()
        self.assertNotIn(FrictionSwitchTask.TASK, TASKS)
        switch = FrictionSwitchTask().build_plan()
        rollout = TASKS['rollout'].build_plan()
        self.assertEqual(len(rich_motion.trials), 37)
        self.assertEqual(TASKS['rich-motion'].default_seeds, (1,))
        self.assertEqual({trial.condition.name for trial in rich_motion.trials}, {'nominal'})
        self.assertTrue(all(trial.motion.stop_at_seconds is None for trial in rich_motion.trials))
        limits = ((-1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0))
        in_range = [trial.motion.command for trial in rich_motion.trials if trial.group == 'in-range']
        self.assertEqual(len(in_range), 31)
        self.assertTrue(all(
            all(low <= value <= high for value, (low, high) in zip(command, limits, strict=True))
            for command in in_range
        ))
        self.assertIn((0, 0.5, 0), in_range)
        self.assertIn((0, 0, -0.25), in_range)
        self.assertEqual(
            {trial.motion.command for trial in rich_motion.trials if trial.group == 'out-of-range'},
            {(-1.25, 0, 0), (2.5, 0, 0), (0, -1.25, 0), (0, 1.25, 0), (0, 0, -1.25), (0, 0, 1.25)},
        )
        self.assertEqual(len(low.trials), 96)
        self.assertEqual(len(switch.trials), 72)
        self.assertEqual(len(rollout.trials), 8)
        self.assertEqual(rollout.trials[0].condition, rich_motion.trials[0].condition)
        expected_sliding = {
            'uniform-01': [0.1, 0.1, 0.1, 0.1],
            'uniform-02': [0.2, 0.2, 0.2, 0.2],
            'left-low': [0.1, 0.6, 0.2, 0.8],
            'right-low': [0.6, 0.1, 0.8, 0.2],
        }
        expected_bodies = {
            'nominal': (1.0, 0.0),
            'mass-max': (1.5, 0.0),
            'com-y-max': (1.0, 0.05),
        }
        expected_commands = {(0, 0, 0), (0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0),
                             (0, 0, 0.5), (2, 0, 0), (0.5, 0, 0.5), (0.5, 0, -0.5)}
        for name, task in TASKS.items():
            if name in ('rich-motion', 'terrain', 'stairs', 'combined'):
                continue
            groups = {}
            for trial in task.build_plan().trials:
                groups.setdefault((trial.scene, trial.condition.name), []).append(trial.motion.command)
            for commands in groups.values():
                self.assertCountEqual(commands, expected_commands, name)
        for body, (mass, y_offset) in expected_bodies.items():
            for name, values in expected_sliding.items():
                trials = [trial for trial in low.trials if trial.condition.name == f'{body}-{name}']
                self.assertEqual(len(trials), 8)
                self.assertEqual({trial.motion.command for trial in trials}, expected_commands)
                condition = trials[0].condition
                self.assertEqual(condition.base_mass['scale_range'], [mass, mass])
                self.assertEqual(condition.base_com['offset_ranges_m'], {
                    'x': [0.0, 0.0], 'y': [y_offset, y_offset], 'z': [0.0, 0.0],
                })
                feet = condition.foot_ground_friction['base_ranges']
                self.assertEqual([feet[foot]['sliding'][0] for foot in ('FL', 'FR', 'RL', 'RR')], values)
        self.assertEqual(low.trials[0].condition.base_mass, rich_motion.trials[0].condition.base_mass)
        source = {'domain_randomization': {'base_mass': {'enabled': False}, 'unrelated': {}}}
        effective = low.trials[0].condition.apply(source)
        self.assertTrue(effective['domain_randomization']['base_mass']['enabled'])
        self.assertNotIn('unrelated', effective['domain_randomization'])
        self.assertFalse(source['domain_randomization']['base_mass']['enabled'])

    def test_friction_switch_physical_samples_are_reproducible(self):
        for seed, sample in enumerate(PHYSICAL_SAMPLES[1:], 1):
            self.assertEqual(sample, PhysicalSample.sample(seed))
            self.assertTrue(0.9 <= sample.mass_scale <= 1.1)
            self.assertTrue(0.75 <= sample.friction_multiplier <= 1.25)
            for offset, limit in zip(sample.com_offset_m, (0.01, 0.03, 0.01), strict=True):
                self.assertLessEqual(abs(offset), limit)
            baseline = sample.condition(sample.name, (0.8,) * 4)
            conditions = [condition for condition in FrictionSwitchTask.conditions
                          if condition.name.startswith(f'{sample.name}-')]
            self.assertTrue(conditions)
            for condition in conditions:
                self.assertEqual(condition.base_mass, baseline.base_mass)
                self.assertEqual(condition.base_com, baseline.base_com)

    def test_body_parameter_sweeps_use_declared_motion_and_physics(self):
        mass = TASKS['mass']
        com = TASKS['com-offset']
        mass_plan = mass.build_plan()
        self.assertEqual(len(mass_plan.trials), 32)
        com_plan = com.build_plan()
        self.assertEqual(len(com_plan.trials), 64)
        for task in (mass, com):
            self.assertEqual(task.environment_task, 'flat_walk')
            for condition in task.conditions:
                self.assertEqual(condition.foot_ground_friction, TASKS['rollout'].conditions[0].foot_ground_friction)
                self.assertEqual(condition.friction_phases, ())
        mass_conditions = {
            'nominal': (1.0, 0.0),
            'mass-0.85': (0.85, 0.0),
            'mass-1.50': (1.5, 0.0),
            'mass-1.50-com-y-plus-5cm': (1.5, 0.05),
        }
        for name, (scale, y_offset) in mass_conditions.items():
            trials = [trial for trial in mass_plan.trials if trial.condition.name == name]
            self.assertEqual(len(trials), 8)
            condition = trials[0].condition
            self.assertEqual(condition.base_mass['scale_range'], [scale, scale])
            self.assertEqual(condition.base_com['offset_ranges_m'], {
                'x': [0.0, 0.0], 'y': [y_offset, y_offset], 'z': [0.0, 0.0],
            })
        offsets = []
        for condition in com.conditions:
            trials = [trial for trial in com_plan.trials if trial.condition.name == condition.name]
            self.assertEqual(len(trials), 8)
            self.assertEqual(condition.base_mass['scale_range'], [1.0, 1.0])
            ranges = condition.base_com['offset_ranges_m']
            for low, high in ranges.values():
                self.assertEqual(low, high)
            offsets.append(tuple(ranges[axis][0] for axis in 'xyz'))
        self.assertCountEqual(offsets, [
            (0, 0, 0), (-0.05, 0, 0), (0.05, 0, 0), (0, -0.05, 0), (0, 0.05, 0),
            (0, 0, -0.05), (0, 0, 0.05), (0.05, 0.05, 0.05),
        ])

    def test_reset_rule_controls_shared_loop(self):
        for continue_after_reset, expected in ((False, 1), (True, 4)):
            with self.subTest(continue_after_reset=continue_after_reset):
                env = ResettingEnvironment()
                trial = Trial(EvaluationCondition('nominal'), MotionTask('forward', 0.5), 'rollout', steps=4,
                              continue_after_reset=continue_after_reset)
                video = SimpleNamespace(frame=lambda *args, **kwargs: None)
                result = execute_trial(env, lambda observation: observation, 1, trial, video)
                self.assertEqual(env.steps, expected)
                self.assertEqual(result['episode_resets'], expected)
                self.assertEqual(result['steps'], expected)
                for command in env.commands:
                    torch.testing.assert_close(command, torch.tensor([[0.5, 0, 0]]))

    def test_point_push_sweeps_lateral_force_and_samples_distinct_feet(self):
        task = TASKS['point-push']
        plan = task.build_plan()
        self.assertEqual(task.default_seeds, (1,))
        self.assertEqual(len(plan.trials), 48)
        geometry = SimpleNamespace(sample_points=lambda directions: directions + torch.rand(directions.shape))
        config = {'robot': {'base_link': 'base'}}
        env = SimpleNamespace(
            backend=SimpleNamespace(link_names=['base']), num_envs=1,
            state=SimpleNamespace(position=torch.zeros(1, 3), quaternion=torch.tensor([[sqrt(0.5), 0, 0, sqrt(0.5)]])),
        )
        expected = {
            'base-50n-300ms': (50.0, 15), 'base-100n-200ms': (100.0, 10),
            'base-200n-100ms': (200.0, 5), 'base-200n-300ms': (200.0, 15),
            'foot-20n-draw-1': (20.0, 10), 'foot-20n-draw-2': (20.0, 10),
        }
        for motion in TASKS['low-friction'].motions['tracking']:
            events = {}
            for trial in plan.select_trials(motions=[motion.name]).trials:
                event = sample_event(config, geometry, 0.02, trial, 1)
                self.assertEqual(event, sample_event(config, geometry, 0.02, trial, 1))
                magnitude, duration_steps = expected[trial.condition.name]
                self.assertEqual((event.start_step, event.end_step), (250, 250 + duration_steps))
                torch.testing.assert_close(torch.tensor(event.force_n).norm(), torch.tensor(magnitude))
                if trial.condition.name.startswith('base-'):
                    self.assertEqual(event.link_name, 'base')
                    self.assertEqual(event.force_frame, 'heading')
                    torch.testing.assert_close(event.loads(env).force_world_n[0, 0], torch.tensor([-magnitude, 0, 0]),
                                               atol=3e-5, rtol=0)
                else:
                    self.assertTrue(event.link_name.endswith('_calf'))
                    self.assertEqual(event.force_frame, 'world')
                events[trial.condition.name] = event
            self.assertEqual(set(events), expected.keys())
            self.assertEqual(len({event.point_local_m for event in events.values() if event.link_name == 'base'}), 1)
            self.assertNotEqual(events['foot-20n-draw-1'].link_name, events['foot-20n-draw-2'].link_name)

    def test_combined_push_starts_only_inside_rough_ground(self):
        plan = TASKS['combined'].build_plan()
        self.assertCountEqual([trial.motion.command for trial in plan.trials], [
            (0.5, 0, 0), (0, 0.5, 0), (2.0, 0, 0),
        ])
        trial = next(trial for trial in plan.trials if trial.name == 'forward_2.0')
        config = plan.scene_config({'robot': {'base_link': 'base'}}, trial)
        geometry = SimpleNamespace(sample_points=lambda directions: directions)
        event = sample_event(config, geometry, 0.02, trial, 1)
        self.assertIsNone(event.start_step)
        self.assertEqual(event.duration_steps, 10)
        self.assertEqual(event.force_n, (0.0, 100.0, 0.0))
        noise_and_delay = trial.condition.apply({})['domain_randomization']
        self.assertEqual(noise_and_delay['observation_noise']['frame_delay_steps'], [1, 1])
        self.assertEqual(noise_and_delay['action_delay_steps'], (1, 1))

        for enters in (False, True):
            with self.subTest(enters=enters):
                observation = torch.zeros(1, 2)
                done = torch.tensor([False])
                state = SimpleNamespace(position=torch.tensor([[0.0, 0.7, 0.42]]),
                                        quaternion=torch.tensor([[1.0, 0, 0, 0]]))
                env = SimpleNamespace(
                    dt=0.02, state=state, reset=lambda: None, observe=lambda: observation,
                    backend=SimpleNamespace(link_names=['base']), num_envs=1,
                    task=SimpleNamespace(commands=torch.zeros(1, 3), body_state=lambda state: (
                        torch.tensor([[2.0, 0, 0]]), torch.zeros(1, 3), torch.tensor([[0.0, 0, -1.0]]),
                    )),
                    foot_friction=None, base_mass_kg=None, base_com_offset_m=None,
                )
                applied = []
                steps = 0

                def step(action, *, link_point_forces=None):
                    nonlocal steps
                    if link_point_forces is not None:
                        applied.append(steps)
                    steps += 1
                    x = min(steps * 0.1, 1.0)
                    y = 0.0 if enters and steps >= 12 else 0.7
                    state.position = torch.tensor([[x, y, 0.42]])
                    transition = SimpleNamespace(next_state=state, terminated=done,
                                                 context={'commands': env.task.commands})
                    return observation, torch.ones(1), done, {'transition': transition}

                env.step = step
                result = execute_trial(env, lambda obs: obs, 1, trial, None, push_event=event)
                recovery = result['push_recovery']
                self.assertEqual(applied, list(range(12, 22)) if enters else [])
                self.assertEqual(recovery['status'], 'recovered' if enters else 'not_triggered')
                self.assertEqual(result['success'], enters)

    def test_motion_and_friction_events_use_their_own_schedules(self):
        switch = FrictionSwitchTask().build_plan()
        for name, expected in (
            ('nominal-switch', {0: 1.0, 250: 0.5, 500: 1.0}),
            ('nominal-switch-low', {0: 1.0, 250: 0.25, 500: 1.0}),
            ('nominal-switch-short', {0: 1.0, 250: 0.25, 350: 1.0}),
        ):
            trial = next(trial for trial in switch.trials
                         if trial.condition.name == name and trial.name == 'walk_to_stop')
            schedule = trial.condition.phase_schedule(0.02)
            self.assertEqual({step: phase.multiplier for step, phase in schedule.items()}, expected)
            self.assertEqual(trial.motion.command_schedule(0.02)[400], (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
