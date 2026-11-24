"""Evaluation plan composition and automatic-reset execution boundaries."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rl_go2.evaluation.push_events import sample_event
from rl_go2.evaluation.conditions import EvaluationCondition, PHYSICAL_SAMPLES, PhysicalSample
from rl_go2.evaluation.executor import execute_plan, execute_trial
from rl_go2.evaluation.motion_tasks import MotionTask
from rl_go2.evaluation.runner import configure_evaluation_scene, output_records
from rl_go2.evaluation.task import Trial
from rl_go2.evaluation.tasks import TASKS


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
    def test_select_point_push_conditions_and_trial(self):
        task = TASKS['point-push']
        plan = task.build_plan()
        short_pushes = plan.select_trials(conditions=['base-80n', 'base-80n-short'])
        self.assertEqual(len(short_pushes.trials) * len(task.default_seeds), 36)
        one_trial = plan.select_trials(conditions=['base-80n-short'], motions=['vx0.2_yaw0'])
        self.assertEqual(
            [(trial.condition.name, trial.name) for trial in one_trial.trials],
            [('base-80n-short', 'vx0.2_yaw0')],
        )
        self.assertEqual(output_records(one_trial, 'base-80n-short'), {Path('base-80n-short'): one_trial})
        with self.assertRaisesRegex(ValueError, 'Unknown condition names: missing'):
            plan.select_trials(conditions=['base-80n', 'missing'])
        with self.assertRaisesRegex(ValueError, 'Unknown trial motion names: missing'):
            plan.select_trials(motions=['missing'])

    def test_legacy_suites_default_to_flat_for_a_step_training_run(self):
        terrain_plan = TASKS['terrain'].build_plan()
        step_scene = terrain_plan.scenes['step-6cm']
        step_scene['sim_options']['dt'] = 0.004
        robot = next(entity for entity in step_scene['entities'] if entity.get('robot'))
        robot.update(file='/training/assets/go2.xml', position=[1.3, 0, 0.48])
        config = {'scene': step_scene, 'robot': {'entity_id': robot['id']}}
        for name, task in TASKS.items():
            with self.subTest(task=name):
                scene = configure_evaluation_scene(config, task.build_plan())['scene']
                if name == 'terrain':
                    self.assertIs(scene, step_scene)
                    continue
                ground, flat_robot = scene['entities']
                self.assertEqual(ground['type'], 'plane')
                self.assertEqual(flat_robot['position'], [0, 0, 0.42])
                self.assertEqual(flat_robot['file'], '/training/assets/go2.xml')
                self.assertEqual(scene['sim_options']['dt'], 0.004)
        self.assertEqual(robot['position'], [1.3, 0, 0.48])

    def test_terrain_plan_resolves_one_box_per_step_and_partitions_outputs(self):
        task = TASKS['terrain']
        plan = task.build_plan()
        scenes = ['flat', 'step-2cm', 'step-4cm', 'step-6cm', 'ramp-5deg', 'ramp-10deg']
        self.assertEqual([trial.scene for trial in plan.trials], scenes)
        self.assertEqual(len({trial.event_key(1) for trial in plan.trials}), len(scenes))
        self.assertEqual(len({trial.motion for trial in plan.trials}), 1)
        self.assertEqual([entity['type'] for entity in plan.scenes['flat']['entities']], ['plane', 'mjcf'])
        for name, height in (('step-2cm', 0.02), ('step-4cm', 0.04), ('step-6cm', 0.06)):
            box, = (entity for entity in plan.scenes[name]['entities'] if entity['type'] == 'box')
            self.assertTrue(box['fixed'])
            self.assertAlmostEqual(box['position'][0] - box['size'][0] / 2, 1.0)
            self.assertAlmostEqual(box['position'][2] - box['size'][2] / 2, 0.0)
            self.assertAlmostEqual(box['position'][2] + box['size'][2] / 2, height)
        records = output_records(plan, 'nominal')
        self.assertEqual([str(path) for path in records], [f'{name}/nominal' for name in scenes])
        for case in records.values():
            trial, = case.trials
            self.assertEqual(case.scene_config({}, trial)['scene'], plan.scenes[trial.scene])

    def test_execute_plan_reuses_environment_until_scene_changes(self):
        plan = TASKS['terrain'].build_plan()
        flat_trial = plan.trials[0]
        second_motion = replace(flat_trial.motion, name='second_motion')
        plan = replace(plan, trials=(flat_trial, replace(flat_trial, motion=second_motion), *plan.trials[1:]))
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
        self.assertEqual(report['by_scene']['flat'], {'trials': 2, 'falls': 0})

    def test_explicit_conditions_and_motion_lists(self):
        standard = TASKS['standard'].build_plan()
        low = TASKS['low-friction'].build_plan()
        switch = TASKS['friction-switch'].build_plan()
        rollout = TASKS['rollout'].build_plan()
        self.assertEqual(len(standard.trials), 81)
        self.assertEqual(len(low.trials), 90)
        self.assertEqual(len(switch.trials), 72)
        self.assertEqual(len(rollout.trials), 1)
        self.assertEqual(rollout.trials[0].condition, standard.trials[0].condition)
        expected_sliding = {
            'fixed': [0.8, 0.8, 0.8, 0.8],
            'uniform-01': [0.1, 0.1, 0.1, 0.1],
            'uniform-02': [0.2, 0.2, 0.2, 0.2],
            'left-low': [0.1, 0.6, 0.2, 0.8],
            'right-low': [0.6, 0.1, 0.8, 0.2],
        }
        for name, values in expected_sliding.items():
            trials = [trial for trial in low.trials if trial.condition.name == f'nominal-{name}']
            self.assertEqual([trial.motion for trial in trials], [trial.motion for trial in switch.trials[:6]])
            feet = trials[0].condition.foot_ground_friction['base_ranges']
            self.assertEqual([feet[foot]['sliding'][0] for foot in ('FL', 'FR', 'RL', 'RR')], values)
        self.assertEqual(low.trials[0].condition.base_mass, standard.trials[0].condition.base_mass)
        source = {'domain_randomization': {'base_mass': {'enabled': False}, 'unrelated': {}}}
        effective = low.trials[0].condition.apply(source)
        self.assertTrue(effective['domain_randomization']['base_mass']['enabled'])
        self.assertNotIn('unrelated', effective['domain_randomization'])
        self.assertFalse(source['domain_randomization']['base_mass']['enabled'])

    def test_physical_samples_are_reproducible_and_shared(self):
        for seed, sample in enumerate(PHYSICAL_SAMPLES[1:], 1):
            self.assertEqual(sample, PhysicalSample.sample(seed))
            self.assertTrue(0.9 <= sample.mass_scale <= 1.1)
            self.assertTrue(0.75 <= sample.friction_multiplier <= 1.25)
            for offset, limit in zip(sample.com_offset_m, (0.01, 0.03, 0.01), strict=True):
                self.assertLessEqual(abs(offset), limit)
            baseline = next(condition for condition in TASKS['standard'].conditions
                            if condition.name == sample.name)
            for task_name in ('low-friction', 'friction-switch'):
                conditions = [condition for condition in TASKS[task_name].conditions
                              if condition.name.startswith(f'{sample.name}-')]
                self.assertTrue(conditions)
                for condition in conditions:
                    self.assertEqual(condition.base_mass, baseline.base_mass)
                    self.assertEqual(condition.base_com, baseline.base_com)

    def test_body_parameter_sweeps_hold_other_parameters_fixed(self):
        mass = TASKS['mass']
        com = TASKS['com-offset']
        self.assertEqual(len(mass.build_plan().trials), 42)
        self.assertEqual(len(com.build_plan().trials), 84)
        for task in (mass, com):
            self.assertEqual(task.motions, TASKS['low-friction'].motions)
            self.assertEqual(task.environment_task, 'flat_walk')
            for condition in task.conditions:
                self.assertEqual(condition.foot_ground_friction, TASKS['rollout'].conditions[0].foot_ground_friction)
                self.assertEqual(condition.friction_phases, ())
        self.assertEqual(
            [condition.base_mass['scale_range'] for condition in mass.conditions],
            [[scale, scale] for scale in (0.50, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50)],
        )
        for condition in mass.conditions:
            self.assertEqual(condition.base_com['offset_ranges_m'], dict.fromkeys('xyz', [0.0, 0.0]))
        offsets = []
        for condition in com.conditions:
            self.assertEqual(condition.base_mass['scale_range'], [1.0, 1.0])
            ranges = condition.base_com['offset_ranges_m']
            for low, high in ranges.values():
                self.assertEqual(low, high)
            offsets.append(tuple(ranges[axis][0] for axis in 'xyz'))
        self.assertCountEqual(offsets, [
            (0, 0, 0), (-0.01, 0, 0), (0.01, 0, 0), (-0.03, 0, 0), (0.03, 0, 0),
            (-0.05, 0, 0), (0.05, 0, 0), (0, -0.03, 0), (0, 0.03, 0),
            (0, -0.04, 0), (0, 0.04, 0), (0, 0, -0.01), (0, 0, 0.01), (0.03, 0.03, 0.01),
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

    def test_point_push_pairs_share_placement_and_timing(self):
        task = TASKS['point-push']
        plan = task.build_plan()
        self.assertEqual(len(plan.trials) * len(task.default_seeds), 162)
        self.assertEqual(sum(trial.condition.point_push is None for trial in plan.trials), 6)
        geometry = SimpleNamespace(sample_points=lambda directions: directions + torch.rand(directions.shape))
        config = {'robot': {'base_link': 'base'}}
        for target, levels in (
            ('base', (
                ('limit', 20.0, 10), ('ood', 30.0, 10),
                ('80n', 80.0, 10), ('160n', 160.0, 10),
                ('80n-short', 80.0, 5), ('160n-short', 160.0, 5),
            )),
            ('foot', (('limit', 8.0, 10), ('ood', 12.0, 10))),
        ):
            events = []
            for level, magnitude, duration_steps in levels:
                trial = next(trial for trial in plan.trials
                             if trial.condition.name == f'{target}-{level}' and trial.name == 'walk_to_stop')
                event = sample_event(config, geometry, 0.02, trial, 1)
                force = torch.tensor(event.force_world_n)
                self.assertEqual((event.start_step, event.end_step), (500, 500 + duration_steps))
                torch.testing.assert_close(force.norm(), torch.tensor(magnitude))
                events.append((event, force / magnitude))
            boundary, direction = events[0]
            for event, event_direction in events[1:]:
                self.assertEqual(boundary.link_name, event.link_name)
                self.assertEqual(boundary.point_local_m, event.point_local_m)
                torch.testing.assert_close(direction, event_direction)

    def test_motion_and_friction_events_use_their_own_schedules(self):
        tracking = TASKS['standard'].build_plan()
        stop = next(trial.motion for trial in tracking.trials if trial.name == 'walk_to_stop')
        self.assertEqual(stop.command_schedule(0.02)[400], (0, 0, 0))
        switch = TASKS['friction-switch'].build_plan()
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
