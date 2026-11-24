"""Evaluation plan composition and automatic-reset execution boundaries."""
from types import SimpleNamespace
import unittest

import torch

from rl_go2.evaluation.push_events import sample_event
from rl_go2.evaluation.conditions import EvaluationCondition, PHYSICAL_SAMPLES, PhysicalSample
from rl_go2.evaluation.executor import execute_trial
from rl_go2.evaluation.motion_tasks import MotionTask
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
        self.assertEqual(len(plan.trials) * len(task.default_seeds), 90)
        self.assertEqual(sum(trial.condition.point_push is None for trial in plan.trials), 6)
        geometry = SimpleNamespace(sample_points=lambda directions: directions + torch.rand(directions.shape))
        config = {'robot': {'base_link': 'base'}}
        for target, limit in (('base', 20.0), ('foot', 8.0)):
            events = []
            for level in ('limit', 'ood'):
                trial = next(trial for trial in plan.trials
                             if trial.condition.name == f'{target}-{level}' and trial.name == 'walk_to_stop')
                events.append(sample_event(config, geometry, 0.02, trial, 1))
            boundary, outside = events
            self.assertEqual((boundary.start_step, boundary.end_step), (500, 510))
            self.assertEqual(boundary.start_step, outside.start_step)
            self.assertEqual(boundary.link_name, outside.link_name)
            self.assertEqual(boundary.point_local_m, outside.point_local_m)
            force = torch.tensor(boundary.force_world_n)
            self.assertAlmostEqual(force.norm().item(), limit, places=5)
            torch.testing.assert_close(torch.tensor(outside.force_world_n), 1.5 * force)

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
