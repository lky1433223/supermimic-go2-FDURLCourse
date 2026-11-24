"""Evaluation plan composition and automatic-reset execution boundaries."""
from types import SimpleNamespace
import unittest

import torch

from rl_go2.evaluation.conditions import EvaluationCondition
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
        self.assertEqual(len(low.trials), 108)
        expected_sliding = {
            'uniform-01': [0.1, 0.1, 0.1, 0.1],
            'uniform-02': [0.2, 0.2, 0.2, 0.2],
            'left-low': [0.1, 0.6, 0.2, 0.8],
            'right-low': [0.6, 0.1, 0.8, 0.2],
        }
        for name, values in expected_sliding.items():
            trials = [trial for trial in low.trials if trial.condition.name == name]
            self.assertEqual([trial.motion for trial in trials], [trial.motion for trial in standard.trials])
            feet = trials[0].condition.foot_ground_friction['base_ranges']
            self.assertEqual([feet[foot]['sliding'][0] for foot in ('FL', 'FR', 'RL', 'RR')], values)
        self.assertEqual(low.trials[0].condition.base_mass, standard.trials[0].condition.base_mass)
        source = {'domain_randomization': {'base_mass': {'enabled': False}, 'unrelated': {}}}
        effective = low.trials[0].condition.apply(source)
        self.assertTrue(effective['domain_randomization']['base_mass']['enabled'])
        self.assertNotIn('unrelated', effective['domain_randomization'])
        self.assertFalse(source['domain_randomization']['base_mass']['enabled'])

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

    def test_motion_and_friction_events_use_their_own_schedules(self):
        tracking = TASKS['standard'].build_plan()
        stop = next(trial.motion for trial in tracking.trials if trial.name == 'walk_to_stop')
        self.assertEqual(stop.command_schedule(0.02)[400], (0, 0, 0))
        switch = TASKS['friction-switch'].build_plan()
        schedule = switch.trials[1].condition.phase_schedule(0.02)
        self.assertEqual({step: phase.multiplier for step, phase in schedule.items()},
                         {0: 1.0, 250: 0.5, 500: 1.0})


if __name__ == '__main__':
    unittest.main()
