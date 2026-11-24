"""Verify that replay details describe each recorded trial's actual inputs."""

import unittest

from dashboard.trial_inputs import trial_inputs


class TrialInputsTest(unittest.TestCase):
    """Keep force matching and time-dependent command details accurate."""

    def test_matches_sampled_force_and_stop_command(self) -> None:
        """Select the correct seed and convert event steps to trial seconds."""
        row = {
            'seed': 2, 'condition': 'foot-limit', 'case': 'walk_to_stop',
            'seconds': 20, 'steps': 1000,
            'segments': [
                {'start_seconds': 0, 'command': [.5, 0, 0]},
                {'start_seconds': 8, 'command': [0, 0, 0]},
            ],
            'base_mass_kg': 6.921, 'base_com_offset_m': [0, 0, 0],
            'foot_ground_friction': [[.1, .02, .01], [.2, .02, .01],
                                     [.3, .02, .01], [.4, .02, .01]],
        }
        event = {
            'seed': 2, 'condition': 'foot-limit', 'case': 'walk_to_stop',
            'start_step': 500, 'end_step': 510, 'link_name': 'RL_calf',
            'force_world_n': [0, 0, 8], 'point_local_m': [.1, 0, 0],
        }
        details = dict(trial_inputs(row, {'point_push_events': [
            {**event, 'seed': 1, 'force_world_n': [0, 0, 99]}, event,
        ]}))
        self.assertIn('8 s 起 · 速度指令', details)
        self.assertEqual(details['受力连杆'], 'RL_calf')
        self.assertIn('合力 8 N', details['推力 · 世界坐标 XYZ (N)'])
        self.assertEqual(details['施力时间 · 试次内'], '10–10.2 s · 持续 0.2 s')
        self.assertIn('FR (0.1,', details['足地摩擦 · 滑动 / 扭转 / 滚动'])
        self.assertIn('RL (0.4,', details['足地摩擦 · 滑动 / 扭转 / 滚动'])

    def test_missing_event_is_not_reported_as_no_push(self) -> None:
        """Distinguish a missing event from an explicitly unforced condition."""
        row = {'condition': 'base-limit'}
        details = dict(trial_inputs(row, {'task': 'point-push'}))
        self.assertEqual(details['推力'], '具体事件未记录')
        row['condition'] = 'no-push'
        details = dict(trial_inputs(row, {
            'task': 'point-push',
            'physical_conditions': [{'name': 'no-push', 'point_push': None}],
        }))
        self.assertEqual(details['推力'], '无推力')
