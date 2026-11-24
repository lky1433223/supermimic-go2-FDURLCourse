"""Check trial offsets against the recorder's terminal-frame rules."""

import unittest

from dashboard.chapters import trial_chapters


class ChapterTest(unittest.TestCase):
    def test_early_fall_and_off_grid_final_frame(self) -> None:
        """Include an extra terminal frame and the half-second fall hold."""
        rows = [
            {'steps': 1000, 'seconds': 20, 'fell': False, 'seed': 1, 'case': 'stand'},
            {'steps': 51, 'seconds': 1.02, 'fell': True, 'seed': 1, 'case': 'walk'},
            {'steps': 2, 'seconds': .04, 'fell': False, 'seed': 2, 'case': 'stand'},
            {'steps': 1000, 'seconds': 20, 'fell': False, 'seed': 2, 'case': 'walk'},
        ]
        chapters = trial_chapters({'per_case': rows}, {'video_encoding': {'fps': 20}})
        self.assertEqual([chapter['start'] for chapter in chapters], [0, 20, 21.55, 21.6])
        self.assertEqual(chapters[1]['result'], '摔倒')
        self.assertEqual(chapters[2]['seed'], 2)
        self.assertEqual(chapters[2]['time'], '00:21.55')

    def test_unknown_trial_timing_disables_navigation(self) -> None:
        """Avoid incorrect later offsets if any trial lacks its timing fields."""
        valid = {'steps': 100, 'seconds': 2, 'fell': False}
        report = {'per_case': [valid, {'steps': 100, 'seconds': 2, 'episode_resets': 1}]}
        self.assertEqual(trial_chapters(report, {}), [])
        self.assertEqual(trial_chapters({}, {}), [])
