"""Exercise dashboard artifact layouts and its byte-range media endpoint."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from dashboard.app import create_app


class DashboardTest(unittest.TestCase):
    """Cover the current evaluation layout and the retained run variants."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'output'
        self.run = self.root / 'flat-supermimic-point-push-40k-08af176-20261124-123526'
        self.suite = self.run / 'evaluation' / 'iteration-40000' / 'terrain-mujoco-icc-20261124-175037'
        self.url = f'/runs/{self.run.name}/eval/40000/{self.suite.name}'
        self.write_json(self.run / 'training' / 'config.json', {'trainer': {'iterations': 40000}})
        self.client = TestClient(create_app(self.root))
        self.addCleanup(self.client.close)

    def write_json(self, path: Path, data: dict) -> None:
        """Write one synthetic report under the test output tree."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_nested_scene_report_and_video_range(self) -> None:
        """Serve the selected scene's video in ranges and link its report."""
        scene = self.suite / 'rough' / 'nominal'
        self.write_json(
            self.suite / 'summary.json',
            {'task': 'terrain', 'by_scene': {'rough': {'trials': 2, 'falls': 1}}},
        )
        self.write_json(scene / 'assessment.json', {
            'trials': 2, 'falls': 1,
            'per_case': [
                {'steps': 51, 'seconds': 1.02, 'fell': True, 'seed': 1, 'case': 'stand'},
                {'steps': 1000, 'seconds': 20, 'fell': False, 'seed': 2, 'case': 'walk'},
            ],
        })
        (scene / 'replay.mp4').write_bytes(b'0123456789')
        (scene / 'replay-preview.mp4').write_bytes(b'01234')

        listing = self.client.get(f'/runs/{self.run.name}')
        self.assertIn('1 段', listing.text)
        result = self.client.get(self.url, params={'case': 'rough/nominal'})
        self.assertEqual(result.status_code, 200)
        self.assertIn('rough/nominal', result.text)
        self.assertIn('data-hls-url', result.text)
        self.assertIn('省流', result.text)
        self.assertIn('preload="none"', result.text)
        self.assertIn('data-chapter-start="1.55"', result.text)
        self.assertIn('播放第 2 次测试', result.text)
        self.assertNotIn('<video src=', result.text)

        url = self.url + '/artifact/video'
        params = {'case': 'rough/nominal'}
        media = self.client.get(url, params=params, headers={'Range': 'bytes=2-5'})
        self.assertEqual(media.status_code, 206)
        self.assertEqual(media.headers['content-range'], 'bytes 2-5/10')
        self.assertEqual(media.headers['content-type'], 'video/mp4')
        self.assertEqual(media.content, b'2345')
        self.assertEqual(self.client.head(url, params=params).headers['accept-ranges'], 'bytes')
        self.assertEqual(self.client.get(url, params=params, headers={'Range': 'bytes=100-'}).status_code, 416)

    def test_unfinished_and_report_only_evaluation(self) -> None:
        """Ignore an in-progress replay and display a completed report alone."""
        self.write_json(self.suite / 'eval-config.json', {'task': 'terrain'})
        in_progress = self.suite / 'rough' / 'nominal'
        in_progress.mkdir(parents=True)
        (in_progress / 'replay.mp4').write_bytes(b'partial')
        completed = self.suite / 'flat' / 'nominal'
        self.write_json(completed / 'assessment.json', {'trials': 1, 'falls': 0})

        result = self.client.get(self.url)
        self.assertIn('flat/nominal', result.text)
        self.assertNotIn('rough/nominal', result.text)
        response = self.client.get(self.url + '/artifact/video', params={'case': 'rough/nominal'})
        self.assertEqual(response.status_code, 404)

    def test_retained_layouts(self) -> None:
        """Read a root assessment and a comparison video without a local report."""
        self.write_json(self.suite / 'assessment.json', {'trials': 3, 'falls': 2})
        comparison = self.suite / 'comparison'
        comparison.mkdir()
        (comparison / 'replay.mp4').write_bytes(b'video')
        result = self.client.get(self.url)
        self.assertIn('comparison', result.text)
        self.assertIn('评测根目录', result.text)
        report = self.client.get(self.url + '/artifact/assessment', params={'case': '.'})
        self.assertEqual(report.json()['falls'], 2)

    def test_legacy_run_has_html_error(self) -> None:
        """Keep an incompatible directory discoverable and explain its error."""
        (self.root / 'old-run').mkdir()
        result = self.client.get('/', params={'q': 'old'})
        self.assertIn('old-run', result.text)
        self.assertNotIn(self.run.name, result.text)
        error = self.client.get('/runs/old-run')
        self.assertEqual(error.status_code, 404)
        self.assertIn('缺少 training/config.json', error.text)

    def test_metrics_ignore_unflushed_line(self) -> None:
        """Continue serving a run while its latest metric row is being written."""
        file = self.run / 'training' / 'metrics.jsonl'
        file.write_text('{"iteration":1,"real_reward":0.25}\n{"iteration":2', encoding='utf-8')
        result = self.client.get(f'/runs/{self.run.name}')
        self.assertEqual(result.status_code, 200)
        self.assertIn('1 / 40000', result.text)
        self.assertIn('[1, 0.25]', result.text)
        self.assertNotIn('地形平均难度', result.text)

    def test_sparse_terrain_levels_are_charted_and_show_the_latest_value(self) -> None:
        """Show the newest terrain sample per interval even between sampled iterations."""
        rows = [
            {'iteration': 107, 'terrain/mean_level': 1.25},
            {'iteration': 110, 'terrain/mean_level': 1.5},
            {'iteration': 158, 'terrain/mean_level': 2.75},
            {'iteration': 200, 'real_reward': 0.25},
        ]
        file = self.run / 'training' / 'metrics.jsonl'
        file.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        result = self.client.get(f'/runs/{self.run.name}')
        self.assertEqual(result.status_code, 200)
        self.assertIn('2.75 / 9', result.text)
        self.assertIn('<option value="terrain/mean_level">地形平均难度</option>', result.text)
        self.assertIn('"terrain/mean_level": [[110, 1.5], [158, 2.75]]', result.text)
