"""Validate real FFmpeg segments, random seeks, reuse and cache eviction."""

import json
import shutil
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from dashboard.app import create_app
from dashboard.streaming import duration_seconds, segment_bytes


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class StreamingTest(unittest.TestCase):
    """Use a 13-second synthetic video to exercise actual media operations."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cache = self.directory / 'cache'
        environment = patch.dict('os.environ', {'GO2_DASHBOARD_CACHE': str(self.cache)})
        environment.start()
        self.addCleanup(environment.stop)
        self.run = self.directory / 'output' / 'smoke'
        self.case = self.run / 'evaluation' / 'iteration-1' / 'test' / 'nominal'
        self.case.mkdir(parents=True)
        (self.run / 'training').mkdir()
        (self.run / 'training' / 'config.json').write_text('{}')
        (self.case.parent / 'summary.json').write_text('{"task":"rollout"}')
        self.source = self.case / 'replay.mp4'
        subprocess.run([
            'ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'lavfi',
            '-i', 'testsrc2=size=640x480:rate=20', '-t', '13',
            '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p',
            str(self.source),
        ], check=True)

    def timestamps(self, data: bytes) -> tuple[float, int]:
        """Inspect a segment's presentation timeline and decoded frame count."""
        segment = self.directory / 'inspect.ts'
        segment.write_bytes(data)
        output = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=start_time,nb_read_frames', '-of', 'json',
            str(segment),
        ])
        stream = json.loads(output)['streams'][0]
        return float(stream['start_time']), int(stream['nb_read_frames'])

    def test_random_seek_and_cache_reuse(self) -> None:
        """Seek ahead without encoding earlier segments and keep continuous PTS."""
        self.assertAlmostEqual(duration_seconds(self.source), 13)
        third = segment_bytes(self.source, 'low', 2)
        self.assertEqual(len(list(self.cache.rglob('*.ts'))), 1)
        self.assertEqual(self.timestamps(third), (8, 80))
        with patch('dashboard.streaming.encode_segment', side_effect=AssertionError('cache missed')):
            self.assertEqual(segment_bytes(self.source, 'low', 2), third)
        second = segment_bytes(self.source, 'high', 1)
        last = segment_bytes(self.source, 'low', 3)
        first = segment_bytes(self.source, 'low', 0)
        self.assertEqual(self.timestamps(second), (4, 80))
        self.assertEqual(self.timestamps(last), (12, 20))
        self.assertEqual(self.timestamps(first), (0, 80))
        self.assertEqual(self.timestamps(first + second + third + last), (0, 260))

    def test_lazy_routes_and_error_response(self) -> None:
        """Loading a page or playlist never transcodes; segment requests do."""
        url = '/runs/smoke/eval/1/test'
        with TestClient(create_app(self.run.parent)) as client:
            self.assertIn('data-hls-url', client.get(url).text)
            master = client.get(url + '/hls/master.m3u8?case=nominal')
            self.assertIn('low/index.m3u8?case=nominal', master.text)
            playlist = client.get(url + '/hls/low/index.m3u8?case=nominal')
            self.assertIn('segment-00003.ts?case=nominal', playlist.text)
            self.assertFalse(self.cache.exists())
            response = client.get(url + '/hls/low/segment-00002.ts?case=nominal')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.timestamps(response.content), (8, 80))
            self.assertEqual(
                client.get(url + '/hls/low/segment-00004.ts?case=nominal').status_code, 404,
            )
            with patch('dashboard.streaming.subprocess.run', side_effect=FileNotFoundError):
                self.assertEqual(
                    client.get(url + '/hls/high/segment-00000.ts?case=nominal').status_code, 503,
                )

    def test_duplicate_requests_and_eviction(self) -> None:
        """Concurrent identical requests share a segment and old files are evicted."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: segment_bytes(self.source, 'low', 0), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(list(self.cache.rglob('*.ts'))), 1)
        old = self.cache / 'on-demand-v1' / 'old' / 'low' / 'segment-00000.ts'
        old.parent.mkdir(parents=True)
        old.write_bytes(b'0' * (2 * 1024 * 1024))
        old.touch()
        with patch.dict('os.environ', {'GO2_DASHBOARD_CACHE_MB': '1'}):
            segment_bytes(self.source, 'low', 0)
        self.assertFalse(old.exists())
