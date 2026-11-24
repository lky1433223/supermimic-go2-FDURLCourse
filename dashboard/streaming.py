"""Generate independently seekable HLS segments on playback requests."""

import fcntl
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from dashboard.cache import bundle_directory, cache_root

SEGMENT_SECONDS = 4
QUALITIES = {'low': (360, 400), 'high': (720, 1400)}


class TranscodeError(Exception):
    """A requested video segment could not be prepared."""


def run_media(command: list[str]) -> str:
    """Run a bounded media operation and retain diagnostic errors."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    except FileNotFoundError as exc:
        raise TranscodeError('服务器需要安装 ffmpeg 和 ffprobe。') from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscodeError('转码超时，请重试或播放原版。') from exc
    except subprocess.CalledProcessError as exc:
        raise TranscodeError(f'视频处理失败 {exc.stderr[-800:]}') from exc
    return result.stdout


@lru_cache(maxsize=128)
def probe_duration(source: str, size: int, modified_ns: int) -> float:
    """Probe a source version, caching metadata without opening video on page load."""
    result = run_media([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=duration', '-of', 'json', source,
    ])
    try:
        duration = float(json.loads(result)['streams'][0]['duration'])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise TranscodeError('无法读取视频时长。') from exc
    if not math.isfinite(duration) or duration <= 0:
        raise TranscodeError('视频时长无效。')
    return duration


def duration_seconds(source: Path) -> float:
    """Return the first video stream's duration in seconds."""
    stat = source.stat()
    return probe_duration(str(source.resolve()), stat.st_size, stat.st_mtime_ns)


def variant_playlist(duration: float) -> str:
    """Advertise the full seekable timeline before any segments are encoded."""
    lines = [
        '#EXTM3U', '#EXT-X-VERSION:3', '#EXT-X-TARGETDURATION:4',
        '#EXT-X-MEDIA-SEQUENCE:0', '#EXT-X-PLAYLIST-TYPE:VOD',
        '#EXT-X-INDEPENDENT-SEGMENTS',
    ]
    for index in range(math.ceil(duration / SEGMENT_SECONDS)):
        length = min(SEGMENT_SECONDS, duration - index * SEGMENT_SECONDS)
        lines.extend([f'#EXTINF:{length:.6f},', f'segment-{index:05d}.ts'])
    return '\n'.join([*lines, '#EXT-X-ENDLIST', ''])


def master_playlist() -> str:
    """Return the two adaptive quality levels."""
    return (
        '#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-INDEPENDENT-SEGMENTS\n'
        '#EXT-X-STREAM-INF:BANDWIDTH=480000,NAME="省流"\nlow/index.m3u8\n'
        '#EXT-X-STREAM-INF:BANDWIDTH=1680000,NAME="清晰"\nhigh/index.m3u8\n'
    )


@contextmanager
def transcode_slot(root: Path) -> Iterator[None]:
    """Serialize requests across web workers, deduplicating and limiting CPU use."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.transcode.lock').open('a') as lock:
        deadline = time.monotonic() + 60
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TranscodeError('转码繁忙，请稍后重试。')
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prune_segments(root: Path) -> None:
    """Evict least recently used on-demand segments under the shared lock."""
    try:
        budget = int(os.environ.get('GO2_DASHBOARD_CACHE_MB', '2048')) * 1024 * 1024
    except ValueError as exc:
        raise TranscodeError('GO2_DASHBOARD_CACHE_MB 必须是正整数。') from exc
    if budget <= 0:
        raise TranscodeError('GO2_DASHBOARD_CACHE_MB 必须是正整数。')
    files = sorted(root.glob('*/*/*.ts'), key=lambda path: path.stat().st_mtime_ns)
    size = sum(path.stat().st_size for path in files)
    for path in files:
        if size <= budget:
            break
        size -= path.stat().st_size
        path.unlink()


def encode_segment(source: Path, target: Path, quality: str, index: int, duration: float) -> None:
    """Accurately seek and encode at most four seconds with a global PTS offset."""
    height, bitrate = QUALITIES[quality]
    offset = index * SEGMENT_SECONDS
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.segment-', dir=target.parent) as temporary:
        work = Path(temporary) / 'segment.ts'
        run_media([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-ss', str(offset), '-threads', '2', '-i', str(source),
            '-t', str(min(SEGMENT_SECONDS, duration - offset)),
            '-map', '0:v:0', '-an',
            '-filter_threads', '1', '-vf', f'scale=-2:min({height}\\,ih),fps=20',
            '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '8',
            '-pix_fmt', 'yuv420p', '-b:v', f'{bitrate}k',
            '-maxrate', f'{bitrate}k', '-bufsize', f'{bitrate * 2}k',
            '-g', '80', '-keyint_min', '80', '-sc_threshold', '0', '-bf', '0',
            '-output_ts_offset', str(offset), '-mpegts_copyts', '1',
            '-muxdelay', '0', '-f', 'mpegts', str(work),
        ])
        work.replace(target)


def segment_bytes(source: Path, quality: str, index: int) -> bytes:
    """Read or lazily transcode one segment, then enforce a bounded cache."""
    duration = duration_seconds(source)
    if quality not in QUALITIES or not 0 <= index < math.ceil(duration / SEGMENT_SECONDS):
        raise TranscodeError('分片编号无效。')
    root = cache_root() / 'on-demand-v1'
    version = bundle_directory(source).name
    target = root / version / quality / f'segment-{index:05d}.ts'
    with transcode_slot(root):
        if not target.is_file():
            encode_segment(source, target, quality, index, duration)
        if bundle_directory(source).name != version:
            target.unlink(missing_ok=True)
            raise TranscodeError('源视频发生变化，请刷新页面。')
        data = target.read_bytes()
        target.touch()
        prune_segments(root)
    return data
