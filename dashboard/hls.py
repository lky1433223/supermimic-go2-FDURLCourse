"""Prepare two HLS quality levels from an existing evaluation replay."""

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from dashboard.cache import bundle_directory


def prepare(source: Path, seconds: float | None = None) -> Path:
    """Publish a completed conversion in the external disposable cache."""
    source = source.resolve(strict=True)
    destination = bundle_directory(source, seconds)
    if destination.exists():
        raise FileExistsError(f'HLS 目录已存在 {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.hls-', dir=destination.parent) as temporary:
        work = Path(temporary)
        for name, height, bitrate in [('low', 360, 400), ('high', 720, 1400)]:
            variant = work / name
            variant.mkdir()
            command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning', '-i', str(source)]
            if seconds is not None:
                command.extend(['-t', str(seconds)])
            command.extend([
                '-an', '-vf', f'scale=-2:min({height}\\,ih),fps=20',
                '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2',
                '-pix_fmt', 'yuv420p', '-b:v', f'{bitrate}k',
                '-maxrate', f'{bitrate}k', '-bufsize', f'{bitrate * 2}k',
                '-g', '80', '-keyint_min', '80', '-sc_threshold', '0',
                '-f', 'hls', '-hls_time', '4', '-hls_playlist_type', 'vod',
                '-hls_flags', 'independent_segments',
                '-hls_segment_filename', str(variant / 'segment-%05d.ts'),
                str(variant / 'index.m3u8'),
            ])
            subprocess.run(command, check=True)
        (work / 'master.m3u8').write_text(
            '#EXTM3U\n#EXT-X-VERSION:3\n'
            '#EXT-X-STREAM-INF:BANDWIDTH=480000,NAME="省流"\nlow/index.m3u8\n'
            '#EXT-X-STREAM-INF:BANDWIDTH=1680000,NAME="清晰"\nhigh/index.m3u8\n',
            encoding='utf-8',
        )
        if bundle_directory(source, seconds) != destination:
            raise RuntimeError('源视频在转码期间发生变化，请稍后重试。')
        work.rename(destination)
    return destination


def main() -> None:
    """Prepare one selected replay without launching an evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--seconds', type=float, help='Only convert this many seconds for a smoke check.')
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error('--seconds 必须大于 0')
    print(prepare(args.source, args.seconds))


if __name__ == '__main__':
    main()
