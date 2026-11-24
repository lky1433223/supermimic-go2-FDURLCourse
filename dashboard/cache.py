"""Locate disposable HLS derivatives separately from experiment artifacts."""

import hashlib
import json
import os
from pathlib import Path


def cache_root() -> Path:
    """Return the shared cache location for the web app and converter."""
    configured = os.environ.get('GO2_DASHBOARD_CACHE')
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return (base / 'rl-go2-dashboard' / 'hls').expanduser().resolve()


def bundle_directory(source: Path, seconds: float | None = None) -> Path:
    """Key a bundle by source location, file version and encoding settings.

    Limited-duration conversions use a separate key from full replays.
    Updating a source file automatically invalidates its previous bundle.
    """
    source = source.resolve(strict=True)
    stat = source.stat()
    identity = [str(source), stat.st_size, stat.st_mtime_ns, 'h264-360-400-720-1400-v1', seconds]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return cache_root() / key
