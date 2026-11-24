"""Read the Go2 training and evaluation files that the dashboard displays."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from dashboard.cache import bundle_directory


class ArtifactError(Exception):
    """An output directory or artifact cannot be shown."""


@dataclass(frozen=True)
class Evaluation:
    """One execution of an evaluation suite for a checkpoint."""

    path: Path
    iteration: int

    @property
    def name(self) -> str:
        """Return the evaluation directory name used in URLs."""
        return self.path.name


@dataclass(frozen=True)
class EvaluationCase:
    """Report and optional replay for a condition, possibly within a scene."""

    name: str
    assessment: Path | None
    video: Path | None
    preview: Path | None
    hls: Path | None


def local_file(path: Path) -> Path | None:
    """Return a regular file owned by this directory, without following links."""
    return path if not path.is_symlink() and path.is_file() else None


def child_directories(path: Path) -> list[Path]:
    """List real, non-hidden child directories in a stable order."""
    if path.is_symlink():
        return []
    try:
        return sorted(
            entry for entry in path.iterdir()
            if not entry.name.startswith('.') and not entry.is_symlink() and entry.is_dir()
        )
    except FileNotFoundError:
        return []


def read_json(path: Path) -> dict | None:
    """Read an optional JSON object, reporting malformed or unreadable files."""
    if local_file(path) is None:
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f'无法读取 {path.name}，文件可能仍在写入或格式无效。') from exc
    if not isinstance(value, dict):
        raise ArtifactError(f'{path.name} 必须是 JSON 对象。')
    return value


def run_directory(root: Path, name: str) -> Path:
    """Resolve a listed run and require the repository's training layout."""
    run = next((path for path in child_directories(root) if path.name == name), None)
    if run is None:
        raise ArtifactError('找不到这个实验目录。')
    training = run / 'training'
    if training.is_symlink() or local_file(training / 'config.json') is None:
        raise ArtifactError('该实验缺少 training/config.json，可能是旧版产物或训练尚未初始化。')
    return run


def evaluations(run: Path) -> list[Evaluation]:
    """Find evaluations using the repository's checkpoint directory layout."""
    items = []
    for directory in child_directories(run / 'evaluation'):
        match = re.fullmatch(r'iteration-(\d+)', directory.name)
        if match:
            items.extend(
                Evaluation(path, int(match[1])) for path in child_directories(directory)
            )
    return sorted(items, key=lambda item: (item.iteration, item.path.stat().st_mtime), reverse=True)


def evaluation_cases(evaluation: Evaluation, *, complete: bool) -> list[EvaluationCase]:
    """Find report and replay pairs at the suite, condition or scene/condition level."""
    root = evaluation.path
    directories = [root]
    for directory in child_directories(root):
        directories.append(directory)
        directories.extend(child_directories(directory))

    cases = []
    for directory in directories:
        assessment = local_file(directory / 'assessment.json')
        video = local_file(directory / 'replay.mp4')
        if not complete and assessment is None:
            # The runner writes each case's report after closing the MP4.
            video = None
        if assessment is None and video is None:
            continue
        name = directory.relative_to(root).as_posix()
        preview = local_file(directory / 'replay-preview.mp4') if video else None
        hls_dir = bundle_directory(video) if video else None
        hls = local_file(hls_dir / 'master.m3u8') if hls_dir and not hls_dir.is_symlink() else None
        cases.append(EvaluationCase(name, assessment, video, preview, hls))
    return cases


def trial_totals(report: dict) -> tuple[int | None, int | None]:
    """Return trial and fall counts for suite, condition or rollout reports."""
    groups = report.get('groups')
    if isinstance(groups, dict):
        totals = [trial_totals(group) for group in groups.values() if isinstance(group, dict)]
        if totals and all(trials is not None and falls is not None for trials, falls in totals):
            return sum(trials for trials, _ in totals), sum(falls for _, falls in totals)
    trials, falls = report.get('trials'), report.get('falls')
    if type(trials) is int and type(falls) is int:
        return trials, falls
    rows = report.get('per_case')
    if isinstance(rows, list) and all(
        isinstance(row, dict) and isinstance(row.get('fell'), bool) for row in rows
    ):
        return len(rows), sum(row['fell'] for row in rows)
    return None, None


def finite_number(value: object) -> bool:
    """Check whether a normal JSON numeric value is usable on a chart."""
    return type(value) in (int, float) and math.isfinite(value)


def training_progress(run: Path, budget: int | None) -> tuple[dict, dict[str, list[list]]]:
    """Read complete metric lines and sample chart points by iteration."""
    metrics = local_file(run / 'training' / 'metrics.jsonl')
    fields = ('real_reward', 'model_loss', 'reset_fraction')
    series = {field: [] for field in fields}
    if metrics is None:
        return {}, series

    latest = {}
    stride = max(1, math.ceil((budget or 40000) / 300))
    try:
        with metrics.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break  # A live writer may not have flushed the whole row yet.
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or type(row.get('iteration')) is not int:
                    raise ArtifactError('metrics.jsonl 缺少有效的训练轮次。')
                latest = row
                if row['iteration'] == 1 or row['iteration'] % stride == 0:
                    append_metrics(series, row)
    except (OSError, ValueError) as exc:
        raise ArtifactError('无法读取 metrics.jsonl，训练指标格式无效。') from exc
    if latest and latest['iteration'] != 1 and latest['iteration'] % stride:
        append_metrics(series, latest)
    return latest, series


def append_metrics(series: dict[str, list[list]], row: dict) -> None:
    """Append chart data for the reported finite metrics."""
    for field, points in series.items():
        value = row.get(field)
        if finite_number(value):
            points.append([row['iteration'], value])
