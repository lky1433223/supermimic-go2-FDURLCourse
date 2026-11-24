"""Build replay navigation from ordered, completed evaluation trials."""

import math

from dashboard.artifacts import finite_number
from dashboard.trial_inputs import trial_inputs


def trial_chapters(report: dict, metadata: dict) -> list[dict]:
    """Map tracking trials to MP4 time using the current recorder's frame rules.

    The recorder emits the final step even between scheduled frames and holds
    a fall frame for half a second. Summing simulation seconds would lose both.
    Rollout reports lack per-reset frame positions, so they cannot be indexed.
    """
    rows = report.get('per_case')
    encoding = metadata.get('video_encoding', {})
    fps = encoding.get('fps', 20) if isinstance(encoding, dict) else 20
    if not isinstance(rows, list) or not finite_number(fps) or fps <= 0:
        return []
    chapters = []
    frame_offset = 0
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            return []
        seconds, steps, fell = row.get('seconds'), row.get('steps'), row.get('fell')
        if (
            not finite_number(seconds) or seconds <= 0
            or type(steps) is not int or steps <= 0 or type(fell) is not bool
        ):
            return []
        dt = seconds / steps
        final_frame = math.floor(seconds * fps + 1e-9)
        previous_frame = math.floor((steps - 1) * dt * fps + 1e-9)
        frame_count = final_frame + int(final_frame == previous_frame)
        if fell:
            frame_count += int(fps) // 2
        recovery = row.get('push_recovery', {})
        outcome = recovery.get('status') if isinstance(recovery, dict) else None
        labels = {
            'recovered': '已恢复', 'not_recovered': '未恢复',
            'fell_after_push_onset': '受推后摔倒', 'fell_before_push': '受推前摔倒',
        }
        start = frame_offset / fps
        minutes, remainder = divmod(start, 60)
        chapters.append({
            'number': number, 'start': start,
            'time': f'{int(minutes):02d}:{remainder:05.2f}',
            'seed': row.get('seed', '—'), 'condition': row.get('condition', ''),
            'motion': row.get('case', ''), 'fell': fell,
            'result': labels.get(outcome, '摔倒' if fell else '完成'),
            'inputs': trial_inputs(row, metadata),
        })
        frame_offset += frame_count
    return chapters
