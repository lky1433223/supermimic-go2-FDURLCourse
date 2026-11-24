"""Describe recorded trial inputs without inferring them from case names."""

import math

from dashboard.artifacts import finite_number


def vector_text(value: object) -> str | None:
    """Format a recorded three-axis vector, retaining its component order."""
    if not isinstance(value, list) or len(value) != 3:
        return None
    if not all(finite_number(component) for component in value):
        return None
    return '(' + ', '.join(f'{component:.3g}' for component in value) + ')'


def trial_inputs(row: dict, metadata: dict) -> list[tuple[str, str]]:
    """Describe commands, sampled forces and measured physical parameters.

    Force events are matched by seed, condition and motion rather than row
    position: report subsets need not have the same order as the full plan.
    Event times are relative to the start of this trial, not the replay.
    """
    details = []
    segments = row.get('command_segments', row.get('segments', []))
    for segment in segments:
        command = vector_text(segment.get('command'))
        start = segment.get('start_seconds')
        if command is not None and finite_number(start):
            details.append((f'{start:g} s 起 · 速度指令',
                            f'{command} · vx / vy (m/s)，偏航角速度 (rad/s)'))
    seconds, steps = row.get('seconds'), row.get('steps')
    dt = seconds / steps if finite_number(seconds) and type(steps) is int and steps > 0 else None
    events = [event for event in metadata.get('point_push_events', [])
              if all(event.get(key) == row.get(key) for key in ('seed', 'condition', 'case'))]
    for event in events:
        force = event.get('force_world_n')
        force_text = vector_text(force)
        if force_text is None:
            continue
        details.append(('推力 · 世界坐标 XYZ (N)',
                        f'{force_text} · 合力 {math.sqrt(sum(v * v for v in force)):.3g} N'))
        details.append(('受力连杆', str(event.get('link_name', '未记录'))))
        point = vector_text(event.get('point_local_m'))
        if point is not None:
            details.append(('作用点 · 连杆局部 XYZ (m)', point))
        start, end = event.get('start_step'), event.get('end_step')
        if dt is not None and type(start) is int and type(end) is int:
            details.append(('施力时间 · 试次内',
                            f'{start * dt:g}–{end * dt:g} s · 持续 {(end - start) * dt:g} s'))
    if not events and metadata.get('task') == 'point-push':
        no_push = any(condition.get('name') == row.get('condition')
                      and condition.get('point_push') is None
                      for condition in metadata.get('physical_conditions', []))
        details.append(('推力', '无推力' if no_push else '具体事件未记录'))
    mass = row.get('base_mass_kg')
    if finite_number(mass):
        details.append(('机身质量', f'{mass:.4g} kg'))
    com = vector_text(row.get('base_com_offset_m'))
    if com is not None:
        details.append(('质心偏移 · 机身局部 XYZ (m)', com))
    friction = row.get('foot_ground_friction')
    if isinstance(friction, list) and len(friction) == 4:
        values = [vector_text(value) for value in friction]
        if all(value is not None for value in values):
            details.append(('足地摩擦 · 滑动 / 扭转 / 滚动',
                            '\n'.join(f'{foot} {value}' for foot, value in
                                     zip(('FR', 'FL', 'RR', 'RL'), values))))
    return details
