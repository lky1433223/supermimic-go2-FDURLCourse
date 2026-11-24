"""Build and navigate the static terrain catalog used by playback."""

from itertools import zip_longest
from pathlib import Path

from rl_go2.config import validate_scene
from rl_go2.training_scene import build_training_scene


def playback_scene(scene: dict, robot: dict, seed: int, repo: Path) -> dict:
    """Use saved terrain patches, or add one standard patch per terrain to a legacy scene."""
    if scene.get('terrain_patches'):
        return scene

    robot_entity = next(entity for entity in scene['entities'] if entity.get('robot'))
    spec = {
        'sim_options': scene['sim_options'],
        'terrains': {kind: {'map_count': 1} for kind in ('flat', 'step', 'ramp', 'rough_ground')},
    }
    robot_path = repo / 'configs/robots/go2.yaml'
    generated = build_training_scene(spec, {**robot, 'file': robot_entity['file']}, robot_path, seed)
    # Keep the original scene at its authored origin and reuse its infinite ground plane.
    patches = generated['terrain_patches']
    patches[0].update(position=robot_entity['position'], rotation=robot_entity['rotation'])
    if any(entity['type'] == 'box' for entity in scene['entities']):
        patches[0].update(name='saved-scene', type='saved-scene')
    combined = {
        **scene,
        'entities': [*scene['entities'], *(e for e in generated['entities'] if e['type'] == 'box')],
        'terrain_patches': patches,
        'height_maps': {**scene.get('height_maps', {}), **generated['height_maps']},
    }
    return validate_scene(combined, robot_path)


def terrain_order(patches: list[dict]) -> list[int]:
    """Visit every terrain type before visiting the next variation of each type."""
    by_type: dict[str, list[int]] = {}
    for index, patch in enumerate(patches):
        by_type.setdefault(patch['type'], []).append(index)
    flat = by_type.pop('flat', [])
    return [index for row in zip_longest(flat, *by_type.values()) for index in row if index is not None]
