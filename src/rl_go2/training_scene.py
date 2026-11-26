"""Build a static collection of terrains for weighted episode resets."""

from copy import deepcopy
import math
from pathlib import Path
from random import Random

import yaml

from rl_go2.scene_geometry import resolve_terrain_entities
from rl_go2.surround_terrain import build_curriculum_patches


def build_training_scene(spec: dict, robot: dict, robot_path: Path, seed: int) -> dict:
    """Expand terrain categories into separated patches and weighted spawn points."""
    if spec.keys() - {'sim_options', 'terrains', 'terrain_curriculum'}:
        raise ValueError('Unsupported training scene settings')
    terrains = spec.get('terrains', {})
    if not terrains or terrains.keys() - {'flat', 'step', 'stairs', 'ramp', 'rough_ground'}:
        raise ValueError('Choose at least one supported training terrain')
    if 'terrain_curriculum' in spec:
        geometry, patches, maps = build_curriculum_patches(terrains, spec['terrain_curriculum'], seed)
        return _assemble_scene(spec, robot, robot_path, geometry, patches, maps)

    scene_dir = Path(__file__).resolve().parents[2] / 'configs/scenes'
    entities = []
    patches = []
    maps = {}
    rng = Random(seed ^ 0x54455252)
    filenames = {'step': 'single_step.yaml', 'stairs': 'stairs.yaml',
                 'ramp': 'ramp.yaml', 'rough_ground': 'rough_ground.yaml'}
    for kind, parameters in terrains.items():
        if not isinstance(parameters, dict):
            raise ValueError('Terrain parameters must be a mapping')
        parameters = deepcopy(parameters)
        weight = parameters.pop('weight', 1.0)
        count = parameters.pop('map_count', 1 if kind == 'flat' else 8)
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Terrain weights must be finite and positive')
        if type(count) is not int or count < 1:
            raise ValueError('Terrain map_count must be a positive integer')
        template = None
        if kind != 'flat':
            with (scene_dir / filenames[kind]).open() as stream:
                template = next(entity for entity in yaml.safe_load(stream)['scene']['entities']
                                if entity['type'] == kind)
        range_key = {'step': 'height_range_m', 'ramp': 'slope_range_deg'}.get(kind)
        bounds = parameters.pop(range_key, None) if range_key else None
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2 or
                                   any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds)
                                   or bounds[0] > bounds[1]):
            raise ValueError(f'Invalid {range_key}')
        if kind == 'flat' and parameters:
            raise ValueError('Flat terrain supports weight and map_count')
        if parameters.keys() & {'id', 'type', 'center_y_m'}:
            raise ValueError('Terrain identity and lateral placement are assigned by the scene builder')
        for index in range(count):
            # Separation keeps neighboring patches outside the normal local scan.
            y = len(patches) * 20.0
            name = f'{kind}-{index}'
            patches.append({'name': name, 'type': kind, 'weight': weight / count,
                            'position': [0.0, y, 0.42], 'rotation': [1, 0, 0, 0]})
            if template is None:
                continue
            entity = {**template, **parameters, 'id': name, 'center_y_m': y}
            if bounds is not None:
                entity['height_m' if kind == 'step' else 'slope_deg'] = rng.uniform(*bounds)
            if kind == 'rough_ground':
                entity['seed'] = parameters.get('seed', seed) + index
            geometry, height_maps = resolve_terrain_entities([entity], {})
            entities.extend(geometry)
            maps.update(height_maps)
    return _assemble_scene(spec, robot, robot_path, entities, patches, maps)


def _assemble_scene(
    spec: dict, robot: dict, robot_path: Path, geometry: list[dict], patches: list[dict], height_maps: dict,
) -> dict:
    ground = {'id': 'ground', 'type': 'plane', 'position': [0, 0, 0],
              'rotation': [1, 0, 0, 0], 'fixed': True, 'friction': 0.8}
    model_file = (robot_path.parent / robot['file']).resolve()
    robot_entity = {'id': robot['entity_id'], 'type': 'mjcf', 'robot': True, 'file': str(model_file),
                    'position': [0, 0, 0.42], 'rotation': [1, 0, 0, 0]}
    scene = {'sim_options': spec['sim_options'], 'entities': [ground, *geometry, robot_entity], 'sensors': [],
             'terrain_patches': patches, 'height_maps': height_maps}
    if 'terrain_curriculum' in spec:
        scene['terrain_curriculum'] = deepcopy(spec['terrain_curriculum'])
    return scene
