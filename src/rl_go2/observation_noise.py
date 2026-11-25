"""Ordered sensor errors with explicit history for simulation and imagination."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Protocol

import torch

from rl_go2.observations import SensorReadings, project_gravity
from rl_go2.state import quat_mul
from rl_go2.step_delay import EpisodeStepDelay, StepDelayState


type NoiseHistory = dict[str, torch.Tensor]
type AxisValues = tuple[float, ...]

# Attitude errors are rotation vectors in body axes, not four quaternion offsets.
ERROR_WIDTHS = dict(angular_velocity=3, attitude=3, joint_position=12, joint_velocity=12, projected_gravity=3)


class SensorNoise(Protocol):
    def apply(self, reading: torch.Tensor, history: torch.Tensor | None, *,
              attitude: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the perturbed batch and any history needed for the next control step."""
        ...


class AdditiveNoise(ABC):
    @abstractmethod
    def error(self, reference: torch.Tensor) -> torch.Tensor:
        """Return a broadcastable error in the units of the channel."""

    def apply(self, reading: torch.Tensor, history: torch.Tensor | None, *,
              attitude: bool) -> tuple[torch.Tensor, None]:
        """Add a measurement error or compose an equivalent attitude rotation."""
        if not attitude:
            return reading + self.error(reading), None
        rotation = self.error(reading[..., 1:]).expand_as(reading[..., 1:])
        half_angle = 0.5 * torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
        delta = torch.cat((half_angle.cos(), 0.5 * torch.sinc(half_angle / math.pi) * rotation), dim=-1)
        return quat_mul(reading, delta), None


@dataclass(frozen=True)
class GaussianNoise(AdditiveNoise):
    std: AxisValues

    def error(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(reference) * reference.new_tensor(self.std)


@dataclass(frozen=True)
class UniformNoise(AdditiveNoise):
    low: AxisValues
    high: AxisValues

    def error(self, reference: torch.Tensor) -> torch.Tensor:
        low, high = reference.new_tensor(self.low), reference.new_tensor(self.high)
        return low + (high - low) * torch.rand_like(reference)


@dataclass(frozen=True)
class FixedBias(AdditiveNoise):
    offset: AxisValues

    def error(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(self.offset)


@dataclass(frozen=True)
class Delay:
    steps: int

    def apply(self, reading: torch.Tensor, history: torch.Tensor | None, *,
              attitude: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read the oldest sample and append this one to the delay line."""
        if self.steps == 0:
            return reading, None
        if history is None:
            history = reading[:, None].expand(-1, self.steps, -1).contiguous()
            return reading, history
        return history[:, 0], torch.cat((history[:, 1:], reading[:, None]), dim=1)


@dataclass(frozen=True)
class Dropout:
    probability: float

    def apply(self, reading: torch.Tensor, history: torch.Tensor | None, *,
              attitude: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop the whole channel for each sampled environment and hold its last value."""
        if history is None:
            return reading, reading
        dropped = torch.rand_like(reading[:, :1]) < self.probability
        held = torch.where(dropped, history, reading)
        return held, held


@dataclass(frozen=True)
class FrameDelay:
    """Hold one shared integer delay across all readings of an episode."""

    delay: EpisodeStepDelay

    def apply(self, readings: SensorReadings, history: NoiseHistory) -> tuple[SensorReadings, NoiseHistory]:
        """Sample the delay at reset, then return one frame of that age per environment."""
        frame = torch.cat(readings, dim=-1)
        if 'frame_delay/steps' not in history:
            state = self.delay.reset(frame)
            return readings, {'frame_delay/steps': state.steps, 'frame_delay/frames': state.history}

        state = StepDelayState(history['frame_delay/steps'], history['frame_delay/frames'])
        delayed = self.delay.select(frame, state)
        widths = [reading.shape[-1] for reading in readings]
        state = self.delay.advance(frame, state)
        return SensorReadings(*delayed.split(widths, dim=-1)), {
            'frame_delay/steps': state.steps, 'frame_delay/frames': state.history,
        }


def _axis_values(value: object, width: int, field: str) -> AxisValues:
    values = value if isinstance(value, (list, tuple)) else (value,) * width
    if len(values) != width or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError(f'{field} must be a finite scalar or a list of {width} finite values')
    return tuple(values)


def _parse_noise(spec: object, width: int, path: str) -> SensorNoise:
    if not isinstance(spec, dict):
        raise ValueError(f'{path} must specify a noise type and its parameters')
    match spec:
        case {'type': 'gaussian', 'std': std} if len(spec) == 2:
            values = _axis_values(std, width, f'{path}.std')
            if min(values) < 0:
                raise ValueError(f'{path}.std must be nonnegative')
            return GaussianNoise(values)
        case {'type': 'uniform', 'low': low, 'high': high} if len(spec) == 3:
            lower = _axis_values(low, width, f'{path}.low')
            upper = _axis_values(high, width, f'{path}.high')
            if any(a > b for a, b in zip(lower, upper, strict=True)):
                raise ValueError(f'{path}.low must be at most high on every axis')
            return UniformNoise(lower, upper)
        case {'type': 'bias', 'offset': offset} if len(spec) == 2:
            return FixedBias(_axis_values(offset, width, f'{path}.offset'))
        case {'type': 'delay', 'steps': steps} if len(spec) == 2 and type(steps) is int and steps >= 0:
            return Delay(steps)
        case {'type': 'dropout', 'probability': p} if len(spec) == 2 and type(p) in (int, float) and 0 <= p <= 1:
            return Dropout(p)
    raise ValueError(f'{path} needs a valid gaussian (std), uniform (low, high), bias (offset), '
                     'delay (nonnegative integer steps) or dropout (probability in [0, 1])')


class ObservationNoise:
    """Apply each channel's ordered noise chain to one batch per control step."""

    def __init__(self, chains: Mapping[str, tuple[SensorNoise, ...]], frame_delay: FrameDelay | None = None):
        self.chains = {channel: chains.get(channel, ()) for channel in ERROR_WIDTHS}
        self.frame_delay = frame_delay

    @classmethod
    def from_config(cls, config: dict) -> 'ObservationNoise | None':
        """Read optional chains from domain_randomization.observation_noise."""
        domain = config.get('domain_randomization', {})
        if not isinstance(domain, dict):
            raise ValueError('domain_randomization must be a mapping')
        spec = domain.get('observation_noise', {})
        if not isinstance(spec, dict) or spec.keys() - (ERROR_WIDTHS.keys() | {'frame_delay_steps'}):
            raise ValueError(f'observation_noise supports these channels: {", ".join(ERROR_WIDTHS)}, '
                             'and optional frame_delay_steps')
        delay = EpisodeStepDelay.from_bounds(spec.get('frame_delay_steps', [0, 0]),
                                             'observation_noise.frame_delay_steps')
        frame_delay = FrameDelay(delay) if delay is not None else None
        chains = {}
        for channel, width in ERROR_WIDTHS.items():
            entries = spec.get(channel, [])
            if not isinstance(entries, list):
                raise ValueError(f'observation_noise.{channel} must be an ordered list')
            chains[channel] = tuple(_parse_noise(entry, width, f'observation_noise.{channel}[{index}]')
                                    for index, entry in enumerate(entries))
        return cls(chains, frame_delay) if frame_delay is not None or any(chains.values()) else None

    def apply(self, readings: SensorReadings, history: NoiseHistory | None = None,
              ) -> tuple[SensorReadings, NoiseHistory]:
        """Advance one step without mutating input history. Omit history at reset."""
        previous = {} if history is None else history
        next_history = {}
        output = readings._asdict()
        for channel, chain in self.chains.items():
            reading = output[channel]
            for index, noise in enumerate(chain):
                key = f'{channel}/{index}'
                reading, memory = noise.apply(reading, previous.get(key), attitude=channel == 'attitude')
                if memory is not None:
                    next_history[key] = memory
            output[channel] = reading
            if channel == 'attitude' and chain:
                output['projected_gravity'] = project_gravity(reading)
        result = SensorReadings(**output)
        if self.frame_delay is not None:
            result, frame_history = self.frame_delay.apply(result, previous)
            next_history.update(frame_history)
        return result, next_history
