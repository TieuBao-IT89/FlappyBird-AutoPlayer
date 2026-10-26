"""Versioned visual calibration profiles. Coordinates use normalized CSS pixels."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit


def _integers(value, count, name, minimum=0):
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(type(v) is not int or v < minimum for v in value)):
        raise ValueError(f'{name}: expected {count} integers >= {minimum}')
    return tuple(value)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{name}: expected fields {", ".join(expected)}')


@dataclass(frozen=True)
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self):
        for name in ('lower', 'upper'):
            values = _integers(getattr(self, name), 3, name)
            if any(v > maximum for v, maximum in zip(values, (179, 255, 255))):
                raise ValueError(f'{name}: H must be 0..179, S/V must be 0..255')
            object.__setattr__(self, name, values)
        if any(self.lower[i] > self.upper[i] for i in (1, 2)):
            raise ValueError('HSV: S/V lower bounds must not exceed upper bounds')
        # H lower > upper explicitly represents a range crossing red (179 -> 0).

    @classmethod
    def from_dict(cls, data):
        _keys(data, ('lower', 'upper'), 'HSV range')
        return cls(data['lower'], data['upper'])


@dataclass(frozen=True)
class TrackingConfig:
    min_bird_confidence: float = .75
    min_pipe_confidence: float = .60
    prediction_seconds: float = .12
    pipe_memory_seconds: float = .30
    reacquire_seconds: float = .05
    expand_search_seconds: float = .30
    lost_seconds: float = .80
    max_frame_age: float = .20
    bird_x_tolerance: float = .05
    max_bird_speed: float = 2.0
    max_gap_change: float = .04
    max_pipe_shift: float = .06
    motion_window_seconds: float = .30
    stationary_bird_seconds: float = .22
    pipe_motion_seconds: float = .12

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'tracking.{key} must be a finite positive number')
        for key in ('min_bird_confidence', 'min_pipe_confidence', 'bird_x_tolerance',
                    'max_gap_change', 'max_pipe_shift'):
            if getattr(self, key) > 1:
                raise ValueError(f'tracking.{key} must be <= 1')
        if not (self.prediction_seconds <= self.expand_search_seconds <= self.lost_seconds
                and self.pipe_memory_seconds <= self.lost_seconds
                and self.reacquire_seconds < self.lost_seconds
                and self.max_frame_age <= self.lost_seconds
                and max(self.stationary_bird_seconds, self.pipe_motion_seconds) <= self.motion_window_seconds <= self.lost_seconds):
            raise ValueError('tracking: prediction/search/memory/frame age must fit within lost_seconds')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError('tracking: unknown configuration fields')
        return cls(**data)


@dataclass(frozen=True)
class VisionConfig:
    bird: HSVRange = field(default_factory=lambda: HSVRange((13, 75, 90), (33, 255, 255)))
    pipe: HSVRange = field(default_factory=lambda: HSVRange((38, 65, 45), (92, 255, 255)))
    beak: HSVRange = field(default_factory=lambda: HSVRange((170, 110, 80), (12, 255, 255)))
    eye: HSVRange = field(default_factory=lambda: HSVRange((0, 0, 210), (179, 85, 255)))
    use_beak: bool = True
    use_eye: bool = True
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self):
        if any(not isinstance(getattr(self, key), HSVRange) for key in ('bird', 'pipe', 'beak', 'eye')):
            raise ValueError('Vision colors must be HSV ranges')
        if type(self.use_beak) is not bool or type(self.use_eye) is not bool:
            raise ValueError('use_beak/use_eye must be booleans')
        if not isinstance(self.tracking, TrackingConfig):
            raise ValueError('tracking must be a TrackingConfig')

    @classmethod
    def from_dict(cls, data):
        required = ('bird', 'pipe', 'beak', 'eye', 'use_beak', 'use_eye')
        _keys(data, required + (('tracking',) if isinstance(data, dict) and 'tracking' in data else ()), 'vision')
        return cls(**{key: HSVRange.from_dict(data[key]) for key in ('bird', 'pipe', 'beak', 'eye')},
                   use_beak=data['use_beak'], use_eye=data['use_eye'],
                   tracking=TrackingConfig.from_dict(data.get('tracking', {})))


@dataclass(frozen=True)
class GameProfile:
    version: int = 1
    source: str = 'website'
    url: str = 'https://flappybird.io/'
    selector: str = 'canvas'
    viewport: tuple[int, int] = (480, 720)
    roi: tuple[int, int, int, int] | None = None
    reference_size: tuple[int, int] | None = None
    region: str = 'element'
    vision: VisionConfig = field(default_factory=VisionConfig)

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError('Unsupported profile version; expected 1')
        if self.source not in ('website', 'demo') or self.region not in ('element', 'viewport'):
            raise ValueError('Invalid profile source or region')
        if not isinstance(self.url, str) or urlsplit(self.url).scheme not in ('http', 'https') or not urlsplit(self.url).netloc:
            raise ValueError('Profile URL must be an http(s) URL')
        if not isinstance(self.selector, str) or not self.selector.strip():
            raise ValueError('Profile selector must not be empty')
        if not isinstance(self.vision, VisionConfig):
            raise ValueError('Profile vision must be a VisionConfig')
        object.__setattr__(self, 'viewport', _integers(self.viewport, 2, 'viewport', 1))
        if self.reference_size is not None:
            object.__setattr__(self, 'reference_size', _integers(self.reference_size, 2, 'reference_size', 1))
        if self.roi is not None:
            object.__setattr__(self, 'roi', _integers(self.roi, 4, 'roi'))
            if self.reference_size is None:
                raise ValueError('ROI requires reference_size')
            validate_roi(self.roi, self.reference_size)

    def validate_capture(self, size, region):
        if region != self.region or (self.reference_size is not None and tuple(size) != self.reference_size):
            raise ValueError(f'Profile expects {self.region} {self.reference_size}, received {region} {tuple(size)}. '
                             'Run --calibrate again for this layout.')

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
                temporary = stream.name
                json.dump(asdict(self), stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: Path):
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
            _keys(data, tuple(cls.__dataclass_fields__), 'profile')
            return cls(**{**data, 'vision': VisionConfig.from_dict(data['vision'])})
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f'Cannot load profile {path}: {exc}') from exc


def validate_roi(roi, size):
    x, y, width, height = _integers(roi, 4, 'roi')
    if width <= 0 or height <= 0 or x + width > size[0] or y + height > size[1]:
        raise ValueError(f'ROI {roi} outside screenshot dimensions {size[0]}x{size[1]}')
