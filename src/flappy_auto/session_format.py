"""Version 1 replay sessions: lossless images and a streaming JSONL event log."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from .profiles import VisionConfig

SESSION_VERSION = 1


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def observation_dict(obs):
    # Canonicalize tuples as JSON arrays for exact recorded/recomputed comparison.
    return json.loads(json.dumps(asdict(obs), allow_nan=False))


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


@dataclass
class ReplaySession:
    directory: Path
    metadata: dict

    @classmethod
    def open(cls, directory):
        directory = Path(directory).resolve()
        try:
            metadata = json.loads((directory/'session.json').read_text(encoding='utf-8'))
            if (not isinstance(metadata, dict) or type(metadata.get('version')) is not int
                    or metadata['version'] != SESSION_VERSION or metadata.get('kind') != 'flappy-session'):
                raise ValueError('unsupported replay session format')
            VisionConfig.from_dict(metadata['vision'])
            size = metadata['frame_size']
            if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError('invalid frame_size')
            if metadata.get('recording_mode') not in ('full', 'rolling'):
                raise ValueError('invalid recording_mode')
            for key in ('code', 'mode', 'profile', 'recording_stats'):
                if key in metadata and not isinstance(metadata[key], dict):
                    raise ValueError(f'invalid {key}')
            for key in ('closed', 'complete_images'):
                if key in metadata and type(metadata[key]) is not bool:
                    raise ValueError(f'invalid {key}')
            for key, value in metadata.get('recording_stats', {}).items():
                if type(value) is not int or value < 0:
                    raise ValueError(f'invalid recording_stats.{key}')
            if not (directory/'events.jsonl').is_file():
                raise ValueError('events.jsonl is missing')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'Cannot load replay session {directory}: {exc}') from exc
        return cls(directory, metadata)

    def image_path(self, name):
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError('Replay image must be a relative path')
        target = (self.directory/name).resolve()
        if not target.is_relative_to(self.directory):
            raise ValueError('Replay image path escapes the session directory')
        return target

    def events(self, *, repair_timestamps=False, fallback_fps=30):
        if not finite_number(fallback_fps) or fallback_fps <= 0:
            raise ValueError('Fallback FPS must be finite and positive')
        previous_clock, previous_id = None, 0
        with (self.directory/'events.jsonl').open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get('type') not in ('frame', 'action'):
                        raise ValueError('unknown event type')
                    if event['type'] == 'action':
                        if not finite_number(event.get('sent_at')):
                            raise ValueError('invalid action timestamp')
                        yield event
                        continue
                    index = event['id']
                    if type(index) is not int or index <= previous_id:
                        raise ValueError('frame ids must increase')
                    previous_id = index
                    for key in ('run', 'segment'):
                        if type(event.get(key)) is not int or event[key] < 1:
                            raise ValueError(f'invalid {key}')
                    repaired = []
                    clock = event.get('processed_at')
                    if not finite_number(clock) or (previous_clock is not None and clock < previous_clock):
                        if not repair_timestamps:
                            raise ValueError('invalid/missing processed_at; use --repair-timestamps explicitly')
                        clock = (previous_clock + 1/fallback_fps) if previous_clock is not None else 0.
                        event['processed_at'] = clock
                        repaired.append('processed_at')
                    previous_clock = clock
                    # Repeated/backwards sensor timestamps are valid observations
                    # of stale input. Preserve them for the reliability tracker.
                    if not finite_number(event.get('captured_at')):
                        if not repair_timestamps:
                            raise ValueError('invalid/missing captured_at; use --repair-timestamps explicitly')
                        event['captured_at'] = clock
                        repaired.append('captured_at')
                    image = event.get('image')
                    if not isinstance(image, dict) or image.get('status') not in (
                            'saved', 'dropped_backpressure', 'sampled_out', 'rolling_discarded'):
                        raise ValueError('invalid image status')
                    if image['status'] == 'saved':
                        self.image_path(image.get('path'))
                    if not finite_number(event.get('vision_speed')):
                        raise ValueError('invalid vision_speed')
                    if type(event.get('action_sent')) is not bool:
                        raise ValueError('invalid action_sent')
                    if event.get('blocked_reason') is not None and not isinstance(event['blocked_reason'], str):
                        raise ValueError('invalid blocked_reason')
                    if not isinstance(event.get('observation'), dict):
                        raise ValueError('missing recorded observation')
                    event['timestamp_repairs'] = repaired
                    yield event
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'Invalid replay event at line {line_number}: {exc}') from exc
