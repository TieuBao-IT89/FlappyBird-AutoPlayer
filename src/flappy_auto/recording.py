"""Bounded asynchronous session recording and the lightweight crash buffer."""
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from threading import Condition, Thread

import cv2

from .profiles import VisionConfig
from .session_format import SESSION_VERSION, observation_dict, write_json


class _RecordingWorker:
    """One pending raw image, one overlay and at most 512 compact events."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.condition = Condition()
        self.events = deque()
        self.image = self.overlay = None
        self.busy = self.closed = False
        self.error = None
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, event, row=None, frame=None):
        with self.condition:
            self._check()
            if self.closed:
                raise RuntimeError('Recording is closed')
            if len(self.events) >= 512:
                old, _ = self.events.popleft()
                self.recorder.stats['events_dropped'] += 1
                self.recorder.stats['actions_dropped'] += int(old.get('action_sent', False) or old['type'] == 'action')
                if self.image is not None and self.image[0] is old:
                    self.image = None
                    self.recorder.stats['images_dropped'] += 1
            self.events.append((event, row))
            if frame is not None:
                if self.image is not None:
                    self.image[0]['image']['status'] = 'dropped_backpressure'
                    self.recorder.stats['images_dropped'] += 1
                self.image = (event, frame)
            self.condition.notify()

    def submit_overlay(self, tick, frame):
        with self.condition:
            self._check()
            self.overlay = (tick, frame)
            self.condition.notify()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Background session recording failed') from self.error

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.events or self.overlay is not None or self.closed)
                    if not self.events and self.overlay is None:
                        break
                    events = list(self.events)
                    self.events.clear()
                    image, self.image = self.image, None
                    overlay, self.overlay = self.overlay, None
                    self.busy = True
                if image is not None:
                    self.recorder._write_image(*image)
                if overlay is not None:
                    tick, frame = overlay
                    if not cv2.imwrite(str(self.recorder.directory/f'frame_{tick:05d}.png'), frame):
                        raise OSError('Cannot write debug overlay')
                for event, row in events:
                    self.recorder.event_file.write(json.dumps(event, allow_nan=False)+'\n')
                    if row is not None:
                        self.recorder.writer.writerow(row)
                with self.condition:
                    self.busy = False
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error, self.busy = exc, False
                self.events.clear()
                self.image = self.overlay = None
                self.condition.notify_all()

    def flush(self):
        with self.condition:
            self.condition.wait_for(lambda: not self.events and self.overlay is None and not self.busy)
            self._check()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Recording worker did not stop')
        self._check()


class Recorder:
    def __init__(self, directory: Path | None, *, full=False, metadata=None):
        self.directory = (Path(directory)/datetime.now(timezone.utc).strftime('session_%Y%m%dT%H%M%S_%fZ')
                          if directory else None)
        self.full = full
        self.frames = deque(maxlen=20)
        self.last_image = float('-inf')
        self.file = self.event_file = self.writer = self.worker = None
        self.index = 0
        self.saved_rolling = set()
        self.stats = {'frames_seen': 0, 'images_requested': 0, 'images_saved': 0,
                      'images_dropped': 0, 'events_dropped': 0, 'actions_dropped': 0}
        self.metadata = {'kind': 'flappy-session', 'version': SESSION_VERSION,
                         'recording_mode': 'full' if full else 'rolling', 'closed': False,
                         'vision': asdict(VisionConfig()), 'frame_size': None, **(metadata or {})}
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=False)
            (self.directory/'frames').mkdir()
            self.file = (self.directory/'telemetry.csv').open('w', newline='', encoding='utf-8')
            self.event_file = (self.directory/'events.jsonl').open('w', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(('run', 'time', 'image_age_ms', 'score', 'bird_x', 'bird_y', 'radius',
                                  'pipe_left', 'pipe_right', 'gap_top', 'gap_bottom', 'flap',
                                  'velocity', 'gravity', 'jump', 'pipe_speed', 'latency_ms', 'extra_latency_ms',
                                  'vision_state', 'fresh', 'can_control', 'can_learn',
                                  'bird_confidence', 'bird_state', 'bird_reason', 'bird_predicted', 'bird_recovery_age',
                                  'pipe_confidence', 'pipe_state', 'pipe_reason', 'pipe_predicted', 'pipe_recovery_age',
                                  'frame_id', 'captured_at', 'processed_at', 'decision_ms', 'action_sent_at', 'segment'))
            write_json(self.directory/'session.json', self.metadata)
            self.worker = _RecordingWorker(self)

    def action(self, kind, sent_at, run=1, segment=1):
        if self.worker is not None:
            self.worker.submit({'type': 'action', 'kind': kind, 'sent_at': sent_at, 'run': run, 'segment': segment})

    def record(self, run, elapsed, age, score, frame, obs, flap, strategy, *, captured_at=None,
               processed_at=None, vision_speed=None, action_sent_at=None, decision_ms=0., segment=None):
        if self.worker is None:
            return
        size = [frame.shape[1], frame.shape[0]]
        if self.metadata['frame_size'] is not None and self.metadata['frame_size'] != size:
            raise ValueError('Recording frame size changed; start a new session')
        self.index += 1
        self.stats['frames_seen'] += 1
        self.metadata['frame_size'] = size
        bird, pipe = obs.bird, obs.pipe
        captured_at = max(0., elapsed-age) if captured_at is None else captured_at
        processed_at = captured_at+age if processed_at is None else processed_at
        segment = run if segment is None else segment
        image = {'status': 'sampled_out', 'path': None}
        selected = self.full or elapsed-self.last_image >= .1
        if selected:
            self.stats['images_requested'] += 1
            image = {'status': 'pending', 'path': f'frames/{self.index:09d}.png' if self.full else
                     f'run_{run:03d}/{elapsed:09.3f}.jpg'}
            self.last_image = elapsed
        event = {'type': 'frame', 'id': self.index, 'run': run, 'segment': segment, 'elapsed': elapsed,
                 'captured_at': captured_at, 'processed_at': processed_at,
                 'vision_speed': strategy.pipe_speed if vision_speed is None else vision_speed,
                 'blocked_reason': obs.bird_quality.reason.removeprefix('blocked: ') if obs.bird_quality.reason.startswith('blocked: ') else None,
                 'action_sent': bool(flap), 'action_sent_at': action_sent_at, 'decision_ms': decision_ms,
                 'score_so_far': score, 'observation': observation_dict(obs), 'image': image}
        row = (run, round(elapsed, 4), round(age*1000, 1), score,
               *([bird.x, bird.y, bird.radius] if bird else ('', '', '')),
               *([pipe.left, pipe.right, pipe.gap_top, pipe.gap_bottom] if pipe else ('', '', '', '')),
               flap, round(strategy.filtered_velocity, 2), round(strategy.gravity, 2),
               round(strategy.jump_speed, 2), round(strategy.pipe_speed, 2),
               round(strategy.latency_seconds*1000, 1), round(strategy._extra_latency*1000, 1),
               obs.state, obs.fresh, obs.can_control, obs.can_learn,
               round(obs.bird_quality.confidence, 3), obs.bird_quality.state, obs.bird_quality.reason,
               obs.bird_quality.predicted, round(obs.bird_quality.age, 4),
               round(obs.pipe_quality.confidence, 3), obs.pipe_quality.state, obs.pipe_quality.reason,
               obs.pipe_quality.predicted, round(obs.pipe_quality.age, 4),
               self.index, captured_at, processed_at, decision_ms, action_sent_at, segment)
        self.worker.submit(event, row, frame.copy() if selected else None)

    def _write_image(self, event, frame):
        if self.full:
            if not cv2.imwrite(str(self.directory/event['image']['path']), frame):
                raise OSError('Cannot write replay frame')
            event['image']['status'] = 'saved'
            self.stats['images_saved'] += 1
        else:
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise OSError('Cannot encode rolling frame')
            event['image']['status'] = 'rolling_discarded'
            self.frames.append((event['id'], event['image']['path'], encoded.tobytes()))

    def overlay(self, tick, frame):
        if self.worker is not None:
            self.worker.submit_overlay(tick, frame)

    def finish_run(self, number):
        if self.worker is None:
            return
        self.worker.flush()  # Between games, outside the control loop.
        (self.directory/f'run_{number:03d}').mkdir(exist_ok=True)
        for index, path, encoded in self.frames:
            (self.directory/path).write_bytes(encoded)
            self.saved_rolling.add(index)
            self.stats['images_saved'] += 1
        self.frames.clear()

    def close(self, results):
        if self.worker is None:
            return
        try:
            self.worker.close()
        finally:
            self.file.close()
            self.event_file.close()
        if not self.full:
            # Resolve ring retention in a streaming pass after control stops.
            path = self.directory/'events.jsonl'
            temporary = self.directory/'events.final.jsonl'
            with path.open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8') as target:
                for line in source:
                    event = json.loads(line)
                    if event['type'] == 'frame' and event['id'] in self.saved_rolling:
                        event['image']['status'] = 'saved'
                    target.write(json.dumps(event)+'\n')
            temporary.replace(path)
        self.metadata.update(closed=True, recording_stats=self.stats,
                             complete_images=self.full and self.stats['frames_seen'] > 0
                             and self.stats['images_saved'] == self.stats['frames_seen'] and not self.stats['events_dropped'])
        write_json(self.directory/'session.json', self.metadata)
        write_json(self.directory/'results.json', results)
        self.worker = None
