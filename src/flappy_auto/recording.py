"""Optional telemetry and a rolling image buffer for collision diagnosis."""
from collections import deque
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

import cv2

from .workers import LatestWorker


class Recorder:
    def __init__(self, directory: Path | None):
        self.directory = (directory / datetime.now(timezone.utc).strftime('session_%Y%m%dT%H%M%S_%fZ')
                          if directory else None)
        self.frames = deque(maxlen=20)
        self.last_image = float('-inf')
        self.file = None
        self.writer = None
        self.worker = None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.file = (self.directory / 'telemetry.csv').open('w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(('run', 'time', 'image_age_ms', 'score', 'bird_x', 'bird_y', 'radius',
                                  'pipe_left', 'pipe_right', 'gap_top', 'gap_bottom', 'flap',
                                  'velocity', 'gravity', 'jump', 'pipe_speed', 'latency_ms', 'extra_latency_ms'))
            self.worker = LatestWorker(self._write_image)

    def record(self, run, elapsed, age, score, frame, obs, flap, strategy):
        if self.writer is None:
            return
        bird, pipe = obs.bird, obs.pipe
        self.writer.writerow((run, round(elapsed, 4), round(age * 1000, 1), score,
                              *([bird.x, bird.y, bird.radius] if bird else ('', '', '')),
                              *([pipe.left, pipe.right, pipe.gap_top, pipe.gap_bottom] if pipe else ('', '', '', '')),
                              flap, round(strategy.filtered_velocity, 2), round(strategy.gravity, 2),
                              round(strategy.jump_speed, 2), round(strategy.pipe_speed, 2),
                              round(strategy.latency_seconds * 1000, 1), round(strategy._extra_latency * 1000, 1)))
        if elapsed - self.last_image >= .1:
            self.worker.submit(('frame', elapsed, frame))
            self.last_image = elapsed

    def _write_image(self, job):
        kind, stamp, frame = job
        if kind == 'frame':
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self.frames.append((stamp, encoded.tobytes()))
        else:
            cv2.imwrite(str(self.directory / f'frame_{stamp:05d}.png'), frame)

    def overlay(self, tick, frame):
        if self.worker is not None:
            self.worker.submit(('overlay', tick, frame))

    def finish_run(self, number):
        if not self.directory:
            return
        self.worker.flush()
        folder = self.directory / f'run_{number:03d}'
        folder.mkdir(exist_ok=True)
        for elapsed, encoded in self.frames:
            (folder / f'{elapsed:09.3f}.jpg').write_bytes(encoded)
        self.frames.clear()
        self.file.flush()

    def close(self, results):
        if self.file:
            self.worker.close()
            self.file.close()
            (self.directory / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
