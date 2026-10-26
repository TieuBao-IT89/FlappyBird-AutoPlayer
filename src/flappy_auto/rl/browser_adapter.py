"""Screenshot measurements mapped to the version-1 FlappyEnv feature contract.

Use an ROI covering the entire arena: normalizing a partial crop changes meaning.
Velocity is measured in screenshot pixels/second, positive downwards. No game
state, simulator physics, or extrapolated missing objects are used.
"""
from __future__ import annotations

import math

import numpy as np

from ..profiles import TrackingConfig
from ..vision import Observation


LOW = np.array([0, -2, -1, 0, 0], dtype=np.float32)
HIGH = np.array([1, 2, 3, 1, 1], dtype=np.float32)


def to_dqn_observation(obs: Observation, velocity: float) -> np.ndarray | None:
    """Return five float32 features, or None for unusable geometry/evidence."""
    if not obs.can_control or obs.bird is None:
        return None
    bird = obs.bird
    if (not all(math.isfinite(v) for v in (obs.width, obs.height, bird.x, bird.y, bird.radius, velocity))
            or min(obs.width, obs.height, bird.radius) <= 0
            or not (0 <= bird.x <= obs.width and 0 <= bird.y <= obs.height)):
        return None
    pipes = obs.pipes or ((obs.pipe,) if obs.pipe else ())
    if any(not all(math.isfinite(v) for v in (p.left, p.right, p.gap_top, p.gap_bottom))
           or not (0 <= p.left < p.right <= obs.width and 0 <= p.gap_top < p.gap_bottom <= obs.height)
           for p in pipes):
        return None
    # A pipe is still relevant while its trailing edge overlaps the bird's
    # radius, exactly as in FlappyEnv._observation (not its score crossing).
    target = min((p for p in pipes if p.right >= bird.x - bird.radius),
                 key=lambda p: p.left, default=None)
    if target is None:
        return None
    values = [bird.y / obs.height, velocity / obs.height * 600 / 400,
              (target.right - bird.x) / obs.width,
              target.gap_top / obs.height, target.gap_bottom / obs.height]
    if not np.isfinite(values).all():
        return None
    return np.clip(np.asarray(values, dtype=np.float32), LOW, HIGH)


class BrowserObservationAdapter:
    """Require consecutive, trusted samples; never differentiate across outages."""

    def __init__(self, limits: TrackingConfig | None = None):
        self.limits = limits or TrackingConfig()
        self.reset()

    def reset(self):
        self.previous = None
        self.last_timestamp = float('-inf')
        self.velocity = 0.0
        self.reason = 'velocity warmup'

    def _reject(self, reason):
        self.previous = None
        self.velocity = 0.0
        self.reason = reason
        return None

    def update(self, obs: Observation, captured_at: float, *, now: float) -> np.ndarray | None:
        if not all(math.isfinite(t) for t in (captured_at, now)):
            return self._reject('invalid timestamp')
        if captured_at <= self.last_timestamp:
            return self._reject('nonmonotonic frame')
        self.last_timestamp = captured_at
        if not 0 <= now - captured_at <= self.limits.max_frame_age:
            return self._reject('stale or future frame')
        if (not obs.can_control
                or not math.isfinite(obs.bird_quality.confidence)
                or not math.isfinite(obs.pipe_quality.confidence)
                or obs.bird_quality.confidence < self.limits.min_bird_confidence
                or obs.pipe_quality.confidence < self.limits.min_pipe_confidence
                or to_dqn_observation(obs, 0) is None):
            return self._reject('unreliable observation')
        sample = (captured_at, obs.bird, obs.width, obs.height)
        previous, self.previous = self.previous, sample
        if previous is None:
            self.reason = 'velocity warmup'
            return None
        stamp, bird, width, height = previous
        dt = captured_at - stamp
        if (width, height) != (obs.width, obs.height) or not .005 <= dt <= self.limits.max_frame_age:
            self.velocity = 0.0
            self.reason = 'velocity warmup after frame gap/resize'
            return None
        self.velocity = (obs.bird.y - bird.y) / dt
        if (abs(self.velocity) > obs.height * self.limits.max_bird_speed
                or abs(obs.bird.x - bird.x) > obs.width * self.limits.bird_x_tolerance):
            return self._reject('bird discontinuity')
        self.reason = 'ready'
        return to_dqn_observation(obs, self.velocity)
