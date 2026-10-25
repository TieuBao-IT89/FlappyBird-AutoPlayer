"""Estimate physics from images and plan flaps through the visible pipe gaps."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import nsmallest
import math

import numpy as np

from .vision import Bird, Observation, Pipe


@dataclass
class Strategy:
    min_flap_interval: float = .18
    latency_seconds: float = .035
    last_bird: Bird | None = None
    last_seen: float | None = None
    last_flap: float = float('-inf')
    filtered_velocity: float = 0.0
    gravity: float = 0.0
    jump_speed: float = 0.0
    pipe_speed: float = 0.0
    _samples: deque = field(default_factory=lambda: deque(maxlen=48))
    _last_pipe: Pipe | None = None
    _last_pipe_time: float = 0.0
    _gravity_estimates: deque = field(default_factory=lambda: deque(maxlen=7))
    _jump_estimates: deque = field(default_factory=lambda: deque(maxlen=7))
    _action_latency: float = .035

    def _observe(self, obs: Observation, now: float) -> None:
        bird = obs.bird
        if not self.gravity:
            self.gravity = obs.height * 1.6
        if not self.jump_speed:
            self.jump_speed = obs.height * .5
        if not self.pipe_speed:
            self.pipe_speed = obs.width * .3
        dt = now - self.last_seen if self.last_seen is not None else 0
        if self.last_bird is not None and .001 <= dt <= .5:
            velocity = (bird.y - self.last_bird.y) / dt + .5 * self.gravity * dt
            alpha = 1 - math.exp(-dt / .035)
            self.filtered_velocity += alpha * (velocity - self.filtered_velocity)
        elif dt > .5:
            self._samples.clear()
            self.filtered_velocity = 0
        self.last_bird, self.last_seen = bird, now

        # Ignore the images in flight when a key was sent. Fit only the new arc.
        if now - self.last_flap >= .06:
            self._samples.append((now, bird.y))
        while self._samples and now - self._samples[0][0] > .3:
            self._samples.popleft()
        if len(self._samples) >= 4:
            times = np.array([t - now for t, _ in self._samples])
            positions = np.array([y for _, y in self._samples])
            if times[-1] - times[0] >= .08:
                velocities = positions - .5 * self.gravity * times ** 2
                self.filtered_velocity = float(np.polyfit(times, velocities, 1)[0])
        self.filtered_velocity = float(np.clip(self.filtered_velocity, -obs.height * 1.5, obs.height * 2))

        pipe = obs.pipe
        if pipe and self._last_pipe and 0 < now - self._last_pipe_time < .5:
            speed = (self._last_pipe.left - pipe.left) / (now - self._last_pipe_time)
            if obs.width * .05 < speed < obs.width * 1.5:
                self.pipe_speed += .2 * (speed - self.pipe_speed)
        self._last_pipe, self._last_pipe_time = pipe, now

    def _calibrate(self, obs: Observation, now: float) -> None:
        # One estimate per flight arc, rather than correlated estimates every
        # frame. Median filtering rejects wing-animation/quantization outliers.
        if len(self._samples) < 6 or self._samples[-1][0] - self._samples[0][0] < .14:
            return
        times = np.array([t - now for t, _ in self._samples])
        positions = np.array([y for _, y in self._samples])
        coefficients = np.polyfit(times, positions, 2)
        acceleration = float(2 * coefficients[0])
        residual = np.sqrt(np.mean((np.polyval(coefficients, times) - positions) ** 2))
        if not (obs.height * .4 < acceleration < obs.height * 4 and residual < 2.5):
            return
        self._gravity_estimates.append(acceleration)
        self.gravity = float(np.median(self._gravity_estimates))
        jump = -(coefficients[1] + acceleration * (self.last_flap + self._action_latency - now))
        if obs.height * .15 < jump < obs.height * 1.2:
            self._jump_estimates.append(float(jump))
            self.jump_speed = float(np.median(self._jump_estimates))

    def _plan(self, obs: Observation, now: float) -> bool:
        bird = obs.bird
        lag = min(.15, max(0, self.latency_seconds))
        y = bird.y + self.filtered_velocity * lag + .5 * self.gravity * lag * lag
        velocity = self.filtered_velocity + self.gravity * lag
        pipes = sorted(obs.pipes or ((obs.pipe,) if obs.pipe else ()), key=lambda pipe: pipe.left)
        radius = bird.radius + max(3, obs.height * .006)
        ceiling, floor = radius, obs.height * .88 - radius
        dt, steps = .07, 20
        # (cost, y, velocity, cooldown, first_action)
        beam = [(0.0, y, velocity, max(0, self.min_flap_interval - (now - self.last_flap)), False)]
        for step in range(steps):
            elapsed = (step + 1) * dt + lag
            active = [p for p in pipes if p.right - self.pipe_speed * elapsed >= bird.x - radius]
            target = active[0].gap_center if active else obs.height * .47
            crossings = []
            for pipe in pipes:
                entry = (pipe.left - bird.x - radius) / self.pipe_speed - lag - step * dt
                leave = (pipe.right - bird.x + radius) / self.pipe_speed - lag - step * dt
                if entry <= dt and leave >= 0:
                    crossings.append((pipe, max(0, entry), min(dt, leave)))
            candidates = []
            for cost, position, speed, cooldown, first in beam:
                for flap in (False, True) if cooldown <= .0001 else (False,):
                    vy = -self.jump_speed if flap else speed
                    ny = position + vy * dt + .5 * self.gravity * dt * dt
                    nv = vy + self.gravity * dt
                    penalty = ((ny - target) / obs.height) ** 2 + .008 * (nv / obs.height) ** 2
                    if flap:
                        penalty += .001
                    outside = max(0, ceiling - ny, ny - floor)
                    for pipe, entry, leave in crossings:
                        ys = [position + vy * t + .5 * self.gravity * t * t for t in (entry, leave)]
                        apex = -vy / self.gravity
                        if entry < apex < leave:
                            ys.append(position - vy * vy / (2 * self.gravity))
                        outside = max(outside, pipe.gap_top + radius - min(ys), max(ys) - (pipe.gap_bottom - radius))
                    if outside > 0:
                        penalty += 1000 + outside ** 2
                    candidates.append((cost + penalty, ny, nv,
                                       self.min_flap_interval - dt if flap else max(0, cooldown - dt),
                                       flap if step == 0 else first))
            beam = nsmallest(14, candidates, key=lambda state: state[0])
        return min(beam, key=lambda state: state[0])[4]

    def should_flap(self, obs: Observation, now: float) -> tuple[bool, str]:
        if obs.bird is None:
            self.last_bird = None
            self.last_seen = None
            self.filtered_velocity = 0
            self._samples.clear()
            return False, 'bird not detected'
        if self.last_seen is not None and now <= self.last_seen:
            return False, 'waiting for a fresh frame'
        self._observe(obs, now)
        flap = now - self.last_flap >= self.min_flap_interval and self._plan(obs, now)
        if flap:
            self._calibrate(obs, now)
            self.last_flap = now
            self._action_latency = self.latency_seconds
            self.filtered_velocity = -self.jump_speed
            self._samples.clear()
        status = 'flap' if flap else 'coast'
        return flap, f'{status}: y={obs.bird.y:.0f}, vy={self.filtered_velocity:.0f}, g={self.gravity:.0f}, jump={self.jump_speed:.0f}'
