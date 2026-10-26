"""Estimate physics from images and plan flaps through the visible pipe gaps."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import nsmallest
import math

import numpy as np

from .vision import Bird, Observation, Pipe, image_edge_margin


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
    _samples: deque = field(default_factory=lambda: deque(maxlen=90))
    _last_pipe: Pipe | None = None
    _last_pipe_time: float = 0.0
    _gravity_estimates: deque = field(default_factory=lambda: deque(maxlen=15))
    _jump_estimates: deque = field(default_factory=lambda: deque(maxlen=15))
    _action_latency: float = .035
    _radii: deque = field(default_factory=lambda: deque(maxlen=30))
    _control_interval: float = 1 / 60
    _pipe_samples: deque = field(default_factory=lambda: deque(maxlen=20))
    _latency_estimates: deque = field(default_factory=lambda: deque(maxlen=15))
    _extra_latency: float = 0.0
    _flap_origin: tuple[float, float] | None = None

    def new_run(self) -> Strategy:
        """Retain learned physics while resetting flight/input history."""
        policy = Strategy(min_flap_interval=self.min_flap_interval, gravity=self.gravity,
                          jump_speed=self.jump_speed, pipe_speed=self.pipe_speed)
        policy._gravity_estimates.extend(self._gravity_estimates)
        policy._jump_estimates.extend(self._jump_estimates)
        policy._latency_estimates.extend(self._latency_estimates)
        policy._extra_latency = self._extra_latency
        return policy

    def note_flap_sent(self, sent_at: float, key_delay: float = .008) -> None:
        """Use the actual dispatch time after planning, in the image's clock."""
        self._action_latency = max(0, sent_at - self.last_flap + key_delay)

    def _effect_time(self) -> float:
        return self.last_flap + max(0, self._action_latency + self._extra_latency)

    def _observe(self, obs: Observation, now: float) -> None:
        if not obs.can_learn:
            return
        bird = obs.bird
        if not self.gravity:
            self.gravity = obs.height * 1.6
        if not self.jump_speed:
            self.jump_speed = obs.height * .5
        if not self.pipe_speed:
            self.pipe_speed = obs.width * .3
        dt = now - self.last_seen if self.last_seen is not None else 0
        if .005 <= dt <= .1:
            self._control_interval += .15 * (dt - self._control_interval)
        self._radii.append(bird.radius)
        if self.last_bird is not None and .001 <= dt <= .5:
            velocity = (bird.y - self.last_bird.y) / dt + .5 * self.gravity * dt
            alpha = 1 - math.exp(-dt / .035)
            self.filtered_velocity += alpha * (velocity - self.filtered_velocity)
        elif dt > .5:
            self._samples.clear()
            self.filtered_velocity = 0
        self.last_bird, self.last_seen = bird, now

        effect_time = self._effect_time()
        # A requested jump has not necessarily affected this screenshot yet.
        # Keep the old flight until delivery, then use the new arc until enough
        # post-input samples exist to measure velocity independently.
        if self._flap_origin is not None and now - effect_time < .15:
            if now < effect_time:
                self.filtered_velocity = self._flap_origin[1] + self.gravity * (now - self.last_flap)
            else:
                self.filtered_velocity = -self.jump_speed + self.gravity * (now - effect_time)
        if now - effect_time >= .04:
            self._samples.append((now, bird.y))
        while self._samples and now - self._samples[0][0] > .5:
            self._samples.popleft()
        if len(self._samples) >= 4:
            times = np.array([t - now for t, _ in self._samples])
            positions = np.array([y for _, y in self._samples])
            if times[-1] - times[0] >= .08:
                velocities = positions - .5 * self.gravity * times ** 2
                self.filtered_velocity = float(np.polyfit(times, velocities, 1)[0])
        self.filtered_velocity = float(np.clip(self.filtered_velocity, -obs.height * 1.5, obs.height * 2))

        pipe = obs.pipe
        same_pipe = (pipe and self._last_pipe
                     and abs(pipe.gap_top - self._last_pipe.gap_top) < 8
                     and abs(pipe.gap_bottom - self._last_pipe.gap_bottom) < 8
                     and abs(pipe.right - self._last_pipe.right) < obs.width * .15)
        if not same_pipe:
            self._pipe_samples.clear()
        if pipe and pipe.right < obs.width - image_edge_margin(obs.width):
            self._pipe_samples.append((now, pipe.right))
        if len(self._pipe_samples) >= 4 and now - self._pipe_samples[0][0] >= .12:
            pt = np.array([t - now for t, _ in self._pipe_samples])
            px = np.array([x for _, x in self._pipe_samples])
            speed = -float(np.polyfit(pt, px, 1)[0])
            if obs.width * .05 < speed < obs.width * 1.5:
                self.pipe_speed += .15 * (speed - self.pipe_speed)
        self._last_pipe, self._last_pipe_time = pipe, now

    def _calibrate(self, obs: Observation, now: float) -> None:
        if not obs.can_learn:
            return
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
        if not math.isfinite(self.last_flap):
            return
        # Fit jump strength with the shared gravity estimate. Extrapolating a
        # noisy, independent quadratic amplifies sprite animation into large
        # changes in jump strength, particularly on short flight arcs.
        arc_times = np.array([t - self.last_flap for t, _ in self._samples])
        slope, intercept = np.polyfit(arc_times, positions - .5 * self.gravity * arc_times ** 2, 1)
        delay = self._action_latency
        if self._flap_origin is not None:
            initial_y, initial_velocity = self._flap_origin
            if initial_velocity - slope > obs.height * .15:
                inferred = (intercept - initial_y) / (initial_velocity - slope)
                if 0 <= inferred <= .25:
                    delay = float(inferred)
                    self._latency_estimates.append(float(np.clip(delay - self._action_latency, -.015, .12)))
                    self._extra_latency = float(np.median(self._latency_estimates))
        jump = -(slope + self.gravity * delay)
        if obs.height * .15 < jump < obs.height * 1.2:
            self._jump_estimates.append(float(jump))
            self.jump_speed = float(np.median(self._jump_estimates))

    def _project(self, obs: Observation, now: float) -> tuple[float, float, float]:
        bird = obs.bird
        lag = min(.25, max(0, self.latency_seconds + self._extra_latency))
        y = bird.y + self.filtered_velocity * lag + .5 * self.gravity * lag * lag
        velocity = self.filtered_velocity + self.gravity * lag
        pending = self._effect_time() - now
        if 0 < pending < lag:
            at_key = bird.y + self.filtered_velocity * pending + .5 * self.gravity * pending ** 2
            after_key = lag - pending
            y = at_key - self.jump_speed * after_key + .5 * self.gravity * after_key ** 2
            velocity = -self.jump_speed + self.gravity * after_key
        return y, velocity, lag

    def _safe_to_flap(self, obs: Observation, now: float) -> bool:
        y, _, lag = self._project(obs, now)
        radius = float(np.median(self._radii)) + max(4, obs.height * .008)
        apex_time = self.jump_speed / self.gravity
        apex_y = y - self.jump_speed ** 2 / (2 * self.gravity)
        if apex_y < radius:
            return False
        for pipe in obs.pipes or ((obs.pipe,) if obs.pipe else ()):
            entry = max(0, (pipe.left - obs.bird.x - radius) / self.pipe_speed - lag)
            leave = (pipe.right - obs.bird.x + radius) / self.pipe_speed - lag
            if leave < entry:
                continue
            low = min(y - self.jump_speed * t + .5 * self.gravity * t * t for t in (entry, leave))
            if entry <= apex_time <= leave:
                low = min(low, apex_y)
            # Once a jump will hit the upper pipe, another jump cannot undo
            # it. Keep this hard constraint outside the approximate search.
            if low < pipe.gap_top + radius:
                return False
        return True

    def _plan(self, obs: Observation, now: float) -> bool:
        bird = obs.bird
        y, velocity, lag = self._project(obs, now)
        pipes = sorted(obs.pipes or ((obs.pipe,) if obs.pipe else ()), key=lambda pipe: pipe.left)
        radius = float(np.median(self._radii)) + max(4, obs.height * .008)
        ceiling, floor = radius, obs.height * .88 - radius
        dt, steps = .07, 24
        # The first decision lasts one actual control tick, rather than the
        # much coarser planning step. Waiting 20 ms and waiting 70 ms can have
        # opposite outcomes inside a narrow gap.
        first_dt = min(.06, max(.025, self._control_interval * 1.7))
        # (cost, y, velocity, cooldown, first_action)
        beam = [(0.0, y, velocity, max(0, self.min_flap_interval - (now - self.last_flap)), False)]
        elapsed = lag
        for step in range(steps):
            step_dt = first_dt if step == 0 else dt
            start = elapsed
            elapsed += step_dt
            active = [p for p in pipes if p.right - self.pipe_speed * elapsed >= bird.x - radius]
            target = active[0].gap_center if active else obs.height * .47
            crossings = []
            for pipe in pipes:
                entry = (pipe.left - bird.x - radius) / self.pipe_speed - start
                leave = (pipe.right - bird.x + radius) / self.pipe_speed - start
                if entry <= step_dt and leave >= 0:
                    crossings.append((pipe, max(0, entry), min(step_dt, leave)))
            candidates = []
            for cost, position, speed, cooldown, first in beam:
                for flap in (False, True) if cooldown <= .0001 else (False,):
                    vy = -self.jump_speed if flap else speed
                    ny = position + vy * step_dt + .5 * self.gravity * step_dt * step_dt
                    nv = vy + self.gravity * step_dt
                    penalty = ((ny - target) / obs.height) ** 2 + .008 * (nv / obs.height) ** 2
                    if flap:
                        penalty += .001
                    outside = max(0, ceiling - min(position, ny), max(position, ny) - floor)
                    apex = -vy / self.gravity
                    if 0 < apex < step_dt:
                        outside = max(outside, ceiling - (position - vy * vy / (2 * self.gravity)))
                    for pipe, entry, leave in crossings:
                        ys = [position + vy * t + .5 * self.gravity * t * t for t in (entry, leave)]
                        apex = -vy / self.gravity
                        if entry < apex < leave:
                            ys.append(position - vy * vy / (2 * self.gravity))
                        outside = max(outside, pipe.gap_top + radius - min(ys), max(ys) - (pipe.gap_bottom - radius))
                    if outside > 0:
                        penalty += 1000 + outside ** 2
                    candidates.append((cost + penalty * step_dt / dt, ny, nv,
                                       self.min_flap_interval - step_dt if flap else max(0, cooldown - step_dt),
                                       flap if step == 0 else first))
            # Similar action histories otherwise fill the entire beam with
            # near-identical states and discard the only safe way to wait.
            cells = {}
            for candidate in candidates:
                cost, py, pv, cooldown, first = candidate
                cell = (round(py / 4), round(pv / 35), round(cooldown / dt), first)
                if cell not in cells or cost < cells[cell][0]:
                    cells[cell] = candidate
            beam = nsmallest(32, cells.values(), key=lambda state: state[0])
        return min(beam, key=lambda state: state[0])[4]

    def should_flap(self, obs: Observation, now: float) -> tuple[bool, str]:
        if not obs.can_control:
            # Keep the last genuine position and cooldown for reacquisition,
            # but never fit an arc or pipe-speed sample across uncertainty.
            self._samples.clear()
            self._pipe_samples.clear()
            self._last_pipe = None
            reason = obs.bird_quality.reason if not obs.bird_quality.trusted or not obs.fresh else obs.pipe_quality.reason
            return False, f'vision {obs.state}: {reason}'
        if self.last_seen is not None and now <= self.last_seen:
            return False, 'waiting for a fresh frame'
        self._observe(obs, now)
        flap = (now - self.last_flap >= self.min_flap_interval
                and self._safe_to_flap(obs, now) and self._plan(obs, now))
        if flap:
            self._calibrate(obs, now)
            self._flap_origin = (obs.bird.y, self.filtered_velocity)
            self.last_flap = now
            self._action_latency = max(0, self.latency_seconds)
            self._samples.clear()
        status = 'flap' if flap else 'coast'
        return flap, f'{status}: y={obs.bird.y:.0f}, vy={self.filtered_velocity:.0f}, g={self.gravity:.0f}, jump={self.jump_speed:.0f}'
