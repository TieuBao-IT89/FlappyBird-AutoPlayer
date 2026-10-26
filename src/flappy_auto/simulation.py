"""Independent closed-loop physics for seeded controller comparisons (no browser/CV)."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
import os
import random
import time

from .strategy import Strategy
from .vision import Bird, Observation, Pipe


@dataclass(frozen=True)
class SimulationConfig:
    width: int = 480
    height: int = 720
    bird_x: float = 180.
    start_y: float = 320.
    initial_velocity: float = 0.
    radius: float = 14.
    gravity: float = 1350.
    jump_speed: float = 400.
    pipe_speed: float = 170.
    pipe_width: float = 66.
    pipe_spacing: float = 280.
    first_pipe_x: float = 480.
    gap_half: float = 86.
    min_gap_half: float = 65.
    gap_narrowing: float = 1.
    center_min: float = 190.
    center_max: float = 450.
    physics_hz: int = 240
    capture_fps: int = 30
    image_delay: float = .025
    input_delay: float = .008
    decision_delay: float = .006
    bird_noise: float = .7
    edge_noise: float = .5
    seconds: float = 40.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'simulation.{key} must be finite')
            if key not in ('initial_velocity',) and value < 0:
                raise ValueError(f'simulation.{key} must be nonnegative')
        for key in ('width', 'height', 'physics_hz', 'capture_fps'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f'simulation.{key} must be a positive integer')
        for key in ('radius', 'gravity', 'jump_speed', 'pipe_speed', 'pipe_width', 'pipe_spacing', 'seconds'):
            if getattr(self, key) <= 0:
                raise ValueError(f'simulation.{key} must be positive')
        if (self.capture_fps > self.physics_hz or self.physics_hz > 2000 or self.seconds > 3600
                or max(self.image_delay, self.input_delay, self.decision_delay) > self.seconds):
            raise ValueError('Invalid simulation rate, duration or delay budget')
        if (not self.radius < self.bird_x < self.width-self.radius
                or not self.radius < self.start_y < self.height*.88-self.radius
                or not self.radius < self.min_gap_half <= self.gap_half
                or self.center_min > self.center_max
                or self.center_min-self.gap_half <= self.radius
                or self.center_max+self.gap_half >= self.height*.88-self.radius
                or self.pipe_spacing <= self.pipe_width+2*self.radius):
            raise ValueError('Invalid initial state, gap geometry or pipe spacing')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data)-set(cls.__dataclass_fields__):
            raise ValueError('Unknown simulation configuration fields')
        return cls(**data)


class ReactivePolicy:
    """Fixed geometric baseline, with no privileged simulator state or physics fit."""

    def __init__(self):
        self.last_flap = float('-inf')
        self.last_seen = None
        self.last_y = None

    def new_run(self):
        return ReactivePolicy()

    def should_flap(self, obs, now):
        if not obs.can_control or (self.last_seen is not None and now <= self.last_seen):
            return False, 'unavailable observation'
        velocity = 0 if self.last_seen is None else (obs.bird.y-self.last_y)/(now-self.last_seen)
        self.last_seen, self.last_y = now, obs.bird.y
        target = obs.pipe.gap_center if obs.pipe else obs.height*.47
        flap = obs.bird.y > target and velocity > -obs.height*.1 and now-self.last_flap >= .18
        if flap:
            self.last_flap = now
        return flap, 'reactive threshold'

    def note_flap_sent(self, sent_at, key_delay):
        pass


def make_policy(name):
    if name == 'predictive':
        return Strategy()
    if name == 'reactive':
        return ReactivePolicy()
    raise ValueError(f'Unknown controller {name}')


def _range(y, velocity, gravity, start, end):
    points = [start, end]
    apex = -velocity/gravity
    if start < apex < end:
        points.append(apex)
    values = [y+velocity*t+.5*gravity*t*t for t in points]
    return min(values), max(values)


def _collision(y, velocity, now, duration, config, centers):
    low, high = _range(y, velocity, config.gravity, 0, duration)
    if low < config.radius:
        return 'ceiling collision'
    if high > config.height*.88-config.radius:
        return 'floor collision'
    for index, center in enumerate(centers):
        x = config.first_pipe_x+index*config.pipe_spacing-config.pipe_speed*now
        enter = max(0., (x-config.bird_x-config.radius)/config.pipe_speed)
        leave = min(duration, (x+config.pipe_width-config.bird_x+config.radius)/config.pipe_speed)
        if enter > leave:
            continue
        low, high = _range(y, velocity, config.gravity, enter, leave)
        half = max(config.min_gap_half, config.gap_half-index*config.gap_narrowing)
        if low < center-half+config.radius or high > center+half-config.radius:
            return f'pipe {index} collision'
    return None


def run_trial(config, policy, *, seed, trial, policy_name, mode, stop_requested=None):
    scenario_seed = seed+trial*1_000_003
    geometry_rng, noise_rng = random.Random(scenario_seed), random.Random(scenario_seed ^ 0x5EED)
    count = math.ceil((config.pipe_speed*config.seconds+config.width)/config.pipe_spacing)+2
    centers = [geometry_rng.uniform(config.center_min, config.center_max) for _ in range(count)]
    history = deque([(0., config.start_y)], maxlen=math.ceil(config.image_delay*config.physics_hz)+4)
    pending = []
    y, velocity, now, next_decision = config.start_y, config.initial_velocity, 0., 0.
    decisions, flaps, ages, decision_times = 0, 0, [], []
    digest = hashlib.sha256()
    started = time.perf_counter()
    status, reason = 'time_limit', 'simulation time budget exhausted'
    dt = 1/config.physics_hz
    while now < config.seconds-1e-10:
        if stop_requested is not None and stop_requested(now):
            status, reason = 'stopped', 'requested stop'
            break
        if now+1e-10 >= next_decision:
            target_time = max(0., now-config.image_delay)
            captured_at, captured_y = next(((t, py) for t, py in reversed(history) if t <= target_time+1e-10), history[0])
            pipes = []
            for index, center in enumerate(centers):
                left = config.first_pipe_x+index*config.pipe_spacing-config.pipe_speed*captured_at
                if -config.pipe_width < left < config.width:
                    half = max(config.min_gap_half, config.gap_half-index*config.gap_narrowing)
                    jitter = noise_rng.gauss(0, config.edge_noise)
                    pipes.append(Pipe(round(max(0, left+jitter)), round(min(config.width, left+config.pipe_width+jitter)),
                                      round(center-half), round(center+half)))
            ahead = [pipe for pipe in pipes if pipe.right >= config.bird_x-config.radius]
            bird = Bird(config.bird_x, captured_y+noise_rng.gauss(0, config.bird_noise), config.radius)
            obs = Observation(bird, ahead[0] if ahead else None, config.width, config.height, tuple(pipes))
            if isinstance(policy, Strategy):
                policy.latency_seconds = now-captured_at+config.decision_delay+config.input_delay
            before = time.perf_counter()
            flap, _ = policy.should_flap(obs, captured_at)
            decision_times.append((time.perf_counter()-before)*1000)
            ages.append((now-captured_at)*1000)
            decisions += 1
            if flap:
                sent_at = now+config.decision_delay
                policy.note_flap_sent(sent_at, config.input_delay)
                heapq.heappush(pending, sent_at+config.input_delay)
                flaps += 1
            digest.update(json.dumps([round(now, 12), round(y, 9), round(velocity, 9), flap], separators=(',', ':')).encode())
            next_decision += 1/config.capture_fps
        end = min(config.seconds, now+dt)
        while now < end-1e-12:
            if pending and pending[0] <= now+1e-12:
                heapq.heappop(pending)
                velocity = -config.jump_speed
                continue
            until = min(end, pending[0] if pending else end)
            step = until-now
            collision = _collision(y, velocity, now, step, config, centers)
            if collision:
                lower, upper = 0., step
                for _ in range(20):
                    middle = (lower+upper)/2
                    if _collision(y, velocity, now, middle, config, centers):
                        upper = middle
                    else:
                        lower = middle
                step = upper
                status, reason = 'completed', collision
            y += velocity*step+.5*config.gravity*step*step
            velocity += config.gravity*step
            now += step
            if collision:
                break
        history.append((now, y))
        if status == 'completed':
            break
    wall = time.perf_counter()-started
    score = sum(config.first_pipe_x+i*config.pipe_spacing+config.pipe_width-config.pipe_speed*now
                < config.bird_x-config.radius for i in range(len(centers)))
    times = sorted(decision_times)
    return {'seed': seed, 'scenario_seed': scenario_seed, 'trial': trial, 'policy': policy_name, 'mode': mode,
            'process_id': os.getpid(), 'status': status, 'reason': reason,
            'score_so_far': score, 'final_score': score if status == 'completed' else None,
            'censored': status != 'completed', 'survival_seconds': round(now, 9),
            'decisions': decisions, 'flaps': flaps, 'configured_fps': config.capture_fps,
            'observations_per_sim_second': decisions/now if now else 0.,
            'processing_fps': decisions/wall if wall else None, 'wall_seconds': wall,
            'image_age_ms_mean': sum(ages)/len(ages) if ages else None,
            'configured_input_delay_ms': config.input_delay*1000,
            'configured_decision_delay_ms': config.decision_delay*1000,
            'decision_ms_mean': sum(times)/len(times) if times else None,
            'decision_ms_p95': times[min(len(times)-1, math.ceil(len(times)*.95)-1)] if times else None,
            'trajectory_sha256': digest.hexdigest(),
            'learned_physics': {key: getattr(policy, key, None) for key in ('gravity', 'jump_speed', 'pipe_speed')}}
