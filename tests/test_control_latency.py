"""Check control against independent physics, delayed input and noisy images."""
from collections import deque
import random

import pytest

from flappy_auto.strategy import Strategy
from flappy_auto.vision import Bird, Observation, Pipe


def test_velocity_updates_with_frames_less_than_15ms_apart():
    policy = Strategy(gravity=740, jump_speed=260, pipe_speed=100)
    for tick in range(13):
        t = tick / 120
        obs = Observation(Bird(105, 120 + 80*t + .5*740*t*t, 12), None, 360, 600)
        assert not policy.should_flap(obs, 1 + t)[0]
    assert abs(policy.filtered_velocity - 154) < 25


def test_do_not_flap_into_upper_pipe_just_before_clearing_its_tail():
    policy = Strategy(gravity=1350, jump_speed=400, pipe_speed=170)
    current = Pipe(110, 180, 190, 350)
    following = Pipe(380, 446, 60, 230)
    obs = Observation(Bird(180, 210, 14), current, 480, 720, (current, following))
    assert not policy.should_flap(obs, 1)[0]


@pytest.mark.parametrize(('fps', 'seed'), [(30, 1), (60, 5), (120, 12)])
def test_cross_twenty_narrowing_gaps_with_delayed_noisy_observations(fps, seed):
    rng = random.Random(seed)
    centres = [rng.uniform(190, 450) for _ in range(23)]
    policy = Strategy()
    y, vy = 320., 0.
    dt = 1 / 240
    history = deque([(0., y)], maxlen=100)
    scheduled_keys = deque()
    passed = set()
    for tick in range(240 * 40):
        now = tick * dt
        while scheduled_keys and scheduled_keys[0] <= now:
            scheduled_keys.popleft()
            vy = -400
        vy += 1350 * dt
        y += vy * dt
        history.append((now, y))
        assert 14 < y < 622, f'Floor/ceiling collision after {now:.2f}s'
        for index, centre in enumerate(centres):
            left = 480 + index*280 - 170*now
            half_gap = max(65, 86-index)
            if left + 66 < 180:
                passed.add(index)
            if left-14 < 180 < left+80:
                assert centre-half_gap+14 <= y <= centre+half_gap-14, (
                    f'Pipe {index}, time={now:.2f}, y={y:.1f}, '
                    f'gravity={policy.gravity:.0f}, jump={policy.jump_speed:.0f}'
                )
        if len(passed) >= 20:
            return
        if tick % round(240 / fps):
            continue
        # Screenshot and key delivery have independent delays.
        captured_at, captured_y = next(((t, py) for t, py in reversed(history) if t <= now-.025), history[0])
        visible = tuple(
            Pipe(round(max(0, 480+i*280-170*captured_at)),
                 round(min(480, 546+i*280-170*captured_at)),
                 round(centre-max(65, 86-i)), round(centre+max(65, 86-i)))
            for i, centre in enumerate(centres)
            if -66 < 480+i*280-170*captured_at < 468
        )
        ahead = [pipe for pipe in visible if pipe.right >= 166]
        observation = Observation(Bird(180, captured_y+rng.gauss(0, .7), 14),
                                  ahead[0] if ahead else None, 480, 720, visible)
        if policy.should_flap(observation, captured_at)[0]:
            scheduled_keys.append(now+.008)
    pytest.fail('Did not reach twenty pipes')
