"""Check control against independent physics, delayed input and noisy images."""
from collections import deque
import random

import pytest
import numpy as np

from flappy_auto.strategy import Strategy
from flappy_auto.vision import Bird, Observation, Pipe


def test_rejects_early_jump_when_dropping_into_a_lower_gap(monkeypatch):
    # Regression from a visible-browser run: at key delivery the falling bird
    # is around y=367. Jumping there bottoms out above the new pipe's opening;
    # waiting until y=410 makes the same jump safe.
    policy = Strategy(gravity=1374, jump_speed=379, pipe_speed=170,
                      latency_seconds=.04, filtered_velocity=598)
    policy._extra_latency = .047
    pipe = Pipe(264, 324, 334, 465)
    monkeypatch.setattr(policy, '_plan', lambda obs, now: True)
    assert not policy.should_flap(Observation(Bird(180, 310, 15), pipe, 480, 720), 1)[0]
    safe = Strategy(gravity=1374, jump_speed=379, pipe_speed=170, latency_seconds=.04)
    monkeypatch.setattr(safe, '_plan', lambda obs, now: True)
    assert safe.should_flap(Observation(Bird(180, 410, 15), pipe, 480, 720), 1)[0]


def test_pending_input_does_not_change_velocity_before_the_key_arrives():
    policy = Strategy(gravity=1350, jump_speed=400, pipe_speed=170)
    policy.last_flap = 10
    policy._action_latency = .08
    policy._flap_origin = (300, 200)
    policy.last_bird = Bird(180, 300, 14)
    policy.last_seen = 10
    t = .04
    position = 300 + 200 * t + .5 * 1350 * t * t
    policy._observe(Observation(Bird(180, position, 14), None, 480, 720), 10 + t)
    assert policy.filtered_velocity == pytest.approx(200 + 1350 * t)
    # Now the image is 40 ms into the *new* flight, with the same key delay.
    t = .12
    action_y = 300 + 200 * .08 + .5 * 1350 * .08 ** 2
    position = action_y - 400 * .04 + .5 * 1350 * .04 ** 2
    policy._observe(Observation(Bird(180, position, 14), None, 480, 720), 10 + t)
    assert policy.filtered_velocity == pytest.approx(-400 + 1350 * .04)


def test_learns_key_delivery_delay_independently_from_jump_strength():
    policy = Strategy(gravity=1350, jump_speed=360)
    policy.last_flap = 10
    policy._action_latency = .035
    policy._flap_origin = (300, 200)
    actual_delay, actual_jump = .09, 400
    action_y = 300 + 200 * actual_delay + .5 * 1350 * actual_delay ** 2
    for elapsed in np.linspace(.16, .4, 16):
        age = elapsed - actual_delay
        policy._samples.append((10 + elapsed, action_y - actual_jump * age + .5 * 1350 * age ** 2))
    policy._calibrate(Observation(Bird(180, 300, 14), None, 480, 720), 10.4)
    assert policy.jump_speed == pytest.approx(actual_jump, abs=1)
    assert policy._extra_latency == pytest.approx(.055, abs=.003)


def test_pipe_speed_rejects_edge_jitter_and_switching_to_the_next_pipe():
    policy = Strategy(pipe_speed=170)
    for tick in range(20):
        now = 1 + tick / 60
        left = 300 - 170 * tick / 60 + (4 if tick % 2 else -4)
        pipe = Pipe(round(left), round(366 - 170 * tick / 60), 180, 340)
        policy._observe(Observation(Bird(180, 300, 14), pipe, 480, 720), now)
    assert policy.pipe_speed == pytest.approx(170, abs=6)
    policy._observe(Observation(Bird(180, 300, 14), Pipe(450, 480, 240, 400), 480, 720), 1.35)
    assert policy.pipe_speed == pytest.approx(170, abs=6)


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


@pytest.mark.parametrize(('fps', 'seed', 'key_delay'), [(30, 1, .008), (60, 5, .008), (120, 12, .008), (20, 7, .06)])
def test_cross_twenty_narrowing_gaps_with_delayed_noisy_observations(fps, seed, key_delay):
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
            scheduled_keys.append(now + key_delay)
    pytest.fail('Did not reach twenty pipes')
