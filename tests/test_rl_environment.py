import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flappy_auto.rl.environment import FlappyEnv


def test_reset_and_step_spaces():
    env = FlappyEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info == {"score": 0, "steps": 0}
    expected = [270/600, 0, (370+52-105)/360,
                (env.pipes[0].opening-84)/600, (env.pipes[0].opening+84)/600]
    np.testing.assert_allclose(obs, expected)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    env.close()


def test_flap_changes_velocity():
    env = FlappyEnv()
    env.reset(seed=0)
    _, _, _, _, _ = env.step(1)
    assert env.bird_vy < 0


def test_same_seed_and_actions_same_outcome():
    a, b = FlappyEnv(), FlappyEnv()
    x, _ = a.reset(seed=11)
    y, _ = b.reset(seed=11)
    np.testing.assert_array_equal(x, y)
    for action in [1, 0, 0, 1, 0, 0]:
        ra, rb = a.step(action), b.step(action)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1:] == rb[1:]


@pytest.mark.parametrize('boundary', ['ceiling', 'floor'])
def test_ceiling_floor_collision_ends_episode(boundary):
    env = FlappyEnv()
    env.reset(seed=5)
    env.bird_y = env.floor-env.bird_radius-.1 if boundary == 'floor' else env.bird_radius+.1
    env.bird_vy = 100 if boundary == 'floor' else -100
    _, reward, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated and reward < 0
    assert env.observation_space.contains(env._observation())
    with pytest.raises(RuntimeError):
        env.step(0)


def test_time_limit_truncation():
    env = FlappyEnv(max_steps=1)
    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and truncated and info["steps"] == 1
    with pytest.raises(RuntimeError, match='reset'):
        env.step(0)
    first, info = env.reset(seed=5)
    assert info == {'score': 0, 'steps': 0}
    assert not env.step(0)[2]
    assert env.observation_space.contains(first)


def test_pipe_crossing_increases_score():
    env = FlappyEnv()
    env.reset(seed=5)
    pipe = env.pipes[0]
    env.bird_y = pipe.opening
    env.bird_vy = 0
    pipe.x = env.bird_x - env.pipe_width + .1
    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    assert info["score"] == 1
    assert reward == pytest.approx(1.01)
    _, reward, terminated, _, info = env.step(0)
    assert not terminated and info['score'] == 1
    assert reward == pytest.approx(.01)  # One reward per pipe, even while overlapping.


@pytest.mark.parametrize('action', [-1, 2, .5, '1', None, np.array([1])])
def test_invalid_action(action):
    env = FlappyEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.steps == 0 and env.bird_y == 270


def test_gymnasium_api_checker():
    env = FlappyEnv()
    try:
        check_env(env, skip_render_check=True)  # This environment exposes no render modes.
    finally:
        env.close()


@pytest.mark.parametrize('max_steps', [0, -1, True, 1.5, float('nan'), float('inf'), '2'])
def test_invalid_step_limit(max_steps):
    with pytest.raises(ValueError, match='positive integer'):
        FlappyEnv(max_steps=max_steps)


@pytest.mark.parametrize('value', ['false', 0, None])
def test_invalid_gap_randomization_setting(value):
    with pytest.raises(ValueError, match='boolean'):
        FlappyEnv(randomize_gaps=value)


def test_reset_required_before_first_step():
    with pytest.raises(RuntimeError, match='reset'):
        FlappyEnv().step(0)


def test_flap_and_coast_use_demo_update_order_and_accept_numpy_actions():
    env = FlappyEnv()
    env.reset(seed=0)
    env.step(np.int64(1))
    velocity = -260+740/60
    assert env.bird_y == pytest.approx(270+velocity/60)
    assert env.bird_vy == pytest.approx(velocity)
    env.step(np.int64(0))
    assert env.bird_vy == pytest.approx(velocity+740/60)


@pytest.mark.parametrize('side', ['upper', 'lower'])
def test_pipe_collision_terminates_without_waiting_for_floor(side):
    env = FlappyEnv()
    env.reset(seed=5)
    pipe = env.pipes[0]
    pipe.x = env.bird_x
    env.bird_y = pipe.opening + (-env.half_gap if side == 'upper' else env.half_gap)
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert reward == pytest.approx(-.99) and info['score'] == 0


def test_collision_takes_priority_at_step_limit():
    env = FlappyEnv(max_steps=1)
    env.reset(seed=5)
    env.bird_y, env.bird_vy = env.floor-env.bird_radius-.1, 100
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated


def test_pipe_recycling_is_bounded_and_reproducible():
    a, b = FlappyEnv(), FlappyEnv()
    for env in (a, b):
        env.reset(seed=9)
        first, second, third = env.pipes
        first.x, second.x, third.x = -59.9, 150, 360
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
        assert len(env.pipes) == 3 and env.pipes[0] is second
        assert env.pipes[2].x-env.pipes[1].x == pytest.approx(210)
        assert 190 <= env.pipes[2].opening < 360
        assert not env.pipes[2].passed
    assert a.pipes == b.pipes


def test_randomized_and_sinusoidal_gap_sequences():
    a, b = FlappyEnv(), FlappyEnv()
    a.reset(seed=1)
    b.reset(seed=2)
    assert [p.opening for p in a.pipes] != [p.opening for p in b.pipes]
    fixed = FlappyEnv(randomize_gaps=False)
    original, _ = fixed.reset(seed=1)
    np.testing.assert_allclose([p.opening for p in fixed.pipes],
                               [255, 255+np.sin(1.7)*85, 255+np.sin(3.4)*85])
    repeated, _ = fixed.reset(seed=2)
    np.testing.assert_array_equal(original, repeated)


def test_reset_rewinds_counters_pipes_and_rng_after_play():
    env = FlappyEnv()
    initial, initial_info = env.reset(seed=42)
    original_pipes = [(p.x, p.opening, p.passed) for p in env.pipes]
    for action in [1, 0, 0, 0]:
        env.step(action)
    reset, info = env.reset(seed=42)
    np.testing.assert_array_equal(initial, reset)
    assert info == initial_info
    assert original_pipes == [(p.x, p.opening, p.passed) for p in env.pipes]
