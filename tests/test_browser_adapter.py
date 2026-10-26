from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('gymnasium')

from flappy_auto.rl.browser_adapter import BrowserObservationAdapter, HIGH, LOW, to_dqn_observation
from flappy_auto.rl.environment import FlappyEnv
from flappy_auto.vision import Bird, Observation, Pipe, TrackQuality


def scene(y=270, scale=1):
    return Observation(Bird(105*scale, y*scale, 12*scale), Pipe(200*scale, 252*scale, 171*scale, 339*scale),
                       360*scale, 600*scale)


def test_contract_matches_environment_and_resolution_does_not_change_features():
    env = FlappyEnv(randomize_gaps=False)
    env.reset(seed=1)
    env.pipes = [env._new_pipe(200)]
    env.pipes[0].opening = 255
    env.bird_vy = -125
    expected = env._observation()
    for scale in (1, 2, .5):
        actual = to_dqn_observation(scene(scale=scale), -125*scale)
        assert actual.shape == (5,) and actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected)
        assert env.observation_space.contains(actual)


def test_target_is_sorted_and_kept_until_radius_has_cleared_trailing_edge():
    obs = scene()
    behind, overlapping, ahead = Pipe(20, 92, 50, 200), Pipe(40, 94, 100, 250), Pipe(200, 252, 171, 339)
    obs = replace(obs, pipe=ahead, pipes=(ahead, behind, overlapping))
    state = to_dqn_observation(obs, 0)
    assert state[2] == pytest.approx((94-105)/360)
    assert state[3] == pytest.approx(100/600)
    assert to_dqn_observation(replace(obs, pipes=(behind,)), 0) is None


@pytest.mark.parametrize('change', [
    {'bird': None}, {'pipe': None}, {'width': 0}, {'height': -1}, {'fresh': False},
    {'bird': Bird(105, float('nan'), 12)}, {'bird': Bird(105, 270, 0)},
    {'pipe': Pipe(250, 200, 100, 300)}, {'pipe': Pipe(100, 200, 400, 300)},
    {'bird_quality': TrackQuality(.9, 'recovering')},
    {'pipe_quality': TrackQuality(.9, predicted=True)},
])
def test_rejects_invalid_or_missing_measurements(change):
    assert to_dqn_observation(replace(scene(), **change), 0) is None


def test_finite_velocity_is_clipped_and_nan_is_rejected():
    assert to_dqn_observation(scene(), float('nan')) is None
    for speed in (-1e6, 1e6):
        state = to_dqn_observation(scene(), speed)
        assert np.all(state >= LOW) and np.all(state <= HIGH)


def test_velocity_uses_actual_frame_intervals_and_resets_between_episodes():
    adapter = BrowserObservationAdapter()
    assert adapter.update(scene(250), 1, now=1.01) is None
    state = adapter.update(scene(252), 1.02, now=1.03)
    assert adapter.velocity == pytest.approx(100)
    assert state[1] == pytest.approx(.25)
    adapter.update(scene(256), 1.06, now=1.07)
    assert adapter.velocity == pytest.approx(100)
    adapter.reset()
    assert adapter.update(scene(300), 2, now=2.01) is None


@pytest.mark.parametrize('stamp,wall', [(1, 1.02), (.9, 1.02), (1.02, 1.5), (2, 1.5), (float('nan'), 1.5)])
def test_bad_timestamps_block_and_require_velocity_warmup(stamp, wall):
    adapter = BrowserObservationAdapter()
    adapter.update(scene(), 1, now=1.01)
    assert adapter.update(scene(), stamp, now=wall) is None
    assert adapter.update(scene(), 3, now=3.01) is None
    assert adapter.update(scene(272), 3.02, now=3.03) is not None


def test_low_confidence_frame_gap_resize_and_teleport_cannot_trigger_inference():
    adapter = BrowserObservationAdapter()
    adapter.update(scene(), 1, now=1.01)
    bad = replace(scene(), bird_quality=TrackQuality(.1))
    assert adapter.update(bad, 1.02, now=1.03) is None
    assert adapter.update(scene(), 1.04, now=1.05) is None
    assert adapter.update(scene(), 1.4, now=1.41) is None
    assert adapter.update(scene(scale=2), 1.42, now=1.43) is None
    assert adapter.update(scene(y=500, scale=2), 1.44, now=1.45) is None


def test_low_pipe_confidence_is_blocked_even_when_track_state_claims_tracking():
    adapter = BrowserObservationAdapter()
    adapter.update(scene(), 1, now=1.01)
    assert adapter.update(replace(scene(), pipe_quality=TrackQuality(.1)), 1.02, now=1.03) is None
