from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('gymnasium')

from flappy_auto.__main__ import make_controller, make_parser
from flappy_auto.profiles import VisionConfig
from flappy_auto.rl.browser_policy import DQNController
from flappy_auto.strategy import Strategy
from flappy_auto.vision import Bird, Observation, Pipe


def scene(y=270):
    return Observation(Bird(105, y, 12), Pipe(200, 252, 171, 339), 360, 600)


def controller(action=1):
    calls = []
    clock = SimpleNamespace(now=1.01)

    def predict(obs, deterministic):
        calls.append((obs.copy(), deterministic))
        return np.array(action), None

    return DQNController(SimpleNamespace(predict=predict), clock=lambda: clock.now), calls, clock


@pytest.mark.parametrize('action', [0, 1])
def test_deterministic_actions_cadence_and_reset(action):
    policy, calls, clock = controller(action)
    assert not policy.should_flap(scene(), 1)[0]
    clock.now = 1.03
    assert policy.should_flap(scene(272), 1.02)[0] == bool(action)
    assert calls[0][1] is True and calls[0][0].dtype == np.float32
    clock.now = 1.04
    assert not policy.should_flap(scene(273), 1.03)[0]
    assert len(calls) == 1
    new = policy.new_run()
    assert new.model is policy.model
    clock.now = 2.01
    assert not new.should_flap(scene(), 2)[0]
    assert len(calls) == 1


def test_slow_inference_and_unreliable_frames_never_send_flap():
    policy, calls, clock = controller()
    policy.should_flap(scene(), 1)

    def slow(obs, deterministic):
        clock.now += .3
        return np.array(1), None

    policy.model.predict = slow
    clock.now = 1.03
    assert not policy.should_flap(scene(272), 1.02)[0]
    assert policy.policy_decision['action'] == 1
    assert policy.policy_decision['reason'] == 'inference exceeded frame age'
    assert not policy.action_is_fresh(1.02, 1.215, .008)
    policy.model.predict = lambda *a, **k: pytest.fail('must not infer')
    policy.should_flap(Observation(None, None, 360, 600), 1.34)


@pytest.mark.parametrize('action', [2, .5, float('nan'), [0, 1]])
def test_invalid_actions_fail_closed(action):
    policy, calls, clock = controller(action)
    policy.should_flap(scene(), 1)
    clock.now = 1.03
    with pytest.raises(ValueError, match='invalid action'):
        policy.should_flap(scene(272), 1.02)


def test_cli_default_validation_and_verified_loader(monkeypatch):
    parser = make_parser()
    assert type(make_controller(parser.parse_args([]), VisionConfig())) is Strategy
    with pytest.raises(ValueError, match='requires --model'):
        make_controller(parser.parse_args(['--policy', 'dqn']), VisionConfig())
    with pytest.raises(ValueError, match='requires --policy'):
        make_controller(parser.parse_args(['--model', 'x.zip']), VisionConfig())
    calls = []

    def load(path):
        calls.append(str(path))
        return object(), {'model_sha256': 'verified'}

    monkeypatch.setattr('flappy_auto.rl.dqn.load_checkpoint', load)
    policy = make_controller(parser.parse_args(['--policy', 'dqn', '--model', 'x.zip']), VisionConfig())
    assert calls == ['x.zip'] and policy.checkpoint_metadata['model_sha256'] == 'verified'
