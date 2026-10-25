from flappy_auto.strategy import Strategy
from flappy_auto.vision import Bird, Observation, Pipe


def observation(y, pipe=None):
    return Observation(Bird(105, y, 12), pipe, 360, 600)


def test_falls_too_low_causes_flap():
    strategy = Strategy()
    decision, status = strategy.should_flap(observation(410, Pipe(160, 212, 160, 330)), 1)
    assert decision is True
    assert 'flap' in status


def test_cooldown_prevents_rapid_flap():
    strategy = Strategy()
    assert strategy.should_flap(observation(430), 1)[0]
    assert not strategy.should_flap(observation(450), 1.03)[0]


def test_no_bird_never_flaps():
    strategy = Strategy()
    assert not strategy.should_flap(Observation(None, None, 360, 600), 1)[0]


def test_high_bird_coasts():
    strategy = Strategy()
    assert not strategy.should_flap(observation(150, Pipe(140, 185, 190, 370)), 1)[0]
