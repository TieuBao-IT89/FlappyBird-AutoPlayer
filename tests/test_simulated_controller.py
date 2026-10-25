"""Exercise the control policy against a deterministic, demo-like physics model."""
import math

from flappy_auto.strategy import Strategy
from flappy_auto.vision import Bird, Observation, Pipe


def test_policy_can_navigate_three_pipe_gaps():
    strategy = Strategy()
    x, y, vy = 105.0, 270.0, -260.0
    pipe_positions = [(370.0, 255.0), (580.0, 255 + math.sin(1.7) * 85), (790.0, 255 + math.sin(3.4) * 85)]
    passed = set()
    dt = 1 / 12
    for i in range(120):
        now = (i + 1) * dt
        vy += 740 * dt
        y += vy * dt
        pipe_positions = [(px - 100 * dt, center) for px, center in pipe_positions]
        for index, (px, center) in enumerate(pipe_positions):
            if px + 52 < x:
                passed.add(index)
            elif (x + 12 > px and x - 12 < px + 52
                  and (y - 12 < center - 84 or y + 12 > center + 84)):
                assert False, f'Hit pipe #{index} after {now:.2f} seconds'
        assert y + 12 < 528, f'Hit floor after {now:.2f} seconds'
        next_pipe = next((Pipe(int(px), int(px + 52), int(center - 84), int(center + 84))
                          for px, center in pipe_positions if px + 52 >= x - 12), None)
        frame = Observation(Bird(x, y, 12), next_pipe, 360, 600)
        flap, _ = strategy.should_flap(frame, now)
        if flap:
            vy = -260
    assert len(passed) >= 3
