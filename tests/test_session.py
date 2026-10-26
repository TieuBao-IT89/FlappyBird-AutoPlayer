import cv2
import numpy as np

from flappy_auto.session import RunTracker, game_over_banner, name_dialog_close
from flappy_auto.vision import Bird, Observation, Pipe


def scene(pipes, bird=True):
    return Observation(Bird(180, 300, 14) if bird else None,
                       pipes[0] if pipes else None, 480, 720, tuple(pipes))


def test_counts_each_pipe_once_despite_jitter_and_a_missing_frame():
    tracker = RunTracker()
    tracker.update(scene([Pipe(165, 231, 180, 340), Pipe(445, 480, 240, 400)]), 1)
    tracker.update(scene([]), 1.03)
    tracker.update(scene([Pipe(140, 206, 181, 341), Pipe(420, 480, 240, 400)]), 1.1)
    assert tracker.score == 1
    tracker.update(scene([Pipe(145, 211, 180, 340)]), 1.13)
    tracker.update(scene([Pipe(136, 202, 180, 340)]), 1.16)
    assert tracker.score == 1
    tracker.update(scene([Pipe(85, 151, 180, 340)]), 1.5)
    tracker.update(scene([Pipe(360, 426, 240, 400)]), 1.7)
    tracker.update(scene([Pipe(168, 234, 240, 400)]), 2)
    tracker.update(scene([Pipe(142, 208, 240, 400)]), 2.1)
    assert tracker.score == 2


def test_counts_crossing_when_detection_skips_to_the_next_pipe():
    tracker = RunTracker()
    tracker.update(scene([Pipe(170, 236, 180, 340)]), 1)
    tracker.update(scene([]), 1.2)
    tracker.update(scene([Pipe(385, 451, 240, 400)]), 1.4)
    assert tracker.score == 1


def test_missing_bird_during_active_game_ends_the_run():
    tracker = RunTracker()
    tracker.update(scene([Pipe(250, 316, 180, 340)]), 1)
    assert tracker.update(scene([], bird=False), 1.2) is None
    assert tracker.update(scene([], bird=False), 1.8) == 'bird lost'


def test_frozen_game_ends_run_but_waiting_on_start_does_not():
    tracker = RunTracker()
    assert tracker.update(scene([]), 0) is None
    assert tracker.update(scene([]), 10) is None
    pipe = Pipe(240, 306, 180, 340)
    active = RunTracker()
    active.update(scene([pipe]), 1)
    assert active.update(scene([pipe]), 2.3) == 'game stopped moving'


def test_red_game_over_banner_ignores_beak_and_lower_buttons():
    frame = np.zeros((720, 480, 3), np.uint8)
    cv2.rectangle(frame, (180, 150), (200, 160), (0, 0, 230), -1)
    cv2.rectangle(frame, (80, 470), (400, 520), (0, 0, 230), -1)
    assert not game_over_banner(frame)
    cv2.putText(frame, 'GAME OVER', (70, 178), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 230), 6)
    assert game_over_banner(frame)


def test_locates_name_dialog_close_and_ignores_separate_score_panels():
    frame = np.zeros((720, 480, 3), np.uint8)
    cream = (205, 231, 250)
    cv2.rectangle(frame, (15, 194), (228, 440), cream, -1)
    cv2.rectangle(frame, (254, 194), (464, 440), cream, -1)
    assert name_dialog_close(frame) is None
    frame = (frame * .4).astype(np.uint8)
    cv2.rectangle(frame, (64, 248), (415, 472), cream, -1)
    cv2.line(frame, (385, 260), (395, 270), (40, 60, 80), 3)
    cv2.line(frame, (395, 260), (385, 270), (40, 60, 80), 3)
    close = name_dialog_close(frame)
    assert close is not None
    assert abs(close[0] - 390) <= 2 and abs(close[1] - 266) <= 2
