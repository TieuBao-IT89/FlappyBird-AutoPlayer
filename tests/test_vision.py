import cv2
import numpy as np

from flappy_auto.vision import Bird, detect


def sample_scene(top=140, bottom=310, pipe_x=190):
    image = np.full((600, 360, 3), (240, 200, 125), dtype=np.uint8)
    cv2.rectangle(image, (pipe_x, 0), (pipe_x + 52, top), (55, 180, 55), -1)
    cv2.rectangle(image, (pipe_x, bottom), (pipe_x + 52, 527), (55, 180, 55), -1)
    cv2.circle(image, (105, 270), 12, (40, 210, 245), -1)
    return image


def test_detect_bird_and_next_pipe():
    obs = detect(sample_scene())
    assert obs.bird is not None
    assert abs(obs.bird.x - 105) <= 2
    assert abs(obs.bird.y - 270) <= 2
    assert obs.pipe is not None
    assert abs(obs.pipe.left - 190) <= 2
    assert abs(obs.pipe.gap_top - 141) <= 2
    assert abs(obs.pipe.gap_bottom - 310) <= 2


def test_no_objects_means_no_detection():
    image = np.zeros((600, 360, 3), dtype=np.uint8)
    obs = detect(image)
    assert obs.bird is None and obs.pipe is None


def test_picks_nearest_approaching_pipe():
    image = sample_scene(pipe_x=30)
    cv2.rectangle(image, (215, 0), (267, 185), (55, 180, 55), -1)
    cv2.rectangle(image, (215, 355), (267, 527), (55, 180, 55), -1)
    obs = detect(image)
    assert obs.pipe is not None
    assert obs.pipe.left == 215


def test_rejects_bad_frame():
    try:
        detect(np.zeros((10, 10), dtype=np.uint8))
        assert False, 'Should reject non-BGR'
    except ValueError:
        pass


def test_score_occluding_upper_pipe_does_not_move_gap():
    image = sample_scene(top=180, bottom=350)
    # The live game's score is drawn on top of pipes crossing the screen centre.
    cv2.rectangle(image, (190, 30), (242, 70), (255, 255, 255), -1)
    obs = detect(image)
    assert obs.pipe is not None
    assert obs.pipe.gap_top == 181
    assert obs.pipe.gap_bottom == 350


def test_bird_beak_disambiguates_large_yellow_title():
    image = sample_scene()
    cv2.rectangle(image, (60, 130), (100, 165), (40, 210, 245), -1)
    cv2.rectangle(image, (113, 270), (122, 276), (20, 30, 220), -1)
    obs = detect(image)
    assert obs.bird is not None
    assert abs(obs.bird.x - 105) < 3
    assert abs(obs.bird.y - 270) < 3


def test_tracking_does_not_jump_to_yellow_pipe_decoration():
    image = sample_scene()
    cv2.rectangle(image, (170, 250), (205, 285), (40, 210, 245), -1)
    obs = detect(image, Bird(105, 265, 12))
    assert obs.bird is not None
    assert abs(obs.bird.x - 105) < 3


def test_bird_still_detected_after_jpeg_capture():
    image = sample_scene()
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    obs = detect(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    assert obs.bird is not None and obs.pipe is not None
    assert abs(obs.bird.x - 105) < 3
    assert abs(obs.pipe.gap_top - 141) < 4
