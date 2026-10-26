from dataclasses import replace

import cv2
import numpy as np
import pytest

from flappy_auto.calibration import CalibrationSession, DisplayTransform, patch_sample, sampled_range
from flappy_auto.profiles import GameProfile, HSVRange, VisionConfig
from flappy_auto.vision import VisionTracker, detect, hsv_mask


def alternate_scene():
    hsv = np.zeros((600, 360, 3), np.uint8)
    hsv[:] = (100, 90, 220)
    cv2.circle(hsv, (105, 270), 12, (177, 220, 220), -1)
    cv2.rectangle(hsv, (190, 0), (242, 140), (135, 220, 200), -1)
    cv2.rectangle(hsv, (190, 310), (242, 527), (135, 220, 200), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_samples_alternate_colors_and_reloaded_profile_match_preview(tmp_path):
    image = alternate_scene()
    assert detect(image).bird is None and detect(image).pipe is None
    session = CalibrationSession(image, GameProfile(source='demo'), 'element')
    session.sample('bird', 105, 270)
    session.sample('pipe', 215, 80)
    session.config = replace(session.config, use_beak=False, use_eye=False)
    original, bird_mask, pipe_mask, preview = session.preview()
    assert preview.bird is not None and preview.pipe is not None
    assert original.shape == image.shape
    assert bird_mask.shape == pipe_mask.shape == (533, 320)
    assert session.config.bird.lower[0] > session.config.bird.upper[0]
    path = tmp_path / 'custom.json'
    session.saved_profile().save(path)
    profile = GameProfile.load(path)
    assert detect(image, config=profile.vision) == preview
    assert VisionTracker(profile.vision).detect(image, 0) == preview


def test_samples_inside_crop_use_crop_relative_pixels_and_keep_source_unchanged():
    image = np.pad(alternate_scene(), ((40, 60), (20, 30), (0, 0)))
    session = CalibrationSession(image, GameProfile(), 'viewport')
    session.set_roi((20, 40, 360, 600))
    session.sample('bird', 125, 310)
    session.sample('pipe', 235, 120)
    before = image.copy()
    preview = session.preview()[-1]
    assert abs(preview.bird.x-105) <= 2
    assert abs(preview.pipe.left-190) <= 2
    assert np.array_equal(session.frame, before)
    with pytest.raises(ValueError, match='bên trong'):
        session.sample('bird', 0, 0)
    session.set_roi(None)
    assert not session.samples['bird']


def test_hue_wraparound_includes_red_and_excludes_other_colors():
    hsv = np.array([[(0, 200, 220), (179, 200, 220), (175, 200, 220), (5, 200, 220),
                     (90, 200, 220), (0, 20, 220), (0, 200, 10)]], np.uint8)
    color = HSVRange((170, 100, 80), (10, 255, 255))
    assert hsv_mask(hsv, color).tolist() == [[255, 255, 255, 255, 0, 0, 0]]


def test_patch_sampling_resists_outlier_pixels_and_red_wrap():
    hsv = np.full((5, 5, 3), (179, 200, 210), np.uint8)
    hsv[:2, :, 0] = 0
    hsv[2, 2] = (90, 0, 0)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hue, saturation, value = patch_sample(image, 2, 2)
    assert hue in (0, 179) and saturation >= 198 and value == 210
    color = sampled_range([(179, 200, 210), (0, 205, 215)])
    assert color.lower[0] == 169 and color.upper[0] == 10
    assert color.lower[1] == 145 and color.upper[1] == 255
    assert patch_sample(image, 0, 0)[0] in (0, 179)


@pytest.mark.parametrize('display', [(360, 600), (720, 1200), (540, 900), (900, 600), (300, 900)])
def test_display_to_css_coordinates_with_dpi_and_letterboxing(display):
    transform = DisplayTransform((360, 600), display)
    ox, oy = transform.offset
    actual = transform.image_point(105*transform.scale+ox+.01, 270*transform.scale+oy+.01)
    assert actual == (105, 270)
    assert transform.image_point(-1, -1) is None
    assert transform.image_point(1e6, 1e6, clamp=True) == (359, 599)


def test_frame_size_change_and_bad_roi_are_rejected():
    session = CalibrationSession(alternate_scene(), GameProfile(), 'element')
    with pytest.raises(ValueError, match='calibrate'):
        session.update(np.zeros((1200, 720, 3), np.uint8))
    with pytest.raises(ValueError, match='outside'):
        session.set_roi((0, 0, 400, 600))


def test_disabled_eye_and_beak_cues_change_candidate_ranking():
    image = np.zeros((600, 360, 3), np.uint8)
    cv2.circle(image, (105, 270), 12, (40, 210, 245), -1)
    cv2.circle(image, (60, 140), 10, (40, 210, 245), -1)
    cv2.circle(image, (64, 136), 3, (255, 255, 255), -1)
    cv2.rectangle(image, (66, 141), (75, 146), (20, 30, 220), -1)
    assert detect(image).bird.y < 150
    config = VisionConfig(use_eye=False, use_beak=False)
    assert abs(detect(image, config=config).bird.y-270) < 2
