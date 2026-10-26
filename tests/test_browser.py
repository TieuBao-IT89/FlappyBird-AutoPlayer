from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from flappy_auto.browser import GameBrowser
from flappy_auto.profiles import GameProfile


def test_compositor_timestamps_preserve_motion_when_delivery_is_jittery(monkeypatch):
    received = iter([100., 100.045, 100.060])
    monkeypatch.setattr('flappy_auto.browser.time.perf_counter', lambda: next(received))
    browser = GameBrowser(SimpleNamespace())
    browser._session = SimpleNamespace(send=lambda method, parameters: None)
    for index, timestamp in enumerate([1700000000., 1700000000.016, 1700000000.032]):
        browser._on_frame({'data': str(index), 'sessionId': index, 'metadata': {'timestamp': timestamp}})
    assert browser._latest_time == pytest.approx(100.032, abs=.00001)
    # Receiving a new frame must not change the timestamp of the image that
    # the controller has already consumed.
    assert browser.frame_time == 0


def test_restart_uses_visible_play_button_with_canvas_and_roi_offsets():
    clicks = []
    page = SimpleNamespace(mouse=SimpleNamespace(click=lambda x, y: clicks.append((x, y))))
    browser = GameBrowser(page)
    browser._bounds = {'x': 20, 'y': 40, 'width': 520, 'height': 800}
    frame = np.zeros((720, 480, 3), np.uint8)
    cv2.rectangle(frame, (260, 470), (400, 520), (40, 190, 80), -1)
    cv2.putText(frame, 'PLAY', (275, 505), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    # A green pipe must not be mistaken for the retry button.
    cv2.rectangle(frame, (50, 380), (110, 700), (40, 190, 80), -1)
    browser.restart(frame, (10, 15, 480, 720))
    assert len(clicks) == 1
    assert abs(clicks[0][0] - 360) < 2
    assert abs(clicks[0][1] - 550) < 2


def test_restart_without_a_button_uses_space():
    keys = []
    browser = GameBrowser(SimpleNamespace(keyboard=SimpleNamespace(press=keys.append)))
    browser.restart(np.zeros((600, 360, 3), np.uint8))
    assert keys == ['Space']


def test_restart_closes_name_dialog_before_clicking_play(monkeypatch):
    clicks = []
    page = SimpleNamespace(mouse=SimpleNamespace(click=lambda x, y: clicks.append((x, y))),
                           wait_for_timeout=lambda delay: None,
                           viewport_size={'width': 480, 'height': 720})
    browser = GameBrowser(page)
    dialog = np.zeros((720, 480, 3), np.uint8)
    cv2.rectangle(dialog, (64, 248), (415, 472), (205, 231, 250), -1)
    cv2.line(dialog, (385, 260), (395, 270), (40, 60, 80), 3)
    cv2.line(dialog, (395, 260), (385, 270), (40, 60, 80), 3)
    ready = np.zeros_like(dialog)
    cv2.rectangle(ready, (260, 470), (400, 520), (40, 190, 80), -1)
    monkeypatch.setattr(browser, 'capture', lambda roi=None: ready)
    browser.restart(dialog)
    assert len(clicks) == 2
    assert abs(clicks[0][0] - 390) < 3 and abs(clicks[0][1] - 266) < 3
    assert abs(clicks[1][0] - 330) < 2 and abs(clicks[1][1] - 495) < 2


@pytest.mark.parametrize('scale', [1, 1.25, 1.5, 2])
def test_capture_normalizes_high_dpi_before_applying_roi(scale):
    import base64
    viewport = {'width': 480, 'height': 720}
    image = np.zeros((round(720*scale), round(480*scale), 3), np.uint8)
    image[round(100*scale):round(300*scale), round(60*scale):round(160*scale)] = (20, 200, 50)
    _, encoded = cv2.imencode('.png', image)
    browser = GameBrowser(SimpleNamespace(viewport_size=viewport))
    browser._bounds = {'x': 20, 'y': 40, 'width': 360, 'height': 600}
    browser._streaming = True
    browser._sequence = 1
    browser._latest = base64.b64encode(encoded).decode()
    crop = browser.capture((40, 60, 100, 200))
    assert crop.shape == (200, 100, 3)
    assert np.allclose(crop[5:-5, 5:-5], (20, 200, 50))


def test_profile_detects_changed_element_bounds_and_disappearing_element():
    bounds = {'x': 0, 'y': 0, 'width': 360, 'height': 600}
    browser = GameBrowser(SimpleNamespace(viewport_size={'width': 480, 'height': 720}))
    browser.game_area = SimpleNamespace(bounding_box=lambda: bounds)
    browser.calibration_profile = GameProfile(reference_size=(360, 600))
    assert browser._rectangle(None) == (0, 0, 360, 600)
    bounds['width'] = 400
    browser._geometry_checked = 0
    with pytest.raises(ValueError, match='calibrate'):
        browser._rectangle(None)
    browser.game_area = SimpleNamespace(bounding_box=lambda: None)
    with pytest.raises(ValueError, match='no longer visible'):
        browser.geometry()


def test_roi_selected_on_rounded_css_image_accepts_fractional_element_bounds():
    browser = GameBrowser(SimpleNamespace(viewport_size={'width': 480, 'height': 720}))
    browser._bounds = {'x': 10.25, 'y': 20.25, 'width': 356.75, 'height': 586.75}
    assert browser._rectangle((0, 0, 357, 587)) == (10.25, 20.25, 357, 587)
    with pytest.raises(ValueError, match='outside'):
        browser._rectangle((0, 0, 358, 587))
