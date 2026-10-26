"""Visual-only browser adapter. No reads from game JS state."""
from __future__ import annotations

import base64
from contextlib import suppress
import re
import time

import cv2
import numpy as np
from playwright.sync_api import BrowserContext, Error, Page

from .session import name_dialog_close


def block_advertising(context: BrowserContext) -> None:
    """Stop ad loaders observed on the public game, before they spawn iframes."""
    domains = (
        'vntsm.com', 'vntsm.io', 'venatusmedia.com', 'googletagmanager.com',
        'googlesyndication.com', 'doubleclick.net', 'primis.tech',
        'atmtd.com', 'ccgateway.net', 'posthog.com',
    )
    pattern = re.compile(r'^https?://(?:[^/]+\.)?(?:' + '|'.join(re.escape(domain) for domain in domains) + r')(?::\d+)?/')
    context.route(pattern, lambda route: route.abort())


class GameBrowser:
    def __init__(self, page: Page, selector: str = 'canvas'):
        self.page = page
        self.selector = selector
        self.game_area = None
        self._bounds = None
        self._session = None
        self._latest = None
        self._latest_time = 0.0
        self._clock_offset = None
        self._sequence = 0
        self._consumed = 0
        self._streaming = False
        self._screenshot_scale = 1.0
        self.frame_time = 0.0
        self.calibration_profile = None
        self._geometry_checked = 0.0

    @property
    def region_kind(self):
        return 'element' if self.game_area is not None else 'viewport'

    def geometry(self):
        """Refresh CSS bounds, including layout changes since initial discovery."""
        self._bounds = self.game_area.bounding_box() if self.game_area is not None else None
        if self.game_area is not None and self._bounds is None:
            raise ValueError('Game element is no longer visible. Run --calibrate again.')
        bounds = self._bounds or self.page.viewport_size
        return round(bounds['width']), round(bounds['height'])

    def locate(self) -> str:
        # Supports canvas in the main document AND inside embedded iframes.
        best, best_area = None, 0
        for frame in self.page.frames:
            try:
                locator = frame.locator(self.selector)
                count = min(locator.count(), 10)
                for i in range(count):
                    item = locator.nth(i)
                    bounds = item.bounding_box(timeout=1500)
                    if bounds and bounds['width'] > 100 and bounds['height'] > 100:
                        area = bounds['width'] * bounds['height']
                        if area > best_area:
                            best, best_area = item, area
            except Exception:
                continue
        self.game_area = best
        self._bounds = best.bounding_box() if best is not None else None
        if self._session is None:
            self._session = self.page.context.new_cdp_session(self.page)
            with suppress(Error):
                self._session.send('Emulation.setFocusEmulationEnabled', {'enabled': True})
            self._session.on('Page.screencastFrame', self._on_frame)
            viewport = self.page.viewport_size
            self._session.send('Page.startScreencast', {
                'format': 'jpeg', 'quality': 80, 'everyNthFrame': 1,
                'maxWidth': viewport['width'], 'maxHeight': viewport['height'],
            })
            self._streaming = True
        return 'game element' if best is not None else 'viewport (fallback; use --roi to crop)'

    def _on_frame(self, event: dict) -> None:
        # Keep only the latest frame, so slow consumers cannot build a backlog.
        self._latest = event['data']
        received = time.perf_counter()
        timestamp = event.get('metadata', {}).get('timestamp')
        if timestamp is not None:
            # Synchronize the compositor clock with a high-resolution local
            # clock. Python 3.10/3.11 monotonic/time on Windows can advance in
            # 15.6 ms chunks, corrupting both frame age and fitted velocity.
            offset = received - timestamp
            if self._clock_offset is None or abs(offset - self._clock_offset) > 2:
                self._clock_offset = offset
            else:
                self._clock_offset = min(self._clock_offset, offset)
            self._latest_time = timestamp + self._clock_offset
        else:
            self._latest_time = received
        self._sequence += 1
        with suppress(Error):
            self._session.send('Page.screencastFrameAck', {'sessionId': event['sessionId']})

    def _rectangle(self, roi):
        if self.calibration_profile is not None and time.perf_counter() - self._geometry_checked >= 1:
            self.calibration_profile.validate_capture(self.geometry(), self.region_kind)
            self._geometry_checked = time.perf_counter()
        bounds = self._bounds or {**self.page.viewport_size, 'x': 0, 'y': 0}
        x, y, width, height = (bounds[key] for key in ('x', 'y', 'width', 'height'))
        if roi:
            rx, ry, width, height = roi
            # ROI is selected on the normalized, integer-sized CSS image.
            # Fractional element bounds must use the same rounding as _resize.
            if rx < 0 or ry < 0 or width <= 0 or height <= 0 or rx + width > round(bounds['width']) or ry + height > round(bounds['height']):
                raise ValueError(f'ROI {roi} outside screenshot dimensions {bounds["width"]}x{bounds["height"]}')
            x, y = x + rx, y + ry
        return x, y, width, height

    def capture(self, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        x, y, width, height = self._rectangle(roi)
        viewport = self.page.viewport_size
        visible = (x >= 0 and y >= 0 and x + width <= viewport['width'] + 1
                   and y + height <= viewport['height'] + 1)
        if self._streaming and visible:
            # Pump Playwright's event loop while waiting for a *new* frame.
            # Reusing the last image here would inflate FPS and corrupt velocity.
            if self._sequence == self._consumed:
                self.page.wait_for_timeout(1)
            deadline = time.perf_counter() + .15
            while self._sequence == self._consumed and time.perf_counter() < deadline:
                self.page.wait_for_timeout(1)
            if self._latest is not None and self._sequence != self._consumed:
                self._consumed = self._sequence
                self.frame_time = self._latest_time
                image = self._decode(self._latest)
                sx, sy = image.shape[1] / viewport['width'], image.shape[0] / viewport['height']
                image = image[round(y * sy):round((y + height) * sy),
                              round(x * sx):round((x + width) * sx)]
                return self._resize(image, width, height)
        # Still pages do not emit screencast frames. Also support canvases that
        # extend outside the viewport without cutting off part of the arena.
        if self._session is None:
            self._session = self.page.context.new_cdp_session(self.page)
        captured_at = time.perf_counter()
        result = self._session.send('Page.captureScreenshot', {
            'format': 'jpeg', 'quality': 80, 'captureBeyondViewport': True,
            'clip': {'x': x, 'y': y, 'width': width, 'height': height, 'scale': self._screenshot_scale},
        })
        self.frame_time = captured_at
        image = self._decode(result['data'])
        # Headed Chromium may capture Windows display pixels at 125/150% DPI.
        if image.shape[1] != round(width):
            self._screenshot_scale *= width / image.shape[1]
        return self._resize(image, width, height)

    @staticmethod
    def _decode(encoded):
        raw = base64.b64decode(encoded)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Could not decode browser screenshot')
        return image

    @staticmethod
    def _resize(image, width, height):
        size = (round(width), round(height))
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image

    def close(self):
        if self._session is not None:
            with suppress(Error):
                if self._streaming:
                    self._session.send('Page.stopScreencast')
                self._session.detach()
            self._streaming = False

    def start(self) -> None:
        if self.game_area is not None:
            try:
                self.game_area.click(timeout=3000)
                return
            except Exception:
                pass
        self.page.keyboard.press('Space')

    def flap(self) -> None:
        self.page.keyboard.press('Space')

    def restart(self, frame: np.ndarray, roi=None) -> None:
        """Click a visible green PLAY button, or use the game's Space retry."""
        close = name_dialog_close(frame)
        if close is not None:
            x, y, _, _ = self._rectangle(roi)
            self.page.mouse.click(x + close[0], y + close[1])
            self.page.wait_for_timeout(250)
            frame = self.capture(roi)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (32, 90, 80), (90, 255, 255))
        h, w = mask.shape
        mask[:int(h * .5)] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        buttons = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if w * .2 < bw < w * .6 and h * .035 < bh < h * .14 and 2 < bw / bh < 5:
                buttons.append((cv2.contourArea(contour), x + bw / 2, y + bh / 2))
        if buttons:
            _, bx, by = max(buttons)
            x, y, _, _ = self._rectangle(roi)
            self.page.mouse.click(x + bx, y + by)
        else:
            self.page.keyboard.press('Space')
