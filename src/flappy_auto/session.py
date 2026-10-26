"""Track runs and passed pipes from images, without reading game state."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .vision import Observation, Pipe


def name_dialog_close(frame: np.ndarray) -> tuple[float, float] | None:
    """Locate the close cross of the game's optional name-entry dialog."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cream = cv2.inRange(hsv, (8, 15, 170), (30, 100, 255))
    contours, _ = cv2.findContours(cream, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if not (w * .55 < cw < w * .9 and h * .2 < ch < h * .5
                and abs(x + cw / 2 - w / 2) < w * .1):
            continue
        cx, cy = round(x + cw * .925), round(y + ch * .08)
        patch = hsv[max(0, cy - 8):cy + 9, max(0, cx - 8):cx + 9]
        if np.count_nonzero(patch[:, :, 2] < 140) >= 12:
            return cx, cy
    return None


def game_over_banner(frame: np.ndarray) -> bool:
    """Recognize the public game's broad red/white GAME OVER banner.

    This intentionally ignores lower buttons, the bird's beak and small score
    decorations. Frozen/missing observations provide a fallback for other games.
    """
    h, w = frame.shape[:2]
    region = frame[int(h * .12):int(h * .29), int(w * .15):int(w * .85)]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 160, 130), (8, 255, 255))
    red |= cv2.inRange(hsv, (174, 160, 130), (179, 255, 255))
    # Separate glyphs form a wide band. A single red sprite cannot satisfy it.
    wide_rows = np.count_nonzero(red, axis=1) > w * .28
    return bool(np.count_nonzero(wide_rows) >= max(5, h * .012))


@dataclass
class RunTracker:
    score: int = 0
    _target: Pipe | None = None
    _target_seen: float = 0
    _target_counted: bool = False
    _last_time: float | None = None
    _last_motion: float | None = None
    _last_bird_seen: float | None = None
    _last_bird_y: float | None = None
    _started: float | None = None
    _had_pipes: bool = False

    @property
    def has_played(self) -> bool:
        return self._had_pipes

    def update(self, obs: Observation, now: float) -> str | None:
        """Return a stop reason on missing/frozen gameplay, else None."""
        if self._last_time is not None and now <= self._last_time:
            return None
        if self._started is None:
            self._started = self._last_motion = self._last_bird_seen = now
        self._last_time = now
        if obs.bird:
            self._last_bird_seen = now
            if self._last_bird_y is not None and abs(obs.bird.y - self._last_bird_y) > .5:
                self._last_motion = now
            self._last_bird_y = obs.bird.y
        pipes = obs.pipes or ((obs.pipe,) if obs.pipe else ())
        self._had_pipes |= bool(pipes)
        pipe = obs.pipe
        if pipe is not None:
            switched = self._target is not None and pipe.right - self._target.right > obs.width * .25
            if switched:
                # A switch to the next pipe can confirm a crossing that fell
                # between images, even when the old gap briefly disappeared.
                if (not self._target_counted and obs.bird and now - self._target_seen < .7
                        and self._target.right < obs.bird.x + obs.width * .15):
                    self.score += 1
                self._target_counted = False
            if self._target is not None and abs(pipe.right - self._target.right) > .5:
                self._last_motion = now
            if obs.bird and not self._target_counted and (pipe.left + pipe.right) / 2 < obs.bird.x:
                self.score += 1
                self._target_counted = True
            self._target, self._target_seen = pipe, now
        if self._had_pipes and now - self._last_bird_seen > .7:
            return 'bird lost'
        if self._had_pipes and now - self._last_motion > 1.2:
            return 'game stopped moving'
        return None
