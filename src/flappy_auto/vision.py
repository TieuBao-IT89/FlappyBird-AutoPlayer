"""Pure OpenCV perception; all coordinates are screenshot pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Bird:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class Pipe:
    left: int
    right: int
    gap_top: int
    gap_bottom: int

    @property
    def gap_center(self) -> float:
        return (self.gap_top + self.gap_bottom) / 2


@dataclass(frozen=True)
class Observation:
    bird: Bird | None
    pipe: Pipe | None
    width: int
    height: int


def _color_mask(hsv: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _bird_from_mask(mask: np.ndarray, previous: Bird | None = None,
                    beak_mask: np.ndarray | None = None) -> Bird | None:
    h, w = mask.shape
    # Ignore the score area and the floor. The bird lives in the left half.
    valid = mask.copy()
    valid[: int(h * .09), :] = 0
    valid[int(h * .86) :, :] = 0
    valid[:, : int(w * .06)] = 0
    valid[:, int(w * .60) :] = 0
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[bool, float, Bird]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, cw, ch = cv2.boundingRect(contour)
        if not (max(12, w * h * .00012) < area < w * h * .015):
            continue
        if cw < 4 or ch < 4 or cw / max(ch, 1) > 3.1 or ch / max(cw, 1) > 3.0:
            continue
        candidate = Bird(x + cw / 2, y + ch / 2, max(cw, ch) / 2)
        if previous and abs(candidate.x - previous.x) > max(previous.radius * 2, w * .07):
            continue
        center_x = previous.x if previous else w * .30
        center_y = previous.y if previous else h * .49
        # Prefer a compact yellow blob where the bird is expected, not UI decorations.
        score = area - abs(candidate.x - center_x) * .48 - abs(candidate.y - center_y) * .055
        has_beak = False
        if beak_mask is not None:
            pad = max(cw, ch)
            nearby = beak_mask[max(0, y - pad // 2):min(h, y + ch + pad // 2),
                               max(0, x - pad // 2):min(w, x + cw + pad // 2)]
            has_beak = np.count_nonzero(nearby) >= max(3, area * .025)
        candidates.append((has_beak, score, candidate))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    result: list[tuple[int, int]] = []
    start = last = int(indices[0])
    for val in indices[1:]:
        val = int(val)
        if val > last + 1:
            result.append((start, last + 1))
            start = val
        last = val
    result.append((start, last + 1))
    return result


def _pipe_from_mask(mask: np.ndarray, bird: Bird | None) -> Pipe | None:
    h, w = mask.shape
    floor = int(h * .88)
    roi = mask[:floor, :]
    # At a pipe x, colored pixels form long sections above and below the gap.
    density = np.count_nonzero(roi, axis=0)
    active_x = np.flatnonzero(density > max(20, floor * .15))
    candidates: list[Pipe] = []
    for left, right in _runs(active_x):
        if right - left < max(9, int(w * .022)):
            continue
        # Sample inner columns to avoid cap overhang and edge shadows.
        a = left + (right - left) // 3
        b = left + 2 * (right - left) // 3
        strip = roi[:, max(0, a):min(w, b + 1)]
        row_counts = np.count_nonzero(strip, axis=1)
        green_rows = np.flatnonzero(row_counts >= max(1, strip.shape[1] * .32))
        row_groups = _runs(green_rows)
        if len(row_groups) < 2:
            continue
        if row_groups[0][0] > int(h * .12) or row_groups[-1][1] < int(h * .70):
            continue
        # Score digits can cut a hole through the upper pipe's green pixels.
        # Find the actual opening instead of treating that hole as its edge.
        gaps = [(upper[1], lower[0]) for upper, lower in zip(row_groups, row_groups[1:])
                if lower[0] - upper[1] >= int(h * .12)]
        if not gaps:
            continue
        gap_top, gap_bottom = max(gaps, key=lambda gap: gap[1] - gap[0])
        candidates.append(Pipe(left, right, gap_top, gap_bottom))
    if not candidates:
        return None
    bx = bird.x if bird else w * .30
    # The relevant pipe is the first one not completely behind the bird.
    ahead = [p for p in candidates if p.right >= bx - (bird.radius if bird else 8)]
    return min(ahead, key=lambda p: p.left) if ahead else None


def detect(frame_bgr: np.ndarray, previous_bird: Bird | None = None) -> Observation:
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels")
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Defaults intended for yellow/orange bird + green pipes. Adjustable here for skins.
    bird_mask = _color_mask(hsv, (13, 75, 90), (33, 255, 255))
    pipe_mask = _color_mask(hsv, (38, 65, 45), (92, 255, 255))
    beak_mask = cv2.inRange(hsv, (0, 110, 80), (12, 255, 255)) | cv2.inRange(hsv, (170, 110, 80), (179, 255, 255))
    bird = _bird_from_mask(bird_mask, previous_bird, beak_mask)
    pipe = _pipe_from_mask(pipe_mask, bird)
    return Observation(bird, pipe, w, h)


def annotate(frame_bgr: np.ndarray, obs: Observation, status: str) -> np.ndarray:
    annotated = frame_bgr.copy()
    if obs.bird:
        b = obs.bird
        cv2.circle(annotated, (round(b.x), round(b.y)), round(b.radius + 4), (0, 0, 255), 2)
    if obs.pipe:
        p = obs.pipe
        cv2.rectangle(annotated, (p.left, p.gap_top), (p.right, p.gap_bottom), (255, 0, 255), 2)
        cv2.line(annotated, (0, round(p.gap_center)), (obs.width, round(p.gap_center)), (255, 0, 255), 1)
    cv2.putText(annotated, status[:65], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
    cv2.putText(annotated, status[:65], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return annotated
