"""Pure OpenCV perception; all coordinates are screenshot pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np

from .profiles import HSVRange, VisionConfig


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
    pipes: tuple[Pipe, ...] = ()


def _color_mask(hsv: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
    mask = hsv_mask(hsv, HSVRange(lower, upper))
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def hsv_mask(hsv: np.ndarray, color: HSVRange) -> np.ndarray:
    """Hue is circular in OpenCV; S and V remain ordinary intervals."""
    lower, upper = color.lower, color.upper
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    return (cv2.inRange(hsv, (0, *lower[1:]), upper)
            | cv2.inRange(hsv, lower, (179, *upper[1:])))


def vision_masks(frame: np.ndarray, config: VisionConfig):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (_color_mask(hsv, config.bird.lower, config.bird.upper),
            _color_mask(hsv, config.pipe.lower, config.pipe.upper),
            hsv_mask(hsv, config.beak) if config.use_beak else None,
            hsv_mask(hsv, config.eye) if config.use_eye else None)


def _bird_from_mask(mask: np.ndarray, previous: Bird | None = None,
                    beak_mask: np.ndarray | None = None,
                    eye_mask: np.ndarray | None = None) -> Bird | None:
    h, w = mask.shape
    # Ignore the score area and the floor. The bird lives in the left half.
    valid = mask.copy()
    valid[: int(h * (.02 if previous else .05)), :] = 0
    valid[int(h * .86) :, :] = 0
    valid[:, : int(w * .06)] = 0
    valid[:, int(w * .60) :] = 0
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, cw, ch = cv2.boundingRect(contour)
        if not (max(12, w * h * .00012) < area < w * h * .015):
            continue
        if cw < 4 or ch < 4 or cw / max(ch, 1) > 3.1 or ch / max(cw, 1) > 3.0:
            continue
        if cw > w * .12 or ch > h * .08:
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
            nearby = beak_mask[max(0, y):min(h, y + ch),
                               max(0, x + cw // 2):min(w, x + cw + max(3, cw // 2))]
            has_beak = np.count_nonzero(nearby) >= max(3, area * .025)
        has_eye = False
        if eye_mask is not None:
            eye = eye_mask[y:y + max(1, int(ch * .65)), x + cw // 3:x + cw]
            has_eye = np.count_nonzero(eye) >= max(3, area * .02)
        candidates.append((has_eye and has_beak, has_beak, has_eye, score, candidate))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[:-1])[-1]


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


def _pipes_from_mask(mask: np.ndarray) -> list[Pipe]:
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
        # The score can cover the entire middle of an upper pipe, including
        # its bottom edge. Use surviving green pixels across the full width,
        # rather than mistaking the white digit's edge for the pipe opening.
        strip = roi[:, left:right]
        row_counts = np.count_nonzero(strip, axis=1)
        green_rows = np.flatnonzero(row_counts >= max(2, strip.shape[1] * .12))
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
    return sorted(candidates, key=lambda pipe: pipe.left)


def _next_pipe(pipes, bird, width):
    if not pipes:
        return None
    bx = bird.x if bird else width * .30
    # The relevant pipe is the first one not completely behind the bird.
    ahead = [p for p in pipes if p.right >= bx - (bird.radius if bird else 8)]
    return min(ahead, key=lambda p: p.left) if ahead else None


def detect(frame_bgr: np.ndarray, previous_bird: Bird | None = None, max_width: int = 320,
           config: VisionConfig | None = None) -> Observation:
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels")
    original_h, original_w = frame_bgr.shape[:2]
    scale = min(1.0, max_width / original_w) if max_width else 1.0
    if scale < 1:
        frame_bgr = cv2.resize(frame_bgr, (max_width, round(original_h * scale)), interpolation=cv2.INTER_AREA)
        if previous_bird:
            previous_bird = Bird(previous_bird.x * scale, previous_bird.y * scale, previous_bird.radius * scale)
    h, w = frame_bgr.shape[:2]
    bird_mask, pipe_mask, beak_mask, eye_mask = vision_masks(frame_bgr, config or VisionConfig())
    bird = _bird_from_mask(bird_mask, previous_bird, beak_mask, eye_mask)
    pipes = _pipes_from_mask(pipe_mask)
    if scale < 1:
        if bird:
            bird = Bird(bird.x / scale, bird.y / scale, bird.radius / scale)
        pipes = [Pipe(round(p.left / scale), round(p.right / scale),
                      round(p.gap_top / scale), round(p.gap_bottom / scale)) for p in pipes]
    pipe = _next_pipe(pipes, bird, original_w)
    return Observation(bird, pipe, original_w, original_h, tuple(pipes))


class VisionTracker:
    """Keep each visible pipe's geometry through temporary score occlusion."""

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        self.previous_bird = None
        self._tracks = []

    def detect(self, frame: np.ndarray, now: float, speed: float = 0) -> Observation:
        obs = detect(frame, self.previous_bird, config=self.config)
        self.previous_bird = obs.bird
        return self.stabilize(obs, now, speed or obs.width * .35)

    def stabilize(self, obs: Observation, now: float, speed: float) -> Observation:
        remaining = [track for track in self._tracks if 0 <= now - track['seen'] < .3]
        tracks, pipes = [], []
        for pipe in obs.pipes or ((obs.pipe,) if obs.pipe else ()):
            matches = [(abs(track['right'] - speed * (now - track['seen']) - pipe.right), index)
                       for index, track in enumerate(remaining)]
            distance, index = min(matches, default=(float('inf'), -1))
            if distance < max(12, obs.width * .06):
                track = remaining.pop(index)
            else:
                track = {'tops': deque(maxlen=120), 'bottoms': deque(maxlen=120)}
            track['tops'].append(pipe.gap_top)
            track['bottoms'].append(pipe.gap_bottom)
            track.update(right=pipe.right, seen=now)
            stable = Pipe(pipe.left, pipe.right, round(float(np.median(track['tops']))),
                          round(float(np.median(track['bottoms']))))
            pipes.append(stable)
            tracks.append(track)
        self._tracks = tracks + remaining
        return Observation(obs.bird, _next_pipe(pipes, obs.bird, obs.width), obs.width, obs.height, tuple(pipes))


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
