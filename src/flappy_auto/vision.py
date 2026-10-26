"""Pure OpenCV perception; all coordinates are screenshot pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import deque
import math
import zlib

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
class TrackQuality:
    """Heuristic evidence score, not a calibrated probability of correctness."""
    confidence: float = 1.0
    state: str = 'tracking'
    reason: str = 'provided'
    predicted: bool = False
    age: float = 0.0

    @property
    def trusted(self):
        return self.state == 'tracking' and not self.predicted


@dataclass(frozen=True)
class Observation:
    bird: Bird | None
    pipe: Pipe | None
    width: int
    height: int
    pipes: tuple[Pipe, ...] = ()
    bird_quality: TrackQuality = field(default_factory=TrackQuality)
    pipe_quality: TrackQuality = field(default_factory=TrackQuality)
    fresh: bool = True

    @property
    def state(self):
        states = ('tracking', 'uncertain', 'recovering', 'lost')
        return max((self.bird_quality.state, self.pipe_quality.state), key=states.index)

    @property
    def can_control(self):
        return self.fresh and self.bird is not None and self.bird_quality.trusted and self.pipe_quality.trusted

    @property
    def can_learn(self):
        return self.can_control


@dataclass(frozen=True)
class _BirdCandidate:
    bird: Bird
    quality: TrackQuality
    rank: tuple


def image_edge_margin(width: int) -> float:
    """Allow canvas borders and resize rounding around a clipped pipe edge."""
    return max(6., width*.025)


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


def _bird_candidates(mask: np.ndarray, previous: Bird | None = None,
                    beak_mask: np.ndarray | None = None,
                    eye_mask: np.ndarray | None = None, search_all=False) -> list[_BirdCandidate]:
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
        if previous and not search_all and abs(candidate.x - previous.x) > max(previous.radius * 2, w * .07):
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
        color = min(1., np.count_nonzero(mask[y:y+ch, x:x+cw]) / (cw*ch*.70))
        perimeter = cv2.arcLength(contour, True)
        shape = min(1., 4*math.pi*area / max(1., perimeter**2) / .8)
        size = min(1., cw / (w*.045), ch / (h*.028))
        position = max(0., 1-abs(candidate.x-center_x) / (w*.3))
        cues = ((int(has_beak) if beak_mask is not None else 1) + (int(has_eye) if eye_mask is not None else 1)) / 2
        confidence = float(.30*color + .25*shape + .20*size + .15*position + .10*cues)
        reason = f'color={color:.2f} shape={shape:.2f} size={size:.2f} position={position:.2f} cues={cues:.2f}'
        candidates.append(_BirdCandidate(candidate, TrackQuality(confidence, reason=reason),
                                         (has_eye and has_beak, has_beak, has_eye, score)))
    return sorted(candidates, key=lambda item: item.rank, reverse=True)


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


def _analyze(frame_bgr: np.ndarray, previous_bird: Bird | None = None, max_width: int = 320,
             config: VisionConfig | None = None, search_all=False):
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.size:
        raise ValueError("Expected a BGR image with 3 channels")
    original_h, original_w = frame_bgr.shape[:2]
    scale = min(1.0, max_width / original_w) if max_width else 1.0
    if scale < 1:
        frame_bgr = cv2.resize(frame_bgr, (max_width, round(original_h * scale)), interpolation=cv2.INTER_AREA)
        if previous_bird:
            previous_bird = Bird(previous_bird.x * scale, previous_bird.y * scale, previous_bird.radius * scale)
    h, w = frame_bgr.shape[:2]
    config = config or VisionConfig()
    bird_mask, pipe_mask, beak_mask, eye_mask = vision_masks(frame_bgr, config)
    candidates = _bird_candidates(bird_mask, previous_bird, beak_mask, eye_mask, search_all)
    pipes = _pipes_from_mask(pipe_mask)
    pipe_scores = []
    for p in pipes:
        sections = (pipe_mask[:p.gap_top, p.left:p.right], pipe_mask[p.gap_bottom:int(h*.88), p.left:p.right])
        support = min(float(np.count_nonzero(s)) / max(1, s.size) for s in sections)
        opening = (p.gap_bottom-p.gap_top) / h
        gap = max(0., 1-max(0., .18-opening, opening-.45)*3)
        width = min(1., (p.right-p.left) / (w*.06))
        pipe_scores.append((.45*support + .30*gap + .25*width, support, gap, width))
    if scale < 1:
        candidates = [replace(c, bird=Bird(c.bird.x / scale, c.bird.y / scale, c.bird.radius / scale)) for c in candidates]
        pipes = [Pipe(round(p.left / scale), round(p.right / scale),
                      round(p.gap_top / scale), round(p.gap_bottom / scale)) for p in pipes]
    bird = candidates[0].bird if candidates else None
    bird_quality = candidates[0].quality if candidates else TrackQuality(0, 'lost', 'no color/shape candidate')
    if bird_quality.confidence < config.tracking.min_bird_confidence:
        bird_quality = replace(bird_quality, state='uncertain')
    pipe_quality = TrackQuality(0, reason='no visible pipe')
    relevant_scores = [score for p, score in zip(pipes, pipe_scores)
                       if p.right >= (bird.x-bird.radius if bird else original_w*.3-8)]
    if relevant_scores:
        confidence, support, gap, width = min(relevant_scores)
        pipe_quality = TrackQuality(float(confidence), reason=f'color={support:.2f} gap={gap:.2f} width={width:.2f}')
        if confidence < config.tracking.min_pipe_confidence:
            pipe_quality = replace(pipe_quality, state='uncertain')
    pipe = _next_pipe(pipes, bird, original_w)
    return Observation(bird, pipe, original_w, original_h, tuple(pipes), bird_quality, pipe_quality), candidates


def detect(frame_bgr: np.ndarray, previous_bird: Bird | None = None, max_width: int = 320,
           config: VisionConfig | None = None) -> Observation:
    return _analyze(frame_bgr, previous_bird, max_width, config)[0]


class VisionTracker:
    """Bounded recovery from fresh evidence; predictions never authorize control."""

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        self.previous_bird = None
        self._tracks = []
        self._bird_seen = None
        self._bird_confidence = 0.
        self._velocity = 0.
        self._pending_bird = None
        self._bird_history = deque(maxlen=32)
        self._missed = False
        self._last_stamp = float('-inf')
        self._clock = float('-inf')
        self._fingerprint = None
        self._size = None

    def detect(self, frame: np.ndarray, now: float, speed: float = 0,
               *, processed_at: float | None = None, blocked_reason: str | None = None) -> Observation:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError('Expected a nonempty BGR image with 3 channels')
        clock = now if processed_at is None else processed_at
        if not math.isfinite(now) or not math.isfinite(clock):
            raise ValueError('Vision timestamps must be finite')
        self._clock = clock = max(clock, self._clock)
        h, w = frame.shape[:2]
        if self._size is not None and self._size != (w, h):
            raise ValueError('Vision screenshot dimensions changed. Restart or run --calibrate again.')
        self._size = (w, h)
        fingerprint = zlib.crc32(np.ascontiguousarray(frame))
        limits = self.config.tracking
        reason = ('old frame timestamp' if now <= self._last_stamp else
                  'frame too old' if clock-now > limits.max_frame_age else
                  'future frame timestamp' if now > clock+.001 else
                  'repeated image' if fingerprint == self._fingerprint else '')
        if now <= clock+.001:
            self._last_stamp = max(self._last_stamp, now)
        if reason:
            obs = Observation(None, None, w, h, pipe_quality=TrackQuality(0, 'uncertain', reason), fresh=False)
            obs = self._recover_bird(obs, clock, reason)
        elif blocked_reason:
            self._fingerprint = fingerprint
            self._pending_bird = None
            self._missed = True
            quality = TrackQuality(0, 'lost', f'blocked: {blocked_reason}')
            return Observation(None, None, w, h, bird_quality=quality, pipe_quality=quality)
        else:
            self._fingerprint = fingerprint
            obs, candidates = _analyze(frame, self.previous_bird, config=self.config, search_all=True)
            if self._bird_seen is not None and now-self._bird_seen > limits.max_frame_age:
                self._missed = True
            usable = [c for c in candidates if c.quality.confidence >= limits.min_bird_confidence
                      and self._plausible_bird(c.bird, now, w, h)]
            if self.previous_bird is not None:
                expected_y = self._predicted_y(now, h)
                usable.sort(key=lambda c: abs(c.bird.x-self.previous_bird.x)*3 + abs(c.bird.y-expected_y)
                            - c.quality.confidence*self.previous_bird.radius)
            candidate = usable[0] if usable else None
            age = now-self._bird_seen if self._bird_seen is not None else 0
            if candidate and self._missed and age >= limits.reacquire_seconds:
                pending = self._pending_bird
                if (pending is None or now-pending['seen'] > limits.max_frame_age
                        or abs(candidate.bird.x-pending['bird'].x) > w*limits.bird_x_tolerance
                        or abs(candidate.bird.y-pending['bird'].y) > h*limits.max_bird_speed*(now-pending['seen'])+5):
                    pending = {'start': now, 'seen': now, 'origin_y': candidate.bird.y, 'bird': candidate.bird}
                    self._pending_bird = pending
                pending.update(seen=now, bird=candidate.bird)
                if (now-pending['start'] < limits.reacquire_seconds
                        or abs(candidate.bird.y-pending['origin_y']) < max(3, h*.004)):
                    candidate = None
                    reason = 'confirming moving bird candidate'
            elif candidate is None:
                self._pending_bird = None
            if candidate is not None and not self._missed:
                history = self._bird_history
                while history and now-history[0][0] > limits.motion_window_seconds:
                    history.popleft()
                if not history or now-history[-1][0] >= limits.motion_window_seconds/30:
                    history.append((now, candidate.bird.y))
                if (history and now-history[0][0] >= limits.stationary_bird_seconds
                        and max(y for _, y in history)-min(y for _, y in history) < max(3, h*.004)):
                    candidate = None
                    reason = 'stationary bird candidate'
            if candidate is not None:
                quality = candidate.quality
                if self.previous_bird is not None:
                    dt = now-self._bird_seen
                    if 0 < dt <= .5:
                        velocity = (candidate.bird.y-self.previous_bird.y)/dt + .8*h*dt
                        self._velocity += (1-math.exp(-dt/.05)) * (velocity-self._velocity)
                        self._velocity = float(np.clip(self._velocity, -h*limits.max_bird_speed, h*limits.max_bird_speed))
                    else:
                        self._velocity = 0.
                    quality = replace(quality, reason=quality.reason + ('; reacquired with motion' if self._missed else '; motion consistent'))
                self.previous_bird = candidate.bird
                self._bird_seen, self._bird_confidence = now, quality.confidence
                if self._missed:
                    self._bird_history.clear()
                    self._bird_history.append((now, candidate.bird.y))
                self._pending_bird = None
                self._missed = False
                obs = replace(obs, bird=candidate.bird, bird_quality=quality)
            else:
                reason = reason or ('bird position/size jump rejected' if candidates else 'bird mask missing')
                if candidates and all(c.quality.confidence < limits.min_bird_confidence for c in candidates):
                    reason = 'low bird color/shape confidence'
                obs = self._recover_bird(obs, clock, reason)
        return self.stabilize(obs, now, speed or w*.35, clock=clock)

    def _predicted_y(self, now, height):
        dt = max(0., now-self._bird_seen)
        return self.previous_bird.y + self._velocity*dt + .8*height*dt*dt

    def _plausible_bird(self, bird, now, width, height):
        if self.previous_bird is None:
            return True
        limits = self.config.tracking
        age = max(0., now-self._bird_seen)
        expansion = 1 + min(1., age/limits.expand_search_seconds)
        dx = max(self.previous_bird.radius*1.5, width*limits.bird_x_tolerance)*expansion
        dy = max(self.previous_bird.radius*2, height*.025) + height*limits.max_bird_speed*min(age, .3)
        # After a longer gap the trajectory is uncertain; the confirmation
        # period still requires genuinely moving, fresh candidates.
        expected_y = self._predicted_y(min(now, self._bird_seen+limits.expand_search_seconds), height)
        return (abs(bird.x-self.previous_bird.x) <= dx and abs(bird.y-expected_y) <= dy
                and abs(bird.y-self.previous_bird.y) <= height*limits.max_bird_speed*age+max(4, self.previous_bird.radius*.5)
                and .5 <= bird.radius/self.previous_bird.radius <= 1.8)

    def _recover_bird(self, obs, clock, reason):
        self._missed = True
        limits = self.config.tracking
        age = max(0., clock-self._bird_seen) if self._bird_seen is not None else limits.lost_seconds
        bird, confidence, predicted = None, 0., False
        state = 'lost' if age >= limits.lost_seconds else 'recovering'
        if self.previous_bird is not None and age <= limits.prediction_seconds:
            state, predicted = 'uncertain', True
            y = self._predicted_y(clock, obs.height)
            if 0 <= y < obs.height*.88:
                bird = replace(self.previous_bird, y=y)
            confidence = min(limits.min_bird_confidence*.99,
                             self._bird_confidence*max(0., 1-age/limits.prediction_seconds))
        return replace(obs, bird=bird, bird_quality=TrackQuality(confidence, state, reason, predicted, age))

    def stabilize(self, obs: Observation, now: float, speed: float, *, clock: float | None = None) -> Observation:
        clock = now if clock is None else clock
        limits = self.config.tracking
        remaining = list(self._tracks)
        edge_margin = image_edge_margin(obs.width)
        tracks, pipes, warnings = [], [], []
        bx = obs.bird.x if obs.bird else (self.previous_bird.x if self.previous_bird else obs.width*.3)
        trailing_edge = bx-(obs.bird.radius if obs.bird else 8)
        measured = (obs.pipes or ((obs.pipe,) if obs.pipe else ())) if obs.fresh and obs.pipe_quality.trusted else ()
        if obs.fresh and not obs.pipe_quality.trusted:
            warnings.append(obs.pipe_quality)
        for pipe in measured:
            matches = [(abs(track['right'] - speed * (now - track['seen']) - pipe.right), index)
                       for index, track in enumerate(remaining)]
            distance, index = min(matches, default=(float('inf'), -1))
            if distance < max(8, obs.width * limits.max_pipe_shift):
                track = remaining.pop(index)
                expected_left = track['left']-speed*(now-track['seen'])
                if (pipe.left > 1 and track['left'] > 1
                        and abs(pipe.left-expected_left) > max(8, obs.width*limits.max_pipe_shift)):
                    remaining.append(track)
                    if pipe.right >= trailing_edge:
                        warnings.append(TrackQuality(0, 'uncertain', 'pipe position/width jump rejected'))
                    continue
            else:
                if self._tracks and pipe.right < obs.width-edge_margin:
                    if pipe.right >= trailing_edge:
                        warnings.append(TrackQuality(0, 'uncertain', 'unmatched pipe position rejected'))
                    continue
                track = {'tops': deque(maxlen=120), 'bottoms': deque(maxlen=120),
                         'motion': deque(maxlen=32), 'motion_speed': None, 'valid_seen': now}
            gap_ok = (not track['tops'] or
                      max(abs(pipe.gap_top-float(np.median(track['tops']))),
                          abs(pipe.gap_bottom-float(np.median(track['bottoms'])))) <= obs.height*limits.max_gap_change)
            history = track['motion']
            if pipe.left > edge_margin and pipe.right < obs.width-edge_margin:
                while history and now-history[0][0] > limits.motion_window_seconds:
                    history.popleft()
                if not history or now-history[-1][0] >= limits.motion_window_seconds/30:
                    history.append((now, pipe.right))
            else:
                history.clear()  # An entering pipe's clipped right edge is stationary.
            motion_ok, measured_speed = True, None
            if history and now-history[0][0] >= limits.pipe_motion_seconds:
                span = now-history[0][0]
                displacement = history[0][1]-pipe.right
                measured_speed = displacement/span
                reference = track['motion_speed']
                if reference is None:
                    # The controller's initial speed is only a scale-based
                    # guess. Learn a visual reference before comparing motion.
                    if displacement >= max(4, obs.width*.01):
                        track['motion_speed'] = measured_speed
                    elif span >= limits.motion_window_seconds*.9:
                        motion_ok = False
                else:
                    motion_ok = (measured_speed > max(obs.width*.01, reference*.2)
                                 and abs(displacement-reference*span) <= max(8, reference*span*.6))
            consistent = gap_ok and motion_ok
            if consistent:
                track['tops'].append(pipe.gap_top)
                track['bottoms'].append(pipe.gap_bottom)
                track['valid_seen'] = now
                if measured_speed is not None and track['motion_speed'] is not None:
                    track['motion_speed'] += .1*(measured_speed-track['motion_speed'])
            track.update(left=pipe.left, right=pipe.right, seen=now)
            stable = Pipe(pipe.left, pipe.right, round(float(np.median(track['tops']))),
                          round(float(np.median(track['bottoms']))))
            if consistent:
                pipes.append(stable)
            else:
                age = max(0., clock-track['valid_seen'])
                if stable.right >= trailing_edge:
                    warnings.append(self._pipe_warning(age, 'pipe gap jump rejected' if not gap_ok else 'pipe motion inconsistent'))
                if age <= limits.pipe_memory_seconds:
                    pipes.append(stable)
            tracks.append(track)
        for track in remaining:
            dt, age = max(0., clock-track['seen']), max(0., clock-track['valid_seen'])
            right, left = round(track['right']-speed*dt), round(track['left']-speed*dt)
            if right < -obs.width*.05:
                continue
            tracks.append(track)
            if right >= trailing_edge:
                warnings.append(self._pipe_warning(age, 'pipe missing; geometry held briefly'))
            if age <= limits.pipe_memory_seconds and right > 0:
                pipes.append(Pipe(left, right, round(float(np.median(track['tops']))),
                                  round(float(np.median(track['bottoms'])))))
        self._tracks = tracks[:16]
        pipes.sort(key=lambda p: p.left)
        quality = obs.pipe_quality
        if warnings:
            states = ('tracking', 'uncertain', 'recovering', 'lost')
            quality = max(warnings, key=lambda q: (states.index(q.state), q.age))
        return replace(obs, pipe=_next_pipe(pipes, obs.bird, obs.width), pipes=tuple(pipes), pipe_quality=quality)

    def _pipe_warning(self, age, reason):
        limits = self.config.tracking
        state = ('uncertain' if age <= limits.pipe_memory_seconds else
                 'recovering' if age < limits.lost_seconds else 'lost')
        return TrackQuality(0, state, reason, True, age)


def annotate(frame_bgr: np.ndarray, obs: Observation, status: str) -> np.ndarray:
    annotated = frame_bgr.copy()
    if obs.bird:
        b = obs.bird
        cv2.circle(annotated, (round(b.x), round(b.y)), round(b.radius + 4),
                   (0, 180, 255) if obs.bird_quality.predicted else (0, 0, 255), 2)
    if obs.pipe:
        p = obs.pipe
        cv2.rectangle(annotated, (p.left, p.gap_top), (p.right, p.gap_bottom), (255, 0, 255), 2)
        cv2.line(annotated, (0, round(p.gap_center)), (obs.width, round(p.gap_center)), (255, 0, 255), 1)
    cv2.putText(annotated, status[:65], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
    cv2.putText(annotated, status[:65], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    evidence = f'vision={obs.state} B={obs.bird_quality.confidence:.2f} P={obs.pipe_quality.confidence:.2f} fresh={obs.fresh}'
    reason = obs.bird_quality.reason if not obs.bird_quality.trusted else obs.pipe_quality.reason
    for y, line in ((44, evidence), (62, reason[:70])):
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3)
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return annotated
