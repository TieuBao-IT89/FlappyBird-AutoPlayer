from dataclasses import asdict, replace
import csv
import json

import cv2
import numpy as np
import pytest

from flappy_auto.profiles import GameProfile, TrackingConfig, VisionConfig
from flappy_auto.recording import Recorder
from flappy_auto.session import RunTracker, game_over_banner, name_dialog_close
from flappy_auto.strategy import Strategy
from flappy_auto.vision import Bird, Observation, Pipe, TrackQuality, VisionTracker, detect


def scene(t=0., *, visible=True, bird_y=None, bird_x=105, badge=False, gap=(140, 310), pipe_shift=0):
    image = np.full((600, 360, 3), (240, 200, 125), np.uint8)
    x = round(190-100*t)+pipe_shift
    cv2.rectangle(image, (x, 0), (x+52, gap[0]), (55, 180, 55), -1)
    cv2.rectangle(image, (x, gap[1]), (x+52, 527), (55, 180, 55), -1)
    if visible:
        y = round(270+80*t+370*t*t) if bird_y is None else bird_y
        cv2.circle(image, (bird_x, y), 12, (40, 210, 245), -1)
    if badge:
        cv2.circle(image, (105, 90), 12, (40, 210, 245), -1)
        cv2.circle(image, (110, 86), 3, (255, 255, 255), -1)
        cv2.rectangle(image, (114, 92), (123, 97), (20, 30, 220), -1)
    # Make independent fresh frames even if subpixel pipe motion rounds to
    # the same pixel; duplicate-image tests deliberately reuse the whole frame.
    image[580, 350] = (round(t*10000) % 256, round(t*100) % 256, 0)
    return image


def observe(tracker, t, **kwargs):
    return tracker.detect(scene(t, **kwargs), 10+t, 100)


def test_short_occlusion_uses_bounded_prediction_then_reacquires_moving_bird():
    tracker = VisionTracker()
    assert observe(tracker, 0).can_control
    brief = observe(tracker, .03, visible=False)
    assert brief.state == 'uncertain' and brief.bird_quality.predicted
    assert brief.bird is not None and not brief.can_control and not brief.can_learn
    searching = observe(tracker, .15, visible=False)
    assert searching.bird is None and searching.state == 'recovering'
    assert not observe(tracker, .18).can_control  # Confirm over real time.
    recovered = observe(tracker, .24)
    assert recovered.can_control and recovered.state == 'tracking'
    assert 'reacquired' in recovered.bird_quality.reason
    assert abs(recovered.bird.y - (270+80*.24+370*.24**2)) < 3


def test_static_badge_with_eye_and_beak_cannot_replace_lost_bird():
    tracker = VisionTracker()
    observe(tracker, 0)
    for t in np.arange(.04, .70, .02):
        obs = observe(tracker, float(t), visible=False, badge=True)
        assert not obs.bird_quality.trusted
        assert not obs.can_control
        assert obs.bird is None or obs.bird.y > 200
    # Return the true bird near its plausible path, with sustained movement.
    observe(tracker, .71, bird_y=350, badge=True)
    recovered = observe(tracker, .77, bird_y=356, badge=True)
    assert recovered.can_control and recovered.bird.y > 340


def test_implausible_teleport_is_not_learned_and_does_not_reset_anchor():
    tracker = VisionTracker()
    observe(tracker, 0)
    rejected = observe(tracker, .02, bird_y=100)
    assert not rejected.can_control and 'jump rejected' in rejected.bird_quality.reason
    assert tracker.previous_bird.y > 260
    assert observe(tracker, .04).can_control


@pytest.mark.parametrize('fps', [15, 30, 60, 120])
def test_recovery_deadline_depends_on_elapsed_time_not_frame_count(fps):
    tracker = VisionTracker()
    observe(tracker, 0)
    for tick in range(1, round(fps*.7)+1):
        t = tick/fps
        obs = observe(tracker, t, visible=False)
        assert obs.bird_quality.state != 'lost'
        assert not obs.can_control
    final = observe(tracker, .85, visible=False)
    assert final.bird_quality.state == 'lost' and final.bird is None
    assert final.bird_quality.age == pytest.approx(.85)


@pytest.mark.parametrize('new_timestamps', [False, True])
def test_duplicate_images_expire_recovery_and_session_even_with_new_timestamps(new_timestamps):
    tracker, session = VisionTracker(), RunTracker()
    image = scene()
    first = tracker.detect(image, 10, 100, processed_at=10)
    session.update(first, 10, wall_time=10)
    for t in (.03, .15, .5, .85):
        stamp = 10+t if new_timestamps else 10
        obs = tracker.detect(image.copy(), stamp, 100, processed_at=10+t)
        assert not obs.fresh and not obs.can_control and not obs.can_learn
        reason = session.update(obs, stamp, wall_time=10+t)
    assert obs.bird_quality.state == 'lost' and obs.bird is None
    assert reason == 'video stalled / stale frames'


def test_new_pixels_with_old_timestamps_and_delayed_frames_are_rejected():
    tracker = VisionTracker()
    observe(tracker, 0)
    old = tracker.detect(scene(.02), 9.9, processed_at=10.02)
    assert not old.fresh and not old.can_control
    assert 'old frame' in old.bird_quality.reason
    late = tracker.detect(scene(.1), 10.1, processed_at=10.5)
    assert not late.fresh and 'too old' in late.bird_quality.reason
    assert tracker._bird_seen == 10
    assert not tracker.detect(scene(.6), 1000, processed_at=10.6).fresh
    assert tracker.detect(scene(.62), 10.62, processed_at=10.62).fresh


def test_frame_drop_requires_confirmation_and_identical_images_cannot_confirm():
    tracker = VisionTracker()
    observe(tracker, 0)
    candidate = scene(.3)
    pending = tracker.detect(candidate, 10.3, 100)
    assert not pending.can_control and 'confirming' in pending.bird_quality.reason
    for t in (.4, .6, .85):
        duplicate = tracker.detect(candidate.copy(), 10+t, 100)
        assert not duplicate.fresh and not duplicate.can_control
    assert duplicate.bird_quality.state == 'lost'


def test_bad_pipe_gap_is_held_briefly_then_expires_without_poisoning_median():
    tracker = VisionTracker()
    original = observe(tracker, 0)
    bad = observe(tracker, .04, gap=(210, 380))
    assert bad.pipe.gap_top == original.pipe.gap_top
    assert bad.pipe_quality.predicted and not bad.can_learn and not bad.can_control
    assert 'gap jump' in bad.pipe_quality.reason
    recovered = observe(tracker, .08)
    assert recovered.can_control and recovered.pipe.gap_top == original.pipe.gap_top
    observe(tracker, .12, gap=(210, 380))
    expired = observe(tracker, .45, gap=(210, 380))
    assert expired.pipe is None and expired.pipe_quality.state == 'recovering'
    lost = observe(tracker, .95, gap=(210, 380))
    assert lost.pipe_quality.state == 'lost' and not lost.can_control


def test_pipe_position_jump_does_not_remove_original_obstacle():
    tracker = VisionTracker()
    initial = observe(tracker, 0)
    bad = observe(tracker, .02, pipe_shift=55)
    assert not bad.can_control and not bad.can_learn
    assert bad.pipe.right < initial.pipe.right
    assert bad.pipe.gap_top == initial.pipe.gap_top
    recovered = observe(tracker, .04)
    assert recovered.pipe_quality.trusted and recovered.can_control


@pytest.mark.parametrize('reverse_speed', [0, 30])
def test_frozen_or_reversing_pipe_is_not_trusted_despite_fresh_pixels(reverse_speed):
    tracker = VisionTracker()
    for t in (0, .04, .08, .12, .16):
        observe(tracker, t)
    for t in np.arange(.20, .6, .04):
        obs = observe(tracker, float(t), pipe_shift=round((100+reverse_speed)*(t-.16)))
    assert not obs.can_control and not obs.can_learn
    assert obs.pipe_quality.reason == 'pipe motion inconsistent'


def test_slow_but_consistent_pipes_are_accepted_while_controller_speed_warms_up():
    tracker = VisionTracker()
    for t in np.arange(0., .6, .02):
        frame = scene(float(t), pipe_shift=round(70*t))  # Actual speed = 30 px/s.
        obs = tracker.detect(frame, 10+float(t), speed=108)  # Initial scale-based guess.
        assert obs.pipe_quality.trusted, obs.pipe_quality
    assert tracker._tracks[0]['motion_speed'] == pytest.approx(30, abs=10)


def test_entering_pipe_clipped_inside_canvas_border_is_not_mistaken_for_stopped_motion():
    tracker = VisionTracker()
    bird = Bird(105, 270, 12)
    for t in (0, .04, .08, .12, .16, .20):
        pipe = Pipe(round(340-100*t), 354, 140, 310)  # 4 px border in a 358 px frame.
        obs = tracker.stabilize(Observation(bird, pipe, 358, 588, (pipe,)), t, 100)
        assert obs.can_control and obs.pipe_quality.trusted
    existing, following = Pipe(316, 348, 140, 310), Pipe(342, 354, 210, 380)
    obs = tracker.stabilize(Observation(bird, existing, 358, 588, (existing, following)), .24, 100)
    assert obs.pipe_quality.trusted and len(obs.pipes) == 2


def test_stationary_bird_candidate_cannot_remain_trusted_while_pipes_move():
    tracker = VisionTracker()
    observe(tracker, 0, bird_y=270)
    for t in np.arange(.03, .4, .03):
        obs = observe(tracker, float(t), bird_y=270)
    assert not obs.can_control and not obs.bird_quality.trusted
    assert tracker._bird_seen < 10.25


def test_bad_gap_of_a_pipe_already_behind_the_bird_does_not_block_control():
    tracker = VisionTracker()
    bird = Bird(180, 300, 14)
    passed = Pipe(70, 130, 140, 310)
    first = tracker.stabilize(Observation(bird, None, 480, 720, (passed,)), 1, 100)
    assert first.can_control
    corrupted = Pipe(68, 128, 210, 380)
    next_frame = tracker.stabilize(Observation(bird, None, 480, 720, (corrupted,)), 1.02, 100)
    assert next_frame.pipe is None and next_frame.can_control


def test_mask_speckle_noise_does_not_beat_real_bird_and_scores_describe_evidence():
    clean = scene()
    noisy = clean.copy()
    rng = np.random.default_rng(4)
    for x, y in rng.integers((20, 40), (180, 500), size=(60, 2)):
        noisy[y:y+2, x:x+2] = (40, 210, 245)
    obs = detect(noisy)
    assert obs.bird is not None and abs(obs.bird.y-270) < 3
    assert 0 <= obs.bird_quality.confidence <= 1 and 0 <= obs.pipe_quality.confidence <= 1
    assert all(key in obs.bird_quality.reason for key in ('color=', 'shape=', 'size=', 'position='))
    glyph = np.zeros_like(clean)
    cv2.rectangle(glyph, (100, 260), (107, 280), (40, 210, 245), -1)
    assert detect(glyph).bird_quality.confidence < detect(clean).bird_quality.confidence
    assert not VisionTracker().detect(glyph, 10).can_control


def test_both_eye_and_beak_contribute_to_the_confidence_score():
    frame = scene()
    cv2.circle(frame, (110, 265), 4, (255, 255, 255), -1)
    cv2.rectangle(frame, (114, 271), (124, 277), (20, 30, 220), -1)
    assert 'cues=1.00' in detect(frame).bird_quality.reason


def test_small_badge_jitter_does_not_count_as_confirmed_bird_motion():
    tracker = VisionTracker()
    observe(tracker, 0)
    for i, t in enumerate(np.arange(.04, .70, .02)):
        # Crop/codec jitter of at most one pixel is not genuine bird motion.
        frame = scene(float(t), visible=False)
        cv2.circle(frame, (105, 90+(i % 3)-1), 12, (40, 210, 245), -1)
        obs = tracker.detect(frame, 10+float(t), 100)
        assert not obs.bird_quality.trusted and not obs.can_control


def test_loss_before_the_first_pipe_ends_an_active_game_but_not_an_idle_start():
    tracker = RunTracker()
    tracker.update(Observation(Bird(105, 270, 12), None, 360, 600), 1)
    tracker.update(Observation(Bird(105, 230, 12), None, 360, 600), 1.1)
    assert tracker.active and not tracker.has_played
    missing = Observation(None, None, 360, 600, bird_quality=TrackQuality(0, 'lost', 'missing'))
    assert tracker.update(missing, 2) == 'bird lost'
    idle = RunTracker()
    ready = Observation(Bird(105, 270, 12), None, 360, 600)
    idle.update(ready, 1)
    assert idle.update(ready, 20) is None and not idle.active


@pytest.mark.parametrize('target', ['bird', 'pipe', 'stale'])
def test_uncertain_observation_never_learns_or_flaps_even_when_planner_demands_it(target, monkeypatch):
    policy = Strategy(gravity=740, jump_speed=260, pipe_speed=100)
    monkeypatch.setattr(policy, '_plan', lambda obs, now: True)
    policy.last_flap = 1.
    policy._samples.extend((1+i*.04, 200+100*i*.04+.5*740*(i*.04)**2) for i in range(8))
    policy._pipe_samples.extend([(1, 242), (1.1, 232), (1.2, 222)])
    good = Observation(Bird(105, 420, 12), Pipe(200, 252, 140, 310), 360, 600)
    weak = TrackQuality(.2, 'uncertain', 'test occlusion', predicted=True, age=.1)
    bad = replace(good, **({'bird_quality': weak} if target == 'bird' else
                           {'pipe_quality': weak} if target == 'pipe' else {'fresh': False}))
    previous = (policy.gravity, policy.jump_speed, policy.pipe_speed, policy.last_flap)
    policy._observe(bad, 2)
    policy._calibrate(bad, 2)
    assert not policy.should_flap(bad, 2)[0]
    assert (policy.gravity, policy.jump_speed, policy.pipe_speed, policy.last_flap) == previous
    assert not policy._samples and not policy._pipe_samples


@pytest.mark.parametrize('menu', ['game over', 'name dialog'])
def test_known_game_over_or_name_menu_blocks_high_quality_candidates(menu):
    image = scene()
    if menu == 'game over':
        cv2.putText(image, 'GAME OVER', (45, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 230), 5)
        assert game_over_banner(image)
    else:
        cv2.rectangle(image, (48, 200), (311, 390), (205, 231, 250), -1)
        cv2.line(image, (283, 208), (293, 218), (40, 60, 80), 3)
        cv2.line(image, (293, 208), (283, 218), (40, 60, 80), 3)
        assert name_dialog_close(image) is not None
    tracker = VisionTracker()
    observe(tracker, 0)
    blocked = tracker.detect(image, 10.02, blocked_reason=menu)
    assert blocked.fresh and blocked.state == 'lost'
    assert blocked.bird is None and not blocked.can_control and not blocked.can_learn
    assert blocked.bird_quality.reason == f'blocked: {menu}'


def test_predicted_pipe_crossing_and_bird_motion_do_not_increment_score_or_extend_run():
    tracker = RunTracker()
    quality = TrackQuality(.2, 'uncertain', 'prediction', True, .1)
    original = Observation(Bird(180, 300, 14), Pipe(230, 290, 180, 340), 480, 720)
    tracker.update(original, 1)
    predicted = replace(original, bird=Bird(180, 350, 14), pipe=Pipe(140, 200, 180, 340),
                        bird_quality=quality, pipe_quality=quality)
    assert tracker.update(predicted, 1.4) is None
    assert tracker.score == 0 and tracker._last_bird_seen == 1
    assert tracker.update(predicted, 1.85) == 'bird lost'


def test_telemetry_records_confidence_state_reason_and_predicted_flags(tmp_path):
    recorder = Recorder(tmp_path)
    quality = TrackQuality(.2, 'recovering', 'bird missing', True, .15)
    obs = Observation(Bird(105, 300, 12), None, 360, 600, bird_quality=quality, fresh=False)
    recorder.record(1, 1, .03, 0, scene(), obs, False, Strategy())
    recorder.close({'runs': []})
    with (recorder.directory/'telemetry.csv').open(encoding='utf-8', newline='') as file:
        row = next(csv.DictReader(file))
    assert row['vision_state'] == 'recovering' and row['bird_confidence'] == '0.2'
    assert row['bird_reason'] == 'bird missing' and row['bird_predicted'] == 'True'
    assert row['fresh'] == row['can_learn'] == row['can_control'] == 'False'


def test_v1_profiles_without_tracking_fields_remain_compatible_and_thresholds_roundtrip(tmp_path):
    data = asdict(GameProfile())
    data['vision'].pop('tracking')
    path = tmp_path/'old-profile.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    assert GameProfile.load(path).vision.tracking == TrackingConfig()
    profile = GameProfile(vision=VisionConfig(tracking=TrackingConfig(min_bird_confidence=.95, lost_seconds=1.1)))
    profile.save(path)
    assert GameProfile.load(path) == profile
    obs = VisionTracker(profile.vision).detect(scene(), 1)
    assert not obs.can_control  # Threshold changed detector behavior, not just JSON.


@pytest.mark.parametrize('values', [{'lost_seconds': -1}, {'max_frame_age': float('nan')},
                                    {'min_bird_confidence': 2}, {'max_gap_change': True},
                                    {'prediction_seconds': .9}, {'stationary_bird_seconds': .5},
                                    {'pipe_motion_seconds': .4}, {'unknown': .1}])
def test_invalid_recovery_settings_are_rejected(values):
    with pytest.raises(ValueError):
        TrackingConfig.from_dict(values)
