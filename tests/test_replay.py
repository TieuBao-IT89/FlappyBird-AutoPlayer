from dataclasses import replace
import json
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from flappy_auto.__main__ import make_parser, run
from flappy_auto.profiles import VisionConfig
from flappy_auto.provenance import code_info
from flappy_auto.recording import Recorder
from flappy_auto.replay import ReplayViewer, replay_session
from flappy_auto.session_format import ReplaySession
from flappy_auto.strategy import Strategy
from flappy_auto.vision import VisionTracker


def scene(index):
    frame = np.full((300, 180, 3), (240, 200, 125), np.uint8)
    cv2.rectangle(frame, (130-index, 0), (154-index, 85), (55, 180, 55), -1)
    cv2.rectangle(frame, (130-index, 175), (154-index, 263), (55, 180, 55), -1)
    cv2.circle(frame, (54, 130+index), 6, (40, 210, 245), -1)
    return frame


def recorded(tmp_path, *, full=True):
    recorder = Recorder(tmp_path/'input', full=full, metadata={'code': code_info()})
    strategy, tracker = Strategy(), VisionTracker()
    recorder.action('start', 9.9)
    for i in range(5):
        frame = scene(i)
        segment = 1 if i < 3 else 2
        if i == 3:
            tracker = VisionTracker()
        # Include a repeated sensor clock and a blocked terminal observation.
        captured = 10+(i if i != 2 else 1)/30
        processed = 10+i/30+.01
        reason = 'game over' if i == 4 else None
        obs = tracker.detect(frame, captured, 60, processed_at=processed, blocked_reason=reason)
        recorder.record(1, i/30, processed-captured, 2, frame, obs, i == 1, strategy,
                        captured_at=captured, processed_at=processed, vision_speed=60,
                        action_sent_at=processed+.001 if i == 1 else None, segment=segment)
        recorder.worker.flush()  # Deterministic fixture; production never waits per frame.
    recorder.finish_run(1)
    recorder.close({'runs': [{'score_so_far': 2, 'status': 'completed', 'final_score': 2}]})
    return recorder


def edit_events(directory, edit):
    path = directory/'events.jsonl'
    events = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    edit(events)
    path.write_text(''.join(json.dumps(e)+'\n' for e in events), encoding='utf-8')


def test_lossless_replay_is_repeatable_and_matches_original_tracking(tmp_path):
    recorder = recorded(tmp_path)
    session = ReplaySession.open(recorder.directory)
    events = [e for e in session.events() if e['type'] == 'frame']
    assert session.metadata['complete_images'] and session.metadata['closed']
    assert events[1]['captured_at'] == events[2]['captured_at']  # Staleness preserved.
    assert events[-1]['blocked_reason'] == 'game over'
    for i, event in enumerate(events):
        assert np.array_equal(cv2.imread(str(session.image_path(event['image']['path']))), scene(i))
    first = replay_session(recorder.directory, tmp_path/'out', export_images=True)
    second = replay_session(recorder.directory, tmp_path/'out')
    assert first['frames_processed'] == first['recorded_observations_equal'] == 5
    assert first['all_recorded_observations_match']
    assert first['observation_sha256'] == second['observation_sha256']
    assert first['output_directory'] != second['output_directory']
    assert not first['controller_scores_evaluated']
    assert len(list((Path(first['output_directory'])/'overlays').glob('*.png'))) == 5


@pytest.mark.parametrize('options', [{'detector': 'raw'}, {'config': replace(VisionConfig(), use_eye=False)}])
def test_changed_detector_does_not_claim_new_controller_scores(tmp_path, options):
    recorder = recorded(tmp_path)
    result = replay_session(recorder.directory, tmp_path/'out', **options)
    assert result['frames_processed'] == 5
    assert not result['live_comparison_evaluable']
    assert not result['controller_scores_evaluated']


@pytest.mark.parametrize('key,value', [('processed_at', None), ('processed_at', -1),
                                      ('captured_at', float('nan')), ('processed_at', 'bad')])
def test_invalid_timestamps_require_explicit_repair(tmp_path, key, value):
    recorder = recorded(tmp_path)
    edit_events(recorder.directory, lambda rows: rows[2].__setitem__(key, value))
    with pytest.raises(ValueError, match='repair-timestamps'):
        replay_session(recorder.directory, tmp_path/'strict')
    result = replay_session(recorder.directory, tmp_path/'repair', repair_timestamps=True)
    assert result['timestamp_repairs'] >= 1 and result['frames_processed'] == 5
    assert not result['live_comparison_evaluable']
    report = next((tmp_path/'strict').glob('*/report.json'))
    assert json.loads(report.read_text(encoding='utf-8'))['status'] == 'error'


def test_missing_images_fail_or_are_explicitly_skipped(tmp_path):
    recorder = recorded(tmp_path)
    (recorder.directory/'frames/000000002.png').unlink()
    with pytest.raises(ValueError, match='Cannot read replay frame 2'):
        replay_session(recorder.directory, tmp_path/'strict')
    result = replay_session(recorder.directory, tmp_path/'skip', skip_missing=True)
    assert result['missing_images'] == 1 and result['frames_processed'] == 4
    assert not result['live_comparison_evaluable']


def test_truncated_log_cannot_claim_exact_match_for_remaining_prefix(tmp_path):
    recorder = recorded(tmp_path)
    edit_events(recorder.directory, lambda rows: rows.pop())
    result = replay_session(recorder.directory, tmp_path/'out')
    assert result['recorded_observations_equal'] == result['frames_processed'] == 4
    assert not result['frame_count_matches_metadata']
    assert not result['all_recorded_observations_match']


def test_reject_version_path_escape_and_invalid_frame_size(tmp_path):
    recorder = recorded(tmp_path)
    session = ReplaySession.open(recorder.directory)
    with pytest.raises(ValueError, match='escapes'):
        session.image_path('../private.png')
    metadata = session.metadata
    metadata['version'] = 2
    (recorder.directory/'session.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='unsupported'):
        ReplaySession.open(recorder.directory)
    metadata.update(version=1, frame_size=[0, 300])
    (recorder.directory/'session.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='frame_size'):
        ReplaySession.open(recorder.directory)


@pytest.mark.parametrize('field,value', [('code', None), ('closed', 'true'), ('recording_stats', []),
                                        ('recording_stats', {'frames_seen': -1})])
def test_reject_malformed_session_metadata(tmp_path, field, value):
    recorder = recorded(tmp_path)
    path = recorder.directory/'session.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match=field):
        ReplaySession.open(recorder.directory)


def test_backpressure_drops_images_without_blocking_control_and_bounds_metadata(tmp_path):
    recorder = Recorder(tmp_path, full=True)
    entered, release = Event(), Event()
    original = recorder._write_image

    def slow_write(event, frame):
        entered.set()
        assert release.wait(5)
        original(event, frame)

    recorder._write_image = slow_write
    frame, policy = scene(0), Strategy()
    obs = VisionTracker().detect(frame, 10, 60, processed_at=10.01)
    recorder.record(1, .1, .01, 0, frame, obs, False, policy)
    assert entered.wait(2)
    try:
        for index in range(2, 6):
            recorder.record(1, .1+index/30, .01, 0, frame, obs, False, policy)
        # The active write is blocked, yet all producers have already returned.
        assert recorder.stats['images_dropped'] == 3
        for index in range(600):
            recorder.action('flap', 11+index/30)
        assert len(recorder.worker.events) <= 512
        assert recorder.stats['events_dropped'] > 0 and recorder.stats['actions_dropped'] > 0
    finally:
        release.set()
        recorder.close({})
    assert not recorder.metadata['complete_images']
    assert recorder.stats['images_saved'] == 1  # Pending image's event was also evicted.


def test_image_drop_status_is_preserved_in_log(tmp_path):
    recorder = Recorder(tmp_path, full=True)
    entered, release = Event(), Event()
    original = recorder._write_image

    def slow_write(event, frame):
        entered.set()
        assert release.wait(5)
        original(event, frame)

    recorder._write_image = slow_write
    frame, policy = scene(0), Strategy()
    obs = VisionTracker().detect(frame, 10, 60, processed_at=10.01)
    recorder.record(1, .1, .01, 0, frame, obs, False, policy)
    assert entered.wait(2)
    try:
        for index in range(2, 6):
            recorder.record(1, .1+index/30, .01, 0, frame, obs, False, policy)
    finally:
        release.set()
        recorder.close({})
    events = list(ReplaySession.open(recorder.directory).events())
    assert [e['image']['status'] for e in events] == ['saved']+['dropped_backpressure']*3+['saved']
    assert recorder.stats['images_saved'] == 2
    result = replay_session(recorder.directory, tmp_path/'out')
    assert result['images_skipped'] == 3 and not result['live_comparison_evaluable']


def test_background_io_failure_is_reported_and_session_not_marked_complete(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path, full=True)
    monkeypatch.setattr(cv2, 'imwrite', lambda *args: False)
    frame = scene(0)
    obs = VisionTracker().detect(frame, 10, 60, processed_at=10.01)
    recorder.record(1, .1, .01, 0, frame, obs, False, Strategy())
    with pytest.raises(RuntimeError, match='recording failed'):
        recorder.close({})
    assert not json.loads((recorder.directory/'session.json').read_text())['closed']
    assert recorder.file.closed and recorder.event_file.closed


def test_rolling_buffer_replay_marks_sparse_jpeg_comparison_ineligible(tmp_path):
    recorder = recorded(tmp_path, full=False)
    result = replay_session(recorder.directory, tmp_path/'out')
    assert 0 < result['frames_processed'] < 5
    assert result['images_skipped'] > 0 and not result['live_comparison_evaluable']


def test_viewer_pause_step_resume_and_quit(monkeypatch):
    keys = iter([ord('n'), ord(' '), ord('q')])
    shown, closed = [], []
    monkeypatch.setattr(cv2, 'imshow', lambda *args: shown.append(args))
    monkeypatch.setattr(cv2, 'waitKeyEx', lambda delay: next(keys))
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(cv2, 'destroyWindow', lambda name: closed.append(name))
    viewer = ReplayViewer(paused=True)
    assert viewer.show(scene(0), {'processed_at': 10}) and viewer.paused
    assert viewer.show(scene(1), {'processed_at': 10}) and not viewer.paused
    assert not viewer.show(scene(2), {'processed_at': 10})
    viewer.close()
    assert len(shown) == 3 and len(closed) == 1
    ReplayViewer().close()  # Failed before first image: no nonexistent window to close.
    assert len(closed) == 1


def test_closing_viewer_title_bar_stops_paused_replay(monkeypatch):
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(cv2, 'waitKeyEx', lambda delay: -1)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: -1)
    assert not ReplayViewer(paused=True).show(scene(0), {'processed_at': 10})


@pytest.mark.parametrize('extra', [['--save-debug', 'other'], ['--calibrate'], ['--capture-only', 'test.png']])
def test_record_full_rejects_incompatible_modes_before_launch(extra):
    args = make_parser().parse_args(['--record-full', 'artifacts/test']+extra)
    with pytest.raises(ValueError, match='record-full'):
        run(args)
