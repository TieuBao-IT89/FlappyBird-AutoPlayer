"""Offline perception replay. Recorded pixels cannot measure a new policy's score."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .profiles import GameProfile, VisionConfig
from .provenance import code_info
from .session_format import ReplaySession, observation_dict, write_json
from .vision import VisionTracker, annotate, detect


def report_directory(root, prefix):
    path = Path(root)/datetime.now(timezone.utc).strftime(prefix+'_%Y%m%dT%H%M%S_%fZ')
    path.mkdir(parents=True, exist_ok=False)
    return path


class ReplayViewer:
    def __init__(self, *, paused=False, speed=1.):
        self.paused, self.speed = paused, speed
        self.name = 'Replay - Space: pause | N/Right: next | Q: quit'
        self.previous_clock = None
        self.created = False

    def show(self, image, event):
        clock = event['processed_at']
        dwell = 0 if self.previous_clock is None else min(1., max(0., clock-self.previous_clock)/self.speed)
        self.previous_clock = clock
        cv2.imshow(self.name, image)
        self.created = True
        deadline = time.perf_counter()+dwell
        while True:
            delay = 20 if self.paused else max(1, min(20, round((deadline-time.perf_counter())*1000)))
            key = cv2.waitKeyEx(delay)
            try:
                if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                    return False
            except cv2.error:
                return False
            if key in (ord('q'), ord('Q'), 27):
                return False
            if key in (ord('n'), ord('N'), 83, 65363, 2555904):
                self.paused = True
                return True
            if key == ord(' '):
                self.paused = not self.paused
            if not self.paused and time.perf_counter() >= deadline:
                return True

    def close(self):
        if self.created:
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass  # Closing the window through its title bar is also a stop.


def replay_session(directory, output, *, config=None, detector='tracked', headless=True,
                   repair_timestamps=False, fallback_fps=30, skip_missing=False,
                   paused=False, speed=1., export_images=False):
    if detector not in ('tracked', 'raw') or not np.isfinite(speed) or speed <= 0:
        raise ValueError('Choose tracked/raw and a finite positive playback speed')
    session = ReplaySession.open(directory)
    original_config = VisionConfig.from_dict(session.metadata['vision'])
    config = config or original_config
    output = report_directory(output, 'replay')
    if export_images:
        (output/'overlays').mkdir()
    current_code = code_info()
    same_config = asdict(config) == asdict(original_config)
    source_code = session.metadata.get('code', {})
    same_code = all(source_code.get(k) == current_code[k] for k in ('source_sha256', 'python', 'opencv', 'numpy'))
    summary = {'version': 1, 'kind': 'perception-replay', 'status': 'complete',
               'session': str(session.directory), 'output_directory': str(output.resolve()),
               'controller_scores_evaluated': False,
               'notice': 'Offline pixels evaluate perception and recorded-action context only; no new controller score or survival rate.',
               'detector': detector, 'vision': asdict(config), 'code': current_code,
               'same_config': same_config, 'same_code': same_code,
               'live_comparison_evaluable': bool(session.metadata.get('closed') and session.metadata.get('complete_images')
                                               and detector == 'tracked' and same_config and same_code),
               'logged_frames_visited': 0, 'frames_processed': 0, 'images_skipped': 0,
               'missing_images': 0, 'timestamp_repairs': 0, 'frame_id_gaps': 0,
               'recorded_observations_equal': 0, 'bird_detected_frames': 0, 'trusted_frames': 0,
               'recorded_flaps_on_new_untrusted_frames': 0}
    digest = hashlib.sha256()
    viewer = None if headless else ReplayViewer(paused=paused, speed=speed)
    tracker, segment, last_id = None, None, 0
    analysis_seconds = 0.
    try:
        with (output/'frames.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(('frame_id', 'run', 'segment', 'captured_at', 'processed_at', 'timestamp_repaired',
                             'bird_confidence', 'pipe_confidence', 'vision_state', 'fresh', 'can_control',
                             'recorded_action_sent', 'recorded_score_so_far', 'matches_recorded_observation', 'perception_ms'))
            for event in session.events(repair_timestamps=repair_timestamps, fallback_fps=fallback_fps):
                if event['type'] != 'frame':
                    continue
                summary['logged_frames_visited'] += 1
                summary['frame_id_gaps'] += event['id']-last_id-1
                last_id = event['id']
                summary['timestamp_repairs'] += len(event['timestamp_repairs'])
                if event['image']['status'] != 'saved':
                    summary['images_skipped'] += 1
                    summary['live_comparison_evaluable'] = False
                    continue
                image_path = session.image_path(event['image']['path'])
                try:
                    frame = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError('cannot decode image')
                except (OSError, ValueError) as exc:
                    if not skip_missing:
                        raise ValueError(f'Cannot read replay frame {event["id"]}: {exc}') from exc
                    summary['missing_images'] += 1
                    summary['live_comparison_evaluable'] = False
                    continue
                if [frame.shape[1], frame.shape[0]] != session.metadata['frame_size']:
                    raise ValueError(f'Replay frame {event["id"]} dimensions differ from session metadata')
                if segment != event['segment']:
                    tracker, segment = VisionTracker(config), event['segment']
                started = time.perf_counter()
                if detector == 'tracked':
                    obs = tracker.detect(frame, event['captured_at'], event['vision_speed'],
                                         processed_at=event['processed_at'], blocked_reason=event.get('blocked_reason'))
                else:
                    obs = detect(frame, config=config)
                duration = time.perf_counter()-started
                analysis_seconds += duration
                normalized = observation_dict(obs)
                equal = normalized == event['observation']
                digest.update(json.dumps({'id': event['id'], 'observation': normalized}, sort_keys=True,
                                         separators=(',', ':'), allow_nan=False).encode())
                summary['frames_processed'] += 1
                summary['recorded_observations_equal'] += int(equal)
                summary['bird_detected_frames'] += int(obs.bird is not None)
                summary['trusted_frames'] += int(obs.can_control)
                summary['recorded_flaps_on_new_untrusted_frames'] += int(event['action_sent'] and not obs.can_control)
                writer.writerow((event['id'], event['run'], segment, event['captured_at'], event['processed_at'],
                                 bool(event['timestamp_repairs']), obs.bird_quality.confidence, obs.pipe_quality.confidence,
                                 obs.state, obs.fresh, obs.can_control, event['action_sent'], event.get('score_so_far'),
                                 equal, round(duration*1000, 4)))
                if export_images or viewer is not None:
                    overlay = annotate(frame, obs, f'OFFLINE vision | frame={event["id"]} sent={event["action_sent"]}')
                    if export_images and not cv2.imwrite(str(output/'overlays'/f'{event["id"]:09d}.png'), overlay):
                        raise OSError('Cannot export replay overlay')
                    if viewer is not None and not viewer.show(overlay, event):
                        summary['status'] = 'stopped'
                        break
    except Exception as exc:
        summary['status'], summary['error'] = 'error', str(exc)
        raise
    finally:
        if viewer is not None:
            viewer.close()
        expected = session.metadata.get('recording_stats', {}).get('frames_seen')
        summary['recorded_frame_count'] = expected
        summary['frame_count_matches_metadata'] = type(expected) is int and expected == summary['logged_frames_visited']
        if (summary['timestamp_repairs'] or summary['frame_id_gaps'] or summary['status'] != 'complete'
                or not summary['frame_count_matches_metadata']):
            summary['live_comparison_evaluable'] = False
        summary['observation_sha256'] = digest.hexdigest()
        summary['perception_seconds'] = analysis_seconds
        summary['perception_processing_fps'] = summary['frames_processed']/analysis_seconds if analysis_seconds else None
        summary['all_recorded_observations_match'] = bool(summary['live_comparison_evaluable'] and summary['frames_processed']
                                                        and summary['frames_processed'] == summary['recorded_observations_equal'])
        write_json(output/'report.json', summary)
        (output/'report.md').write_text(
            '# Phát lại nhận diện\n\n'
            'Ảnh có sẵn chỉ đánh giá nhận diện và ngữ cảnh lệnh gốc. **Không đo điểm hoặc tỷ lệ sống của bộ điều khiển mới.**\n\n'
            f'- Trạng thái: {summary["status"]}\n- Frame xử lý: {summary["frames_processed"]}\n'
            f'- Frame bỏ qua/thiếu ảnh: {summary["images_skipped"]}/{summary["missing_images"]}\n'
            f'- Timestamp đã sửa: {summary["timestamp_repairs"]}\n'
            f'- Có đủ điều kiện đối chiếu lần chạy gốc: {summary["live_comparison_evaluable"]}\n'
            f'- Observation khớp log: {summary["recorded_observations_equal"]}/{summary["frames_processed"]}\n'
            f'- SHA256 chuỗi kết quả: `{summary["observation_sha256"]}`\n\n'
            'FPS trong JSON/CSV đo tốc độ tính nhận diện offline, không phải FPS game. '
            'Ảnh JPEG của bộ đệm nhẹ và phiên bị bỏ frame không bảo đảm khớp trạng thái theo dõi lúc chạy trực tiếp.\n',
            encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/replays'))
    parser.add_argument('--profile', type=Path, help='Override vision settings only; recorded images are already cropped')
    parser.add_argument('--detector', choices=('tracked', 'raw'), default='tracked')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--step', action='store_true', help='Open paused; N/Right advances one frame, Space resumes')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--repair-timestamps', action='store_true')
    parser.add_argument('--fallback-fps', type=float, default=30)
    parser.add_argument('--skip-missing', action='store_true')
    parser.add_argument('--export-images', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = GameProfile.load(args.profile).vision if args.profile else None
        result = replay_session(args.session, args.output, config=config, detector=args.detector, headless=args.headless,
                                repair_timestamps=args.repair_timestamps, fallback_fps=args.fallback_fps,
                                skip_missing=args.skip_missing, paused=args.step, speed=args.speed, export_images=args.export_images)
        print(f'[REPLAY] frames={result["frames_processed"]} status={result["status"]} report={result["output_directory"]}')
        return 0
    except (ValueError, OSError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
