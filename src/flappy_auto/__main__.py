"""Command line entry point: python -m flappy_auto [--demo]."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import cv2
from playwright.sync_api import sync_playwright

from .browser import GameBrowser, block_advertising
from .demo_server import demo_server
from .recording import Recorder
from .profiles import GameProfile, VisionConfig
from .session import BrowserMetrics, RunTracker, game_over_banner, name_dialog_close
from .strategy import Strategy
from .vision import VisionTracker, annotate
from .workers import DebugWindow


def make_controller(args, vision_config):
    """Validate/load before launching a browser; rule mode needs no RL extras."""
    if args.policy == 'rule':
        if args.model is not None:
            raise ValueError('--model requires --policy dqn')
        return Strategy()
    if args.model is None:
        raise ValueError('--policy dqn requires --model PATH (ZIP plus matching JSON)')
    try:
        from .rl.dqn import load_checkpoint
        from .rl.browser_policy import DQNController, browser_policy_hash
    except ImportError as exc:
        raise ValueError('DQN requires optional dependencies: pip install -e ".[dqn]"') from exc
    model, metadata = load_checkpoint(args.model)
    controller = DQNController(model, vision_config.tracking)
    controller.checkpoint_metadata = metadata
    controller.policy_source_sha256 = browser_policy_hash()
    return controller


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Play Flappy Bird visually, using OpenCV + Playwright.')
    parser.add_argument('--demo', action='store_true', default=None, help='Use the bundled offline demo instead of flappybird.io')
    parser.add_argument('--url', help='Public game URL (default: profile URL or https://flappybird.io/)')
    parser.add_argument('--selector', help='CSS selector for game region (default: profile selector or canvas)')
    crop = parser.add_mutually_exclusive_group()
    crop.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), help='Crop screenshot (element-relative, else viewport-relative)')
    crop.add_argument('--full-frame', action='store_true', help='Ignore saved ROI and use the entire game screenshot')
    parser.add_argument('--calibrate', action='store_true', help='Open interactive visual calibration, save profile, then exit')
    parser.add_argument('--profile', type=Path, help='Load/save a versioned game profile (calibration default: profiles/site.json or demo.json)')
    parser.add_argument('--seconds', type=float, default=120, help='Total session duration in seconds, 0 = unlimited (default: 120)')
    parser.add_argument('--runs', type=int, default=1, help='Number of games, restarting after each loss (default: 1)')
    parser.add_argument('--policy', choices=('rule', 'dqn'), default='rule', help='Browser controller (default: rule)')
    parser.add_argument('--model', type=Path, help='DQN checkpoint ZIP with matching JSON metadata; requires --policy dqn')
    parser.add_argument('--input-latency-ms', type=float, default=8, help='Estimated key delivery delay in milliseconds (default: 8)')
    parser.add_argument('--fps', type=float, default=60, help='Target control rate; actual rate may be lower (default: 60)')
    parser.add_argument('--viewport', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Browser viewport (default: profile viewport or 480 720)')
    parser.add_argument('--debug', action='store_true', help='Show computer vision overlay; press Q to stop')
    parser.add_argument('--debug-fps', type=float, default=30, help='Debug window refresh rate, independent of control rate (default: 30)')
    parser.add_argument('--save-debug', type=Path, help='Save overlays, telemetry, run results and images before each loss')
    parser.add_argument('--record-full', type=Path, help='Record a replay session with lossless images (bounded writer; dropped frames are logged)')
    parser.add_argument('--capture-only', type=Path, help='Capture one screenshot then exit; useful for adjusting detection')
    parser.add_argument('--dry-run', action='store_true', help='Show decisions but do not press Space')
    parser.add_argument('--no-start', action='store_true', help='Do not click to start; start the game manually')
    parser.add_argument('--headless', action='store_true', help='Hide browser (primarily for CI and local demo)')
    parser.add_argument('--browser-executable', type=Path, help='Optional path to installed Chrome/Chromium executable')
    parser.add_argument('--allow-ads', action='store_true', help='Allow third-party ads on flappybird.io (disabled for performance)')
    return parser


def resolve_profile(args: argparse.Namespace) -> GameProfile | None:
    """Explicit CLI options > saved profile > original defaults."""
    if args.calibrate and args.profile is None:
        args.profile = Path('profiles/demo.json' if args.demo else 'profiles/site.json')
    args.roi_explicit = args.roi is not None or args.full_frame
    profile = None
    if args.profile is not None and (args.profile.exists() or not args.calibrate):
        profile = GameProfile.load(args.profile)
    base = profile or GameProfile()
    source = 'demo' if args.demo else ('website' if args.url is not None else base.source)
    args.demo = source == 'demo'
    args.url = args.url if args.url is not None else base.url
    args.selector = args.selector if args.selector is not None else base.selector
    args.viewport = tuple(args.viewport) if args.viewport is not None else base.viewport
    args.roi = None if args.full_frame else (tuple(args.roi) if args.roi is not None else base.roi)
    if profile is not None:
        # Retain reference dimensions for validation; CLI overrides do not disable the guard.
        profile = replace(profile, source=source, url=args.url, selector=args.selector, viewport=args.viewport)
    return profile


def run(args: argparse.Namespace) -> int:
    profile = resolve_profile(args)
    vision_config = profile.vision if profile is not None else VisionConfig()
    if args.record_full and args.save_debug:
        raise ValueError('Choose either --record-full DIRECTORY or --save-debug DIRECTORY')
    if args.record_full and (args.calibrate or args.capture_only):
        raise ValueError('--record-full requires a bot session; do not combine it with --calibrate or --capture-only')
    if args.calibrate and (args.headless or args.capture_only or args.debug or args.dry_run):
        raise ValueError('--calibrate opens its own window; do not combine it with --headless, --capture-only, --debug or --dry-run')
    if (args.fps <= 0 or args.debug_fps <= 0 or args.seconds < 0 or min(args.viewport) <= 0
            or args.runs < 1 or args.input_latency_ms < 0):
        raise ValueError('--fps, --runs and viewport dimensions must be positive; durations must be nonnegative')
    strategy = make_controller(args, vision_config)
    # Small vision frames do not benefit from scheduling a worker per CPU core.
    cv2.setNumThreads(2)
    server_context = demo_server() if args.demo else nullcontext(args.url)
    with server_context as url:
        with sync_playwright() as playwright:
            launch_options = {'headless': args.headless, 'args': [
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
            ]}
            if args.browser_executable:
                launch_options['executable_path'] = str(args.browser_executable)
            browser = playwright.chromium.launch(**launch_options)
            game = None
            recorder = None
            preview = None
            results = {'url': url, 'runs': [], 'score_method': 'visually tracked pipe crossings',
                       'policy': args.policy, 'model': str(args.model) if args.model else None,
                       'checkpoint': getattr(strategy, 'checkpoint_metadata', None),
                       'policy_source_sha256': getattr(strategy, 'policy_source_sha256', None),
                       'headless': args.headless, 'debug': args.debug,
                       'target_fps': args.fps, 'viewport': list(args.viewport)}
            outcome = 'error'
            try:
                page = browser.new_page(viewport={'width': args.viewport[0], 'height': args.viewport[1]}, device_scale_factor=1)
                if not args.allow_ads and urlsplit(url).hostname in ('flappybird.io', 'www.flappybird.io'):
                    block_advertising(page.context)
                print(f'[INFO] Opening {url}')
                page.goto(url, wait_until='domcontentloaded', timeout=35000)
                page.wait_for_timeout(1400)
                game = GameBrowser(page, args.selector)
                region = game.locate()
                print(f'[INFO] Screenshot region: {region}')
                if game.game_area is None:
                    print('[WARN] No matching canvas found. Try --selector or --roi X Y W H.')
                roi = tuple(args.roi) if args.roi else None
                if args.calibrate:
                    from .calibration import CalibrationSession, CalibrationWindow
                    frame = game.capture()
                    size = (frame.shape[1], frame.shape[0])
                    if profile is not None and (profile.reference_size != size or profile.region != game.region_kind):
                        print('[INFO] Layout changed; updating reference dimensions. Saved color thresholds are retained.')
                        if not args.roi_explicit:
                            roi = None
                    editing = GameProfile(source='demo' if args.demo else 'website', url=args.url,
                                          selector=args.selector, viewport=args.viewport, roi=roi,
                                          reference_size=size, region=game.region_kind, vision=vision_config)
                    session = CalibrationSession(frame, editing, game.region_kind)

                    def capture_calibration():
                        game.geometry()
                        return game.capture()

                    def restart_calibration():
                        game.restart(game.capture())
                        page.wait_for_timeout(500)
                        game.flap()

                    CalibrationWindow(session, args.profile, capture_calibration, game.start,
                                      game.flap, restart_calibration).run()
                    return 0
                if profile is not None:
                    profile.validate_capture(game.geometry(), game.region_kind)
                    game.calibration_profile = profile
                frame = game.capture(roi)
                print(f'[INFO] Screenshot dimensions: {frame.shape[1]}x{frame.shape[0]}')
                if args.capture_only:
                    args.capture_only.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.capture_only), frame)
                    print(f'[INFO] Saved capture to {args.capture_only}')
                    return 0
                if args.save_debug:
                    args.save_debug.mkdir(parents=True, exist_ok=True)
                if args.debug:
                    preview = DebugWindow()
                    preview.show(frame)
                    preview.flush()
                    page.bring_to_front()
                tracker = RunTracker(vision_timeout=vision_config.tracking.lost_seconds)
                metrics = BrowserMetrics()
                run_number = segment = 1
                perception = VisionTracker(vision_config)
                metadata = None
                if args.record_full or args.save_debug:
                    from .provenance import code_info
                    snapshot = GameProfile(source='demo' if args.demo else 'website', url=args.url,
                                           selector=args.selector, viewport=args.viewport, roi=roi,
                                           reference_size=game.geometry(), region=game.region_kind, vision=vision_config)
                    metadata = {'vision': asdict(vision_config), 'profile': asdict(snapshot), 'code': code_info(),
                                'frame_size': [frame.shape[1], frame.shape[0]],
                                'mode': {'source': 'demo' if args.demo else 'website', 'url': url,
                                         'headless': args.headless, 'dry_run': args.dry_run,
                                         'policy': args.policy, 'model': str(args.model) if args.model else None,
                                         'target_fps': args.fps, 'input_latency_ms': args.input_latency_ms},
                                'browser_version': browser.version}
                    metadata['checkpoint'] = results['checkpoint']
                    metadata['policy_source_sha256'] = results['policy_source_sha256']
                start = run_start = time.perf_counter()
                recorder = Recorder(args.record_full or args.save_debug, full=bool(args.record_full), metadata=metadata)
                if not args.no_start and not args.dry_run:
                    sent_at = time.perf_counter()
                    game.start()
                    recorder.action('start', sent_at, run_number, segment)
                    page.wait_for_timeout(130)
                if recorder.directory is not None:
                    print(f'[INFO] Diagnostics: {recorder.directory.resolve()}', flush=True)
                model_seconds = .006
                outcome = 'time_limit'
                tick = 0
                flap_count = 0
                last_report = start
                last_report_tick = 0
                last_debug = last_saved = float('-inf')
                print('[INFO] Bot running. Stop with Ctrl+C' + (' or Q in debug window.' if args.debug else '.'))
                while not args.seconds or time.perf_counter() - start < args.seconds:
                    if preview is not None and preview.stop_requested.is_set():
                        outcome = 'stopped'
                        break
                    frame_start = time.perf_counter()
                    frame = game.capture(roi)
                    observed_at = game.frame_time
                    dialog = name_dialog_close(frame) is not None
                    banner = game_over_banner(frame)
                    vision_speed = strategy.pipe_speed
                    processed_at = time.perf_counter()
                    obs = perception.detect(frame, observed_at, vision_speed,
                                            processed_at=processed_at,
                                            blocked_reason='game over' if banner else 'name dialog' if dialog else None)
                    metrics.observe(obs, max(0, time.perf_counter() - observed_at))
                    if dialog and obs.fresh and not tracker.has_played and not args.dry_run:
                        if args.policy == 'dqn':
                            strategy.policy_decision = None
                        recorder.record(run_number, time.perf_counter()-start, processed_at-observed_at,
                                        tracker.score, frame, obs, False, strategy, captured_at=observed_at,
                                        processed_at=processed_at, vision_speed=vision_speed, segment=segment)
                        # Some scores open a name dialog after the retry has
                        # begun. Close it and recover this attempt without
                        # interacting with the name field or leaderboard.
                        sent_at = time.perf_counter()
                        game.restart(frame, roi)
                        recorder.action('restart', sent_at, run_number, segment)
                        page.wait_for_timeout(500)
                        sent_at = time.perf_counter()
                        game.flap()
                        recorder.action('flap', sent_at, run_number, segment)
                        page.wait_for_timeout(130)
                        strategy, tracker, perception = (strategy.new_run(), RunTracker(vision_timeout=vision_config.tracking.lost_seconds),
                                                         VisionTracker(vision_config))
                        segment += 1
                        continue
                    age = max(0, time.perf_counter() - observed_at)
                    # Include time spent planning and delivering the key, not
                    # just the age of the image before the planner runs.
                    strategy.latency_seconds = min(.25, age + model_seconds + args.input_latency_ms / 1000)
                    reason = ('game over' if (banner or dialog) and obs.fresh else
                              tracker.update(obs, observed_at, wall_time=time.perf_counter()))
                    if reason and not args.dry_run:
                        if args.policy == 'dqn':
                            strategy.policy_decision = None
                        elapsed = time.perf_counter() - start
                        recorder.record(run_number, elapsed, age, tracker.score, frame, obs, False, strategy,
                                        captured_at=observed_at, processed_at=processed_at, vision_speed=vision_speed, segment=segment)
                        terminal_status = 'completed' if reason == 'game over' else 'stopped'
                        results['runs'].append({'run': run_number, 'score': tracker.score,
                                                **metrics.summary(time.perf_counter() - run_start),
                                                'score_so_far': tracker.score,
                                                'final_score': tracker.score if terminal_status == 'completed' else None,
                                                'status': terminal_status,
                                                'seconds': round(time.perf_counter() - run_start, 2), 'reason': reason})
                        recorder.finish_run(run_number)
                        print(f'[RUN {run_number}] pipes={tracker.score} reason={reason}', flush=True)
                        if run_number >= args.runs:
                            outcome = terminal_status
                            break
                        # The banner appears before the retry controls finish
                        # fading in. Wait for an actionable screen instead of
                        # sending a key into the death animation.
                        page.wait_for_timeout(1000)
                        frame = game.capture(roi)
                        sent_at = time.perf_counter()
                        game.restart(frame, roi)
                        recorder.action('restart', sent_at, run_number, segment)
                        page.wait_for_timeout(500)
                        sent_at = time.perf_counter()
                        game.flap()
                        recorder.action('flap', sent_at, run_number, segment)
                        page.wait_for_timeout(130)
                        run_number += 1
                        segment += 1
                        strategy, tracker, perception = (strategy.new_run(), RunTracker(vision_timeout=vision_config.tracking.lost_seconds),
                                                         VisionTracker(vision_config))
                        metrics = BrowserMetrics()
                        run_start = time.perf_counter()
                        continue
                    before_plan = time.perf_counter()
                    flap, status = strategy.should_flap(obs, observed_at)
                    decision_ms = (time.perf_counter()-before_plan)*1000
                    model_seconds += .2 * (decision_ms/1000 - model_seconds)
                    action_sent_at = None
                    if flap and not args.dry_run:
                        action_sent_at = time.perf_counter()
                        if args.policy == 'dqn' and not strategy.action_is_fresh(observed_at, action_sent_at, args.input_latency_ms / 1000):
                            flap = False
                            action_sent_at = None
                            strategy.policy_decision['reason'] = 'dispatch exceeded frame age'
                            status = 'dqn: dispatch exceeded frame age'
                        else:
                            strategy.note_flap_sent(action_sent_at, args.input_latency_ms / 1000)
                            game.flap()
                            flap_count += 1
                    metrics.decision(getattr(strategy, 'policy_decision', None))
                    tick += 1
                    render_time = time.perf_counter()
                    recorder.record(run_number, render_time - start, age, tracker.score, frame, obs,
                                    flap and not args.dry_run, strategy, captured_at=observed_at,
                                    processed_at=processed_at, vision_speed=vision_speed, action_sent_at=action_sent_at,
                                    decision_ms=decision_ms, segment=segment)
                    show_debug = args.debug and render_time - last_debug >= 1 / args.debug_fps
                    save_debug = args.save_debug and render_time - last_saved >= 1
                    if show_debug or save_debug:
                        overlay = annotate(frame, obs, f'run={run_number} pipes={tracker.score} | {status}')
                        if save_debug:
                            recorder.overlay(tick, overlay)
                            last_saved = render_time
                        if show_debug:
                            last_debug = render_time
                            preview.show(overlay)
                    now = time.perf_counter()
                    if now - last_report > 2:
                        print(f'[INFO] run={run_number} pipes={tracker.score} fps={(tick-last_report_tick) / (now-last_report):.1f} age={max(0, now-observed_at)*1000:.0f}ms '
                              f'vision={obs.state} B={obs.bird_quality.confidence:.2f} P={obs.pipe_quality.confidence:.2f} | {status}', flush=True)
                        last_report = now
                        last_report_tick = tick
                    remaining = 1 / args.fps - (time.perf_counter() - frame_start)
                    if remaining > 0:
                        # Let frame events arrive while waiting for the next tick.
                        page.wait_for_timeout(remaining * 1000)
                elapsed = time.perf_counter() - start
                if len(results['runs']) < run_number:
                    results['runs'].append({'run': run_number, 'score': tracker.score,
                                            **metrics.summary(time.perf_counter() - run_start),
                                            'score_so_far': tracker.score, 'final_score': None, 'status': outcome,
                                            'seconds': round(time.perf_counter() - run_start, 2), 'reason': outcome})
                    recorder.finish_run(run_number)
                scores = [result['score'] for result in results['runs']]
                completed = [r['final_score'] for r in results['runs'] if r['status'] == 'completed']
                completed_mean = f'{sum(completed)/len(completed):.1f}' if completed else 'n/a'
                print(f'[DONE] observed_best={max(scores, default=0)} observed_mean={sum(scores)/max(1, len(scores)):.1f} '
                      f'completed_mean={completed_mean} '
                      f'runs={len(scores)} fps={tick/max(.001, elapsed):.1f} flaps={flap_count} elapsed={elapsed:.1f}s', flush=True)
                return 0
            except KeyboardInterrupt:
                outcome = 'interrupted'
                raise
            except Exception:
                outcome = 'error'
                raise
            finally:
                try:
                    if recorder is not None:
                        try:
                            if len(results['runs']) < run_number:
                                results['runs'].append({'run': run_number, 'score': tracker.score,
                                                        **metrics.summary(time.perf_counter() - run_start),
                                                        'score_so_far': tracker.score, 'final_score': None, 'status': outcome,
                                                        'seconds': round(time.perf_counter() - run_start, 2), 'reason': outcome})
                                recorder.finish_run(run_number)
                        finally:
                            results['status'] = outcome
                            recorder.close(results)
                finally:
                    try:
                        if game is not None:
                            game.close()
                        if preview is not None:
                            preview.close()
                    finally:
                        browser.close()


def main() -> None:
    args = make_parser().parse_args()
    try:
        sys.exit(run(args))
    except KeyboardInterrupt:
        print('\n[INFO] Stopped by user.')
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
