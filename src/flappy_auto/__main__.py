"""Command line entry point: python -m flappy_auto [--demo]."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import cv2
from playwright.sync_api import sync_playwright

from .browser import GameBrowser, block_advertising
from .demo_server import demo_server
from .strategy import Strategy
from .vision import annotate, detect


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Play Flappy Bird visually, using OpenCV + Playwright.')
    parser.add_argument('--demo', action='store_true', help='Use the bundled offline demo instead of flappybird.io')
    parser.add_argument('--url', default='https://flappybird.io/', help='Public game URL (or another compatible page)')
    parser.add_argument('--selector', default='canvas', help='CSS selector for game region, including in iframes')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), help='Crop screenshot (canvas-relative, else viewport-relative)')
    parser.add_argument('--seconds', type=float, default=40, help='Max run duration in seconds, 0 = unlimited')
    parser.add_argument('--fps', type=float, default=40, help='Target capture rate; actual rate may be lower (default: 40)')
    parser.add_argument('--viewport', type=int, nargs=2, default=(480, 720), metavar=('WIDTH', 'HEIGHT'), help='Browser viewport (default: 480 720)')
    parser.add_argument('--debug', action='store_true', help='Show computer vision overlay; press Q to stop')
    parser.add_argument('--save-debug', type=Path, help='Directory for occasional annotated screenshots')
    parser.add_argument('--capture-only', type=Path, help='Capture one screenshot then exit; useful for adjusting detection')
    parser.add_argument('--dry-run', action='store_true', help='Show decisions but do not press Space')
    parser.add_argument('--no-start', action='store_true', help='Do not click to start; start the game manually')
    parser.add_argument('--headless', action='store_true', help='Hide browser (primarily for CI and local demo)')
    parser.add_argument('--browser-executable', type=Path, help='Optional path to installed Chrome/Chromium executable')
    parser.add_argument('--allow-ads', action='store_true', help='Allow third-party ads on flappybird.io (disabled for performance)')
    return parser


def run(args: argparse.Namespace) -> int:
    if args.fps <= 0 or args.seconds < 0 or min(args.viewport) <= 0:
        raise ValueError('--fps and viewport dimensions must be positive; --seconds must be nonnegative')
    # Small vision frames do not benefit from scheduling a worker per CPU core.
    cv2.setNumThreads(2)
    server_context = demo_server() if args.demo else nullcontext(args.url)
    with server_context as url:
        with sync_playwright() as playwright:
            launch_options = {'headless': args.headless}
            if args.browser_executable:
                launch_options['executable_path'] = str(args.browser_executable)
            browser = playwright.chromium.launch(**launch_options)
            game = None
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
                frame = game.capture(roi)
                print(f'[INFO] Screenshot dimensions: {frame.shape[1]}x{frame.shape[0]}')
                if args.capture_only:
                    args.capture_only.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.capture_only), frame)
                    print(f'[INFO] Saved capture to {args.capture_only}')
                    return 0
                if args.save_debug:
                    args.save_debug.mkdir(parents=True, exist_ok=True)
                if not args.no_start and not args.dry_run:
                    game.start()
                    page.wait_for_timeout(130)
                strategy = Strategy()
                previous = None
                start = time.monotonic()
                tick = 0
                flap_count = 0
                last_report = start
                print('[INFO] Bot running. Stop with Ctrl+C' + (' or Q in debug window.' if args.debug else '.'))
                while not args.seconds or time.monotonic() - start < args.seconds:
                    frame_start = time.monotonic()
                    frame = game.capture(roi)
                    observed_at = game.frame_time
                    obs = detect(frame, previous)
                    previous = obs.bird
                    flap, status = strategy.should_flap(obs, observed_at)
                    if flap and not args.dry_run:
                        game.flap()
                        flap_count += 1
                    tick += 1
                    if args.debug or (args.save_debug and tick % max(1, round(args.fps)) == 0):
                        overlay = annotate(frame, obs, status)
                        if args.save_debug and tick % max(1, round(args.fps)) == 0:
                            cv2.imwrite(str(args.save_debug / f'frame_{tick:05d}.png'), overlay)
                        if args.debug:
                            cv2.imshow('Flappy AutoPlayer - press Q to quit', overlay)
                            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                                break
                    now = time.monotonic()
                    if now - last_report > 2:
                        print(f'[INFO] fps={tick / (now - start):.1f} frames={tick} flaps={flap_count} bird={bool(obs.bird)} pipe={bool(obs.pipe)} | {status}')
                        last_report = now
                    remaining = 1 / args.fps - (time.monotonic() - frame_start)
                    if remaining > 0:
                        # Let frame events arrive while waiting for the next tick.
                        page.wait_for_timeout(remaining * 1000)
                print(f'[DONE] frames={tick}, flaps={flap_count}, elapsed={time.monotonic()-start:.1f}s')
                return 0
            finally:
                if game is not None:
                    game.close()
                if args.debug:
                    cv2.destroyAllWindows()
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
