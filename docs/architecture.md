# Architecture and tuning guide

## Flow

1. `browser.GameBrowser` finds the largest canvas element across frames and subscribes to Chromium's JPEG screencast via Playwright/CDP. It retains only the newest frame, acknowledges received frames, and sends Space as a browser keyboard event. Static pages or offscreen regions use direct screenshots as a fallback.
2. `vision.detect` converts the image to HSV, uses yellow and green masks, prefers bird candidates with a red beak, and tracks their horizontal position. Pipe detection finds the largest opening between green sections so score digits drawn over the upper pipe do not move the detected gap.
3. `strategy.Strategy` estimates vertical movement from consecutive observations and chooses whether to flap, respecting cooldown.
4. `__main__.run` coordinates frames and actions, optionally displaying/saving overlay images. Its default viewport is 480 × 720 and its target is 40 FPS; the actual measured average is printed in the console. Known ad loaders are filtered on flappybird.io unless `--allow-ads` is set.

## Coordinate system

Screenshot origin is upper-left: increasing Y means descending on screen. Output images are normalized to CSS pixels so Windows display scaling does not inflate the frame size. `Pipe.gap_top` and `gap_bottom` are relative to the captured canvas or ROI. No browser JavaScript game internals are accessed. If the whole page is captured instead of the canvas, all colors and coordinates may be wrong.

## Tune HSV colors

Edit the two masks in `src/flappy_auto/vision.py`:

```python
bird_mask = _color_mask(hsv, (13, 75, 90), (33, 255, 255))
pipe_mask = _color_mask(hsv, (38, 65, 45), (92, 255, 255))
```

HSV here follows OpenCV's convention: H from 0–179, S/V from 0–255. Different browser games, night themes, skins and anti-aliasing require different thresholds. Run `--capture-only artifacts/screenshot.png` to inspect actual pixels. In the debug window, the red circle indicates detected bird, the magenta rectangle indicates target gap. If detection misses, inspect the image and tune thresholds. For non-canvas pages use `--roi X Y W H` after measuring screenshot pixel coordinates.

## Tune controller

Modify `Strategy` defaults in `src/flappy_auto/strategy.py`:

- `min_flap_interval` (seconds): limits flap frequency.
- `deadband_fraction`: how far below target before flapping.
- `anticipation_seconds`: predict altitude based on recent measured velocity.
- `target_offset_fraction`: bias target Y relative to center of gap.

The controller is heuristic and not reinforcement learning. Avoid interpreting the demo result as guaranteed success on the actual website.

## Ideas for future milestones

- Interactive color calibration and region selection.
- Learn pipe velocity from previous frames and dynamic collision prediction.
- Retry flow based on detecting menu/game-over state visually.
- Debugging dashboard with score inferred from screenshots.
- Self-contained reinforcement learning environment with reproducible physics and reward metrics.
- Smoke test via headless Chromium against the bundled demo.
