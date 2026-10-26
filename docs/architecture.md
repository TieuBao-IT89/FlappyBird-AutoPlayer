# Architecture and tuning guide

## Flow

1. `browser.GameBrowser` finds the largest canvas element across frames and subscribes to Chromium's JPEG screencast via Playwright/CDP. It retains only the newest frame, acknowledges received frames, and sends Space as a browser keyboard event. Static pages or offscreen regions use direct screenshots as a fallback.
2. `vision.detect` downsamples large images to at most 320 pixels wide, converts the image to HSV, and prefers compact yellow bird candidates with an eye and red beak. A tracked bird can enter the upper score area. Pipe detection samples surviving green pixels across the full pipe width, including when score digits cover the end of an upper pipe. `VisionTracker` matches moving pipes and retains median gap geometry across their visible lifetime. Results are scaled back to screenshot coordinates.
3. `strategy.Strategy` estimates velocity at the image timestamp. It fits separate flight arcs and median-filters gravity and jump strength. The intersection of the predicted pre-input arc and the observed post-input arc estimates additional key delivery delay, instead of treating every delayed jump as a change in jump strength. A diverse beam search considers flap/coast sequences over approximately 1.65 seconds, accounting for image age, planning time, learned input delay, cooldown and visible pipes. Collision checks include the entire movement between steps, apexes and the trailing edge of the current pipe.
4. `__main__.run` coordinates frames and actions. Its default viewport is 480 × 720 and its target control rate is 60 FPS. All durations use `perf_counter`, since older Python builds on Windows can quantize `monotonic` at 15.6 ms. CDP frame timestamps are synchronized to this clock using the shortest observed delivery delay. The timestamp exposed to the controller belongs to the consumed image, rather than a subsequently received image. Known ad loaders are filtered on flappybird.io unless `--allow-ads` is set; background throttling is disabled and focus is emulated for the game page.
5. `session.RunTracker` counts the nearest pipe once when its centre crosses the bird. Switching to the next pipe confirms a crossing missed between frames. A broad red banner, missing bird, or frozen gameplay ends a run. Retries wait for the death animation, close an optional name-entry dialog by its visible cross, click the visible green PLAY button when available, then send Space. `Strategy.new_run()` preserves learned physics and delay while resetting flight history.
6. `workers.LatestWorker` keeps one pending image while a dedicated worker performs OpenCV display or disk encoding. New images replace pending work instead of delaying control. `DebugWindow` owns all its GUI calls on that worker, including Q handling and cleanup. `recording.Recorder` writes telemetry and run results into a new `session_*` directory per launch. Its background encoder retains a rolling buffer of 20 JPEGs at up to 10 FPS; run endings and interruption drain pending work before flushing the buffer. No game JavaScript state is read.

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
- `latency_seconds`: delay between the captured image and the key taking effect; the runner updates this from image age.
- `gravity`, `jump_speed`, `pipe_speed`: initial estimates in screenshot pixels and seconds. Zero selects scale-based defaults; observations update them during play.

The planner uses an initial step based on the measured control interval (25–60 ms, with allowance for jitter), followed by 23 steps of 70 ms. It retains up to 32 candidates, merging similar positions, velocities and cooldowns while keeping the first actions separate. This prevents near-identical histories from crowding the safe waiting trajectory out of the beam. Increasing the horizon or beam width increases CPU work.

Before accepting a flap, a separate analytic check projects the entire jump through each visible pipe's overlap interval. A flap that will unavoidably hit an upper pipe or the ceiling is rejected, even if the approximate search prefers it. Further flaps cannot undo an upper-pipe collision. This matters when dropping from a high opening to a low opening: jumping too early can place the apex inside the incoming upper pipe.

Physics calibration uses separate flight arcs so overlapping fits cannot amplify one noisy observation. Jump strength is fitted with the shared gravity estimate. Samples around pending input are excluded. Pipe speed is fitted over multiple frames of the same pipe's right edge, so edge jitter, clipped pipes and switching targets do not produce abrupt speed changes. The collision radius uses recent sprite sizes rather than one animated frame.

A flap decision retains the velocity at the image timestamp until the expected delivery time. The new impulse is applied only to observations after delivery; the planner also accounts for a pending impulse during its latency projection. The runner reports the actual key dispatch time after planning, rather than relying only on average planner duration. This prevents a delayed input from being interpreted as an immediate jump followed by an unexpected fall.

`--input-latency-ms` supplies the initial estimate for key delivery (8 ms by default). The runner also adds measured image age and planning time; the controller learns the remaining delay from pixels. `--runs` chooses the number of attempts, while `--seconds` is the total session budget. This is model-based control, without reinforcement learning or reads of game state.

## Ideas for future milestones

- Interactive color calibration and region selection.
- Broader theme/skin support for visual retry controls.
- Score OCR and a dashboard comparing completed runs.
- Self-contained reinforcement learning environment with reproducible physics and reward metrics.
- Smoke test via headless Chromium against the bundled demo.
