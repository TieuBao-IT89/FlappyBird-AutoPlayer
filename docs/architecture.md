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

## Hiệu chỉnh nhận diện và profile

Chạy `python -m flappy_auto --calibrate --profile profiles/site.json` để chọn ROI bằng chuột, lấy nhiều mẫu màu chim/ống, chỉnh H/S/V và xem ba ảnh đồng thời. Thêm `--demo` để hiệu chỉnh bản ngoại tuyến. Chạy bot bằng `--profile <tệp>` sau khi lưu. Quy trình đầy đủ nằm trong README tiếng Việt.

`profiles.GameProfile` kiểm tra định dạng JSON phiên bản 1, URL, selector, viewport, gốc tọa độ, kích thước ảnh tham chiếu, ROI và `VisionConfig`. CLI ghi đè giá trị tương ứng trong profile; mặc định cũ được giữ khi không có profile. `--full-frame` xóa ROI đã lưu. Profile demo lưu chế độ demo thay vì URL máy chủ tạm thời. Tệp mẫu trong `examples/` không có ROI và kích thước tham chiếu; profile lưu từ giao diện luôn có kích thước thật. Việc lưu dùng tệp tạm và thay thế nguyên tử để không ghi dở profile đang dùng.

`calibration.CalibrationSession` giữ ảnh gốc, ROI, mẫu và ngưỡng; không phụ thuộc Tk. `DisplayTransform` ánh xạ chuột qua tỷ lệ ảnh và khoảng đệm trong cửa sổ về pixel CSS. Mỗi lần nhấp lấy mẫu vùng 5 × 5, dùng trung vị S/V và medoid Hue theo khoảng cách vòng tròn. Nhiều mẫu xác định cung Hue nhỏ nhất rồi mở rộng biên; dải có H thấp > H cao đi qua 179 → 0. Các ô ngưỡng cho phép chỉnh trực tiếp; gợi ý mỏ/mắt có thể tắt trong giao diện hoặc đổi màu trong JSON.

`CalibrationWindow` dùng Tkinter có sẵn trong Python, tạo ảnh PNG bằng OpenCV và `PhotoImage`, không cần Pillow. Tk và Playwright đồng bộ cùng luồng sở hữu; chỉ chụp ảnh theo thao tác hoặc bộ hẹn giờ 150 ms khi bật ảnh trực tiếp. Nhấn vào ảnh sẽ đóng băng trước khi chọn vùng/lấy mẫu. Hiệu chỉnh kết thúc trước vòng điều khiển, không chạy cùng cửa sổ OpenCV hoặc worker gỡ lỗi. Bộ xử lý lỗi callback hiển thị lý do trên giao diện.

Xem trước gọi `vision.detect` với cùng ROI, `VisionConfig` và giới hạn xử lý 320 px như bot. Hai mặt nạ lấy từ cùng hàm `vision_masks` trên ảnh đã thu nhỏ. `VisionTracker` giữ cấu hình qua lần chơi lại; theo dõi thời gian vẫn có thể làm mượt tọa độ ống giữa các khung hình. Browser chuẩn hóa ảnh về kích thước CSS trước khi áp dụng ROI. Khi nạp profile, bot kiểm tra kích thước và gốc tọa độ; tiếp tục kiểm tra lại phần tử khoảng mỗi giây để báo lỗi khi bố cục đổi. Hiệu chỉnh lại sẽ bỏ ROI cũ khi kích thước khác, giữ ngưỡng màu.

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

- Vision confidence, lost-bird recovery and broader game profiles.
- Broader theme/skin support for visual retry controls.
- Score OCR and a dashboard comparing completed runs.
- Self-contained reinforcement learning environment with reproducible physics and reward metrics.
- Smoke test via headless Chromium against the bundled demo.
