# DQN trong trình duyệt

`rule` vẫn là mặc định. DQN dùng ảnh chụp qua Playwright và OpenCV, không đọc biến JavaScript của game. Cần cài extra DQN và Chromium theo [hướng dẫn huấn luyện](dqn.md), giữ `best.zip` và `best.json` cùng thư mục. Bộ nạp hiện có kiểm tra schema, phiên bản môi trường, SHA256 và không gian observation/action trước khi mở browser.

```powershell
python -m pip install -e ".[dev,dqn]"
python -m playwright install chromium
Get-ChildItem artifacts -Filter best.zip -Recurse

python -m flappy_auto --demo --policy rule --runs 5 --debug
python -m flappy_auto --demo --policy dqn --model artifacts/dqn/<session>/best.zip --runs 5 --debug --save-debug artifacts/dqn-browser
# Sau khi kiểm thử demo:
python -m flappy_auto --policy dqn --model artifacts/dqn/<session>/best.zip --runs 5 --debug --save-debug artifacts/dqn-site
```

`--model` bắt buộc với DQN và không được dùng với rule. Không tự huấn luyện nếu thiếu checkpoint. `--seconds` giới hạn toàn phiên; `--seconds 0 --runs 5` chờ đủ năm lượt. `--dry-run` ghi quyết định nhưng không gửi phím.

## Hợp đồng observation

Chọn toàn bộ sân chơi khi hiệu chỉnh ROI; một vùng chỉ chứa chim/khe ống sẽ làm thay đổi ý nghĩa đặc trưng. Kích thước ảnh bao gồm border hoặc vùng giao diện cũng tạo sai lệch so với tọa độ simulator; phép chia kích thước không tự sửa sai lệch đó.

| Đặc trưng | Công thức từ ảnh | Giới hạn |
| --- | --- | --- |
| Độ cao chim | `bird.y / height` | 0…1 |
| Vận tốc dọc | `pixels_per_second * 600 / height / 400` | −2…2 |
| Khoảng cách tới mép phải ống | `(pipe.right - bird.x) / width` | −1…3 |
| Mép trên khe | `gap_top / height` | 0…1 |
| Mép dưới khe | `gap_bottom / height` | 0…1 |

Output là `float32`, shape `(5,)`, cùng thứ tự và giới hạn với `FlappyEnv`. Chọn ống đầu tiên theo x còn thỏa `pipe.right >= bird.x - bird.radius`, kể cả khi tâm chim đã vượt ống. Không dựng ống giả khi ngoài màn hình. Đầu vào không hữu hạn hoặc hình học không hợp lệ bị từ chối trước khi clip.

Vận tốc lấy từ chênh lệch vị trí giữa hai ảnh đáng tin liên tiếp và **timestamp ảnh**, không dùng FPS mục tiêu hay thời điểm nhận ảnh. Đây là vận tốc trung bình trong khoảng lấy mẫu, có độ trễ so với vận tốc tức thời khi train. Không áp vật lý simulator để ngoại suy website. Sau mất nhận diện, timestamp lùi/trùng, khung hình quá cũ, resize hoặc khoảng lấy mẫu > `max_frame_age`, cần làm nóng lại vận tốc. Nhảy vị trí bất thường bị chặn. Các ngưỡng confidence và tuổi ảnh dùng `vision.tracking` của profile.

## Quyết định và ghi nhận

`model.predict(state, deterministic=True)` trả `0=coast`, `1=Space`. Tối đa một suy luận mỗi 1/60 giây theo timestamp ảnh; khi browser chậm, không gửi bù hành động. DQN không dùng cooldown/lập kế hoạch của rule. Giữ logic học chuyển động ảnh để hỗ trợ ghép ống và telemetry; vận tốc đầu vào DQN do adapter đo riêng.

Ảnh không đáng tin không được suy luận. Kiểm tra tuổi ảnh lần nữa sau suy luận và ngay trước gửi Space, bao gồm độ trễ phím cấu hình. Khởi động/chơi lại vẫn là thao tác vòng đời của browser; lịch sử adapter/cadence được xóa cho từng ván và khi khôi phục hộp thoại tên. Bot chỉ đóng hộp thoại tên, không nhập hoặc gửi điểm lên leaderboard.

Với `--save-debug` hoặc `--record-full`, `events.jsonl` thêm `policy_decision` gồm năm đặc trưng, action dự đoán, `inference_ms` và lý do chặn. `action_sent` ghi việc thực sự gửi phím; action=1 có thể bị chặn do độ trễ hoặc dry-run. `session.json`/`results.json` chứa policy và metadata checkpoint. Mỗi ván có FPS thực, số frame không đủ điều kiện nhận diện, frame cũ, số suy luận/quyết định bị chặn và tuổi ảnh/độ trễ suy luận trung bình. Frame cuối có banner cũng được tính vào số frame nhận diện không dùng được; đây không phải tỷ lệ lỗi được gán nhãn thủ công.

## Đánh giá tái lập

```powershell
python scripts/evaluate_browser.py --model artifacts/dqn/<session>/best.zip --trials 3 --seconds 25 --output artifacts/browser-demo
# Cùng cấu hình, website công khai; thêm --headed nếu cần browser có cửa sổ:
python scripts/evaluate_browser.py --model artifacts/dqn/<session>/best.zip --trials 3 --seconds 25 --site --output artifacts/browser-site
```

Script mở browser mới cho mỗi lượt, xen kẽ thứ tự rule/DQN và đảo thứ tự ở lượt kế tiếp, cùng viewport 480×720, target 60 FPS, mặc định headless và không debug window. Không mang vật lý đã học giữa các lượt. Ghi toàn bộ lượt vào `trials.csv`/`trials.json`, kể cả lỗi; giữ command, console và session. Checkpoint được kiểm tra trước khi bắt đầu. Tệp `config.json` giữ cấu hình, code provenance và metadata mô hình. Chỉ chạy một bộ đánh giá tại một thời điểm để tránh cạnh tranh CPU.

`score_so_far` là số ống bộ theo dõi thị giác đếm được, có thể khác điểm trên màn hình. Chỉ banner game-over được xác nhận mới có `status=completed` và `final_score`. Mất chim/ống, game ngừng chuyển động hoặc ảnh bị treo là `stopped`; thời hạn là `time_limit`; Ctrl+C là `interrupted`. Demo dùng chữ game-over trắng khác banner đỏ website, nên có thể kết thúc bằng `stopped` và không có final score. Không đổi các lượt này thành ván hoàn tất dựa vào suy đoán.

So sánh với simulator phải nêu rõ seed, thời hạn, vật lý, hành động bắt đầu, mép ống nhìn thấy, border, tốc độ cập nhật, độ trễ và sai số CV. Website không có cơ chế seed chung với Gymnasium. Điểm simulator 22,6 là trung bình quan sát của benchmark 60 giây, không phải trung bình điểm cuối, và không trực tiếp so sánh với lượt browser 25 giây. Xem [kết quả browser](benchmarks/dqn-browser/README.md).

## Kiểm thử

```powershell
python -m pytest -q
$env:FLAPPY_BROWSER_SMOKE = "1"
python -m pytest -q tests/test_dqn_browser_smoke.py
```

Smoke opt-in dùng Chromium thật, demo thật và model giả để kiểm tra đường đi từ ảnh tới Space/telemetry; kiểm tra checkpoint thật được thực hiện riêng bằng CLI. CI cài Chromium và chạy smoke này. Unit tests đối chiếu trực tiếp normalization với `FlappyEnv`, kiểm tra timestamp/confidence, vận tốc, reset, action bất hợp lệ, suy luận chậm, cadence và bộ nạp checkpoint.
