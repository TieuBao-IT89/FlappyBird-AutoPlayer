# DQN từ simulator sang browser — Issue #15

Đo ngày **26/10/2026**, cùng checkpoint best ở bước 140.000 của [đợt train 200k](../dqn-200k/README.md), SHA256 `8395ab6a9effd3baf01736693ca0160a060911d9697292b532f623a28ddc89c7`. Không retrain hoặc chọn lại checkpoint bằng kết quả browser.

Mỗi nguồn có ba lượt mỗi policy, browser mới từng lượt, xen kẽ rule/DQN rồi đảo thứ tự ở lượt thứ hai. Cùng viewport 480×720, target 60 FPS, headless, không debug window, mặc định vision, giới hạn **25 giây mỗi lượt**, chẩn đoán rolling bật. Đây là mẫu nhỏ trên một máy Windows; FPS thực phụ thuộc kết xuất và thời gian lập kế hoạch. Website không có seed chung, demo dùng chuỗi khe cố định; không phải các cặp seed tương đương với simulator.

| Nguồn / policy | Điểm quan sát từng lượt | Điểm quan sát TB | Thời lượng lượt TB (s) | FPS TB | Frame nhận diện không dùng được TB | Completed / time_limit / stopped |
| --- | --- | --- | --- | --- | --- | --- |
| Demo / rule | 11, 11, 11 | 11,00 | 25,02 | 37,56 | 1,33 | 0 / 3 / 0 |
| Demo / DQN | 4, 2, 3 | 3,00 | 9,24 | 44,26 | 38,67 | 0 / 0 / 3 |
| flappybird.io / rule | 13, 0, 0 | 4,33 | 11,00 | 22,73 | 44,33 | 2 / 1 / 0 |
| flappybird.io / DQN | 0, 0, 0 | 0,00 | 1,74 | 27,24 | 46,67 | 3 / 0 / 0 |

Mỗi giá trị TB ở bảng là trung bình trên ba lượt, kể cả lượt xấu. Thời lượng đo đến lúc CLI xác nhận kết thúc hoặc dừng, có thể gồm thời gian chờ timeout nhận diện; không phải thời điểm va chạm chính xác. Điểm quan sát lấy từ bộ đếm ống thị giác, không OCR. `detection_failures` gồm frame thiếu ống, ảnh cũ, phục hồi và màn game-over; đây là số frame không đủ điều kiện điều khiển, không phải số lỗi CV đã được gán nhãn.

**Điểm cuối:** tất cả demo và lượt website rule 13 điểm đều chưa có `final_score`. Demo DQN dừng do `pipe tracking lost`, nên không khẳng định đây là ván va chạm hoàn tất. Banner game-over website xác nhận hai lượt rule và ba lượt DQN có điểm cuối 0; trung bình điểm cuối của hai nhóm đó là 0 với n=2 và n=3. Không gán điểm 11/13 ở ván hết giờ thành điểm cuối.

**DQN demo:** thực sự chạy 470, 230, 355 suy luận tất định qua ảnh và điều khiển Space. Độ trễ suy luận TB từng lượt 0,772 / 0,789 / 0,844 ms; trung bình các lượt 0,802 ms. Tuổi ảnh TB 25,436 / 25,459 / 27,302 ms. DQN chưa vượt rule trong cấu hình này. Adapter chuẩn hóa đúng hợp đồng chưa loại bỏ độ trễ vận tốc sai phân, FPS thấp hơn 60, biên canvas, cap ống vẽ rộng hơn thân, và khác biệt tích phân thời gian giữa simulator với requestAnimationFrame. Không có thí nghiệm tách biến để quy trách nhiệm cho một yếu tố cụ thể.

**DQN website:** cả ba lượt có **0 suy luận**. Trước khi chim rơi, ảnh không cung cấp ống mục tiêu đủ điều kiện; adapter từ chối state và không gửi Space từ DQN. Chỉ thao tác khởi động của browser được thực hiện. Đây là hạn chế khởi động với cổng an toàn khi ống chưa xuất hiện, chưa phải phép đo chất lượng hành động mạng neural trên website. Muốn nghiên cứu tiếp cần đặc tả riêng trạng thái trước ống hoặc bước khởi động phù hợp, rồi đánh giá lại; đợt này không bịa hình học ống hoặc tự chuyển qua rule. Rule cũng biến động đáng kể giữa các lượt và không chứng minh website ổn định.

## So với Gymnasium

Benchmark simulator cũ dùng 20 test seed 2000–2019 và tối đa **60 giây mô phỏng**, DQN quan sát TB 22,60; 6 va chạm/14 hết giờ, trung bình điểm cuối 10,00 (n=6). Baseline reactive simulator là policy cố định, khác planner rule browser. Vì khác thời hạn, seed, không gian hiển thị, cách đếm ống, hành động khởi động, nhận diện và độ trễ, không tính tỷ lệ suy giảm phần trăm từ 22,60 xuống 3,00 hoặc 0. Xem toàn bộ [trials simulator](../dqn-200k/trials.csv).

## Dữ liệu và tái lập

- [demo-trials.csv](demo-trials.csv) và [site-trials.csv](site-trials.csv): tất cả 12 lượt có truy cập browser thực, không lọc kết quả.
- [demo-config.json](demo-config.json), [site-config.json](site-config.json): cấu hình, thứ tự chạy, checkpoint, source provenance.
- [demo-sessions.json](demo-sessions.json), [site-sessions.json](site-sessions.json): metadata và kết quả từng session, gồm Chromium, kích thước ảnh, thông số ghi nhận và điểm chưa hoàn tất.
- [network-blocked-trials.csv](network-blocked-trials.csv): giữ riêng sáu lần thử ban đầu bị sandbox chặn `net::ERR_NETWORK_ACCESS_DENIED` trước khi vào game; tất cả là `error`, không tính vào trung bình điểm/FPS. Sau khi được cấp quyền mạng, chạy lại đầy đủ sáu lượt website ở bảng trên. Không thay lỗi hạ tầng bằng điểm 0.
- [source.json](source.json): hash từng file source cuối và hash hai file policy browser. Các lượt chạy trên working tree bẩn từ main base ghi trong config; sau đo chỉ thêm hash provenance và sửa bộ đếm khi khôi phục hộp thoại tên. Không đổi normalization hoặc hành động mô hình từ dữ liệu đánh giá. Hash policy cuối khác bản đo do bổ sung hàm hash, không phải tuning.

Giữ ZIP và log/ảnh/events đầy đủ dưới `artifacts/dqn-browser/` được Git bỏ qua. Bản gọn này không chứa trọng số. Source trong PR tích hợp Issue #15 tương ứng với manifest; provenance gốc không được thay thành commit sạch giả định. Môi trường: Python 3.11.9, NumPy 2.4.6, OpenCV 4.14.0, SB3 2.9.0, PyTorch 2.14.1+cpu; phiên bản Chromium và ảnh thực xem metadata từng session.

```powershell
python scripts/evaluate_browser.py --model artifacts/dqn/<session>/best.zip --trials 3 --seconds 25 --output artifacts/browser-demo
python scripts/evaluate_browser.py --model artifacts/dqn/<session>/best.zip --trials 3 --seconds 25 --site --output artifacts/browser-site
```

Chạy tuần tự, cùng checkpoint/phiên bản/máy và không chạy test nặng cùng lúc. Không kỳ vọng điểm hoặc FPS trùng tuyệt đối: screenshot/compositor/website không tất định. Không nhập tên hoặc gửi leaderboard. Hướng dẫn adapter, safety, telemetry và smoke test ở [DQN browser](../../dqn-browser.md).
