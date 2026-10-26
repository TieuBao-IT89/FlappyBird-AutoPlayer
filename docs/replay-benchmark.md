# Ghi phiên và benchmark phiên bản 1

## Hai phép đo khác nhau

`python -m flappy_auto.replay` phát lại ảnh cố định qua OpenCV. Nó đánh giá observation, độ tin cậy, trạng thái phục hồi và ngữ cảnh các lệnh gốc. Khi thay profile/detector, ảnh tiếp theo vẫn là ảnh của lần chơi cũ. Vì vậy mọi báo cáo đều có `controller_scores_evaluated: false`; không tính điểm hoặc thời gian sống cho chính sách mới trên clip.

`python -m flappy_auto.benchmark` chạy mô phỏng độc lập có phản hồi: Space đổi vận tốc, đổi vị trí tương lai và có thể gây/tránh va chạm. Nó dùng observation hình học có nhiễu để kiểm tra controller, không chạy browser hoặc kiểm chứng màu sắc OpenCV. Môi trường mô phỏng có `environment_version: 1`; không khẳng định khớp chính xác game website.

## Thư mục phiên

- `session.json`: `kind: flappy-session`, `version: 1`, `recording_mode: full|rolling`, `closed`, `frame_size: [width, height]`, `vision`, snapshot `profile`, thông tin `mode`, `code`, phiên bản browser. `recording_stats` và `complete_images` được chốt lúc đóng.
- `events.jsonl`: đọc tuần tự, một đối tượng JSON mỗi dòng, không nạp toàn bộ video vào RAM. `type: frame` gồm `id`, `run`, `segment`, thời gian, tốc độ ống đầu vào tracker, lý do chặn, observation, điểm đang thấy, lệnh thực gửi và ảnh. `type: action` ghi các thao tác start/restart/flap ngoài nhánh quyết định thường.
- `frames/*.png`: ảnh BGR đầu vào đã chuẩn hóa pixel CSS và cắt ROI, lưu không mất dữ liệu trong chế độ full. Overlay nằm riêng, không thay ảnh này.
- `telemetry.csv`: cột chẩn đoán cũ và thêm `frame_id`, `captured_at`, `processed_at`, `decision_ms`, `action_sent_at`, `segment`.
- `results.json`: trạng thái phiên và từng ván; giữ cột `score` cũ để tương thích, thêm `score_so_far`, `final_score`, `status`. Chỉ `completed` có điểm cuối. CLI ghi điểm ước lượng từ ống, chưa có OCR.

`captured_at`, `processed_at`, `action_sent_at` và `sent_at` dùng cùng đồng hồ `perf_counter` của phiên gốc, đơn vị giây, không phải ngày giờ. `elapsed` tính từ lúc bắt đầu phiên; `segment` tăng mỗi lần reset tracker, kể cả phục hồi hộp thoại trong cùng một ván. Replay dùng đúng thời gian xử lý gốc để chạy hạn phục hồi, không dùng tốc độ phát lại hoặc đồng hồ máy hiện tại. Các phím quyết định thường nằm trong frame qua `action_sent` và `action_sent_at`; `dry-run` không đánh dấu là đã gửi.

Nguồn browser vẫn có thể là JPEG của Chromium; PNG giữ nguyên pixel **đã đưa cho detector**, không khôi phục dữ liệu trước JPEG. Đối chiếu chính xác cần đủ mọi frame, segment, timestamp, tốc độ ống và lý do chặn như lúc chơi. Replay so SHA256 toàn bộ file Python trong package (chuẩn hóa LF), phiên bản Python/NumPy/OpenCV và cấu hình, kiểm tra số frame đọc được với metadata để phát hiện log bị cắt cụt. Nó đếm observation khớp log từng frame và tạo SHA256 chuỗi kết quả. Thời gian CPU thực và đường dẫn báo cáo không tham gia hash. Đổi code sau khi ghi sẽ tắt cờ đối chiếu gốc nhưng vẫn cho phép chạy replay và so chuỗi kết quả giữa hai lần phát lại.

## Tải ghi, thiếu dữ liệu và phục hồi

Producer chỉ sao chép ảnh được chọn và tạo telemetry gọn; worker sở hữu mã hóa/ghi file. Worker có một ảnh đang xử lý và tối đa một ảnh thô chờ, một overlay chờ. Hàng đợi gọn có tối đa 512 sự kiện chờ; mỗi batch worker đang xử lý cũng tối đa 512. Khi quá tải, producer thay ảnh chờ hoặc bỏ sự kiện cũ thay vì chờ ổ đĩa. `images_dropped`, `events_dropped`, `actions_dropped` cho biết mức mất dữ liệu; id bị bỏ tạo khoảng trống. Lệnh điều khiển không bị trì hoãn để đợi bộ ghi. Lỗi I/O được chuyển về vòng gọi, không báo phiên ghi thành công.

Trạng thái ảnh: `saved`, `dropped_backpressure`, `sampled_out`, `rolling_discarded`. Bộ đệm rolling mã hóa tối đa khoảng 10 FPS, giữ 20 JPEG; sau khi kết thúc ván, chỉ các ảnh còn giữ được ghi ra và đổi thành `saved`. Việc flush/đóng diễn ra ngoài vòng chơi. Không có hàng đợi ảnh tăng vô hạn. Lượt ghi cứng bị ngắt có thể chưa có metadata/kết quả chốt; `closed: false` không đủ điều kiện đối chiếu chính xác.

Parser từ chối phiên bản không hỗ trợ, id không tăng, kích thước không hợp lệ, đường dẫn ảnh thoát thư mục phiên và cấu hình sai. Ảnh saved bị mất/hỏng mặc định làm replay lỗi và xuất báo cáo phần đã đọc; `--skip-missing` bỏ qua có đếm. `processed_at` cần hữu hạn, không âm và không lùi; giá trị thiếu/sai yêu cầu `--repair-timestamps`. Khi sửa, frame đầu dùng 0 hoặc mốc trước cộng `1/fallback_fps`; `captured_at` thiếu/sai dùng mốc xử lý đã sửa. Giá trị `captured_at` hợp lệ nhưng lặp/lùi được giữ nguyên vì đó có thể là ảnh cũ thật. Mọi sửa timestamp, bỏ ảnh hoặc khoảng trống id đều tắt đối chiếu chính xác.

## Quy tắc benchmark

Mỗi seed gốc và chỉ số trial tạo `scenario_seed = seed + trial * 1000003`. Hai bộ sinh số ngẫu nhiên riêng sinh khe ống và nhiễu. Mọi policy nhận cùng điều kiện đầu và chuỗi môi trường cho cùng cặp, trong cùng ngân sách thời gian mô phỏng. `reactive` là baseline cố định theo vị trí/tốc độ, không được biết vật lý thật. `predictive` dùng `Strategy` và tự học như CLI; không lấy tham số vật lý từ simulator. CLI truyền policy tuổi ảnh và độ trễ dự kiến, tương tự vòng điều khiển thật.

Simulator tích phân parabol tại tần số cố định, chia bước tại thời điểm phím có hiệu lực, kiểm tra toàn bộ khoảng chồng ống và đỉnh quỹ đạo để tránh xuyên vật cản giữa hai bước. Lệnh có độ trễ `decision_delay + input_delay`; ảnh trễ `image_delay`, có nhiễu tọa độ. Thời gian tính thực chỉ ghi số đo, không thay đổi độ trễ mô phỏng. Đổi tần số/nhiễu/độ trễ bằng cấu hình để kiểm tra điều kiện khó hơn. Trần thời gian mô phỏng và worker wall-clock timeout là hai giới hạn khác nhau.

`cold` tạo một Python child process cho mỗi ván, khởi tạo policy từ đầu. `warm` tạo một child cho mỗi policy/seed và gọi `new_run()` giữa các trial; giữ thông số đã học, xóa lịch sử bay. `process_id` giúp kiểm tra việc dùng tiến trình. Các trial có cùng chỉ số được ghép với baseline trong từng chế độ; không gộp cold/warm vào một trung bình.

Mỗi trial luôn có một dòng, kể cả lỗi hoặc chưa chạy. Worker gửi JSONL ngay sau từng trial; quá hạn/gián đoạn giữ phần kết quả đã nhận, gắn `error`/`interrupted` cho lượt đang làm và `not_run` cho lượt chưa bắt đầu. Mỗi nhóm hoàn thành đều ghi báo cáo tạm có cả các lượt còn chờ. Ctrl+C xuất lại báo cáo trước khi CLI trả mã lỗi. Không trộn score bị cắt thời gian vào `completed_scores`; phân phối điểm cuối chỉ mô tả các ván đã hoàn thành trong ngân sách và cần đọc cùng tỷ lệ hết giờ/số mẫu để tránh thiên lệch.

Đầu ra: `config.json` đầy đủ; `results.json` có tất cả trial, tổng hợp, so sánh và provenance; `trials.csv`; `comparisons.csv`; `report.md` tiếng Việt với một lệnh chạy lại. `final_score_delta` chỉ tính khi cả hai ván va chạm; chênh lệch điểm/thời gian quan sát có nhãn giới hạn riêng. Tỷ lệ vượt 20/50 dùng `>`: ván hết giờ đã vượt ngưỡng được tính là đã quan sát thành công; ván hết giờ dưới ngưỡng còn chưa biết. Mọi lần đã thử, kể cả lỗi, vào mẫu số cận dưới; `not_run` báo riêng. Có thêm tỷ lệ chỉ trên ván hoàn thành, kèm số mẫu.

`configured_fps`/`observations_per_sim_second` mô tả lịch quan sát; `processing_fps` là số lần tính trên thời gian CPU thực. `decision_ms_mean`/`decision_ms_p95` đo chi phí controller; tuổi ảnh và độ trễ cấu hình dùng ms. Hash quỹ đạo ghi các trạng thái tại thời điểm quyết định và hành động, không gồm CPU/PID. Tái lập bằng cùng `config.json`, mã nguồn và phiên bản thư viện; không yêu cầu số đo CPU khớp. Dữ liệu ảnh lớn luôn để trong `artifacts/` đã được Git bỏ qua.
