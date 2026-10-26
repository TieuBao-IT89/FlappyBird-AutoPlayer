# FlappyBird-AutoPlayer 🐤

**Bot thử chơi Flappy Bird bằng hình ảnh, không đọc biến game hoặc can thiệp mã JavaScript của trò chơi.**

Dự án sử dụng **Python, OpenCV và Playwright**, gồm bot tự chơi và một bản game ngoại tuyến để kiểm thử trên máy. Khả năng chơi trên website công khai vẫn mang tính **thử nghiệm**; bạn có thể cần hiệu chỉnh ngưỡng màu và vùng chụp khi giao diện trò chơi thay đổi.

## Tính năng chính

- Mở trình duyệt Chromium bằng Playwright.
- Tự chọn phần tử `<canvas>` hiển thị lớn nhất, kể cả trong iframe.
- Nhận luồng khung hình từ Chromium và chỉ xử lý ảnh mới nhất bằng OpenCV, tránh tồn đọng ảnh cũ.
- Nhận diện chim và cặp ống bằng mặt nạ màu HSV; mặc định dùng chim vàng/cam, ống xanh lá.
- Hiệu chỉnh trực quan bằng chuột: chọn vùng game, lấy mẫu màu, xem mặt nạ và lưu/nạp profile JSON.
- Đánh giá độ tin cậy, giữ dự đoán ngắn hạn khi mất dấu, xác nhận chim khi tìm lại và chặn khe ống thay đổi bất thường.
- Bỏ qua ảnh cũ/lặp; tạm ngừng nhảy và học vật lý khi nhận diện chưa đáng tin.
- Ước lượng trọng lực, lực nhảy và tốc độ ống từ hình ảnh để lập kế hoạch nhấn **Space**.
- Chặn lệnh nhảy có quỹ đạo dự kiến đâm vào ống trên, kể cả khi chuyển từ khe cao xuống khe thấp.
- Tự ước lượng độ trễ phím và giữ các thông số đã hiệu chỉnh giữa những ván trong cùng phiên chạy.
- Nhận diện kết thúc ván, tự chơi lại và đếm số ống đã vượt qua từ hình ảnh.
- Tùy chọn lưu dữ liệu chẩn đoán và bộ đệm ảnh khoảng hai giây trước khi chết.
- Hỗ trợ cửa sổ gỡ lỗi, chụp ảnh, quan sát không điều khiển, bắt đầu thủ công và tùy chỉnh vùng chụp.
- Có **game demo ngoại tuyến**, bộ kiểm thử bằng Pytest, CI trên GitHub Actions và giấy phép MIT.
- Có môi trường Gymnasium độc lập và script agent ngẫu nhiên để chuẩn bị thử nghiệm RL.
- Huấn luyện DQN bằng PyTorch/Stable-Baselines3, lưu/nạp checkpoint và đánh giá trên seed tách riêng.

## Cài đặt nhanh trên Windows 11 bằng PowerShell

Cài Python 3.11 hoặc 3.12 và Git. Mở terminal của VS Code tại thư mục dự án, rồi chạy:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,rl]"
python -m playwright install chromium
python -m pytest -q
```

Nếu máy không có lệnh `py -3.11`, hãy cài phiên bản Python phù hợp hoặc thử `py -m venv .venv`. Nếu PowerShell chặn kích hoạt môi trường ảo, chạy `Set-ExecutionPolicy -Scope Process Bypass` rồi kích hoạt lại. Bạn cũng có thể dùng trực tiếp `.\.venv\Scripts\python.exe` mà không cần kích hoạt.

**Nên thử bản demo ngoại tuyến trước**. Sau khi cài các thư viện và Chromium, bản demo không cần kết nối Internet:

```powershell
python -m flappy_auto --demo --seconds 40 --debug
```

Chơi trên **flappybird.io**:

```powershell
python -m flappy_auto --seconds 180 --runs 5 --debug --save-debug artifacts\latest
```

Sau khi cài đặt, bạn cũng có thể nhấp đúp vào `run_demo.cmd` hoặc `run_site.cmd` trong File Explorer.

`run_site.cmd` chạy **ba ván hoàn chỉnh**, có cửa sổ trình duyệt và cửa sổ gỡ lỗi. Tệp này tự nạp `profiles/site.json` nếu có; `run_demo.cmd` tự nạp `profiles/demo.json`. Chưa lưu profile thì hai tệp vẫn dùng mặc định. Nhấn **Q** trong cửa sổ gỡ lỗi hoặc **Ctrl+C** trong terminal để dừng sớm. Bot giữ các thông số vật lý đã học giữa các lần chơi lại. `--seconds` giới hạn thời gian của **toàn bộ phiên chạy**; tệp khởi chạy dùng `--seconds 0` để một ván dài không làm hết thời gian dành cho những ván sau.

Khi bật `--save-debug`, mỗi lần mở bot sẽ tạo một thư mục `session_*` riêng trong đường dẫn đã chọn, giúp giữ lại log của những lần thử trước. Đường dẫn này được in trong terminal. Với `run_site.cmd`, các phiên được lưu dưới `artifacts/latest/`:

- `telemetry.csv`: kết quả nhận diện, lệnh điều khiển, thông số vật lý và độ trễ ước lượng.
- `results.json`: kết quả từng ván và chế độ hiển thị trình duyệt, gỡ lỗi.
- `run_001/`, `run_002/`, …: bộ đệm ảnh trước khi từng ván kết thúc.

Chỉ số `pipes` là số ống bot đếm từ hình ảnh, có thể khác điểm hiển thị trên website. Ván bị dừng giữa chừng được lưu kèm lý do. Bot cũng xử lý hộp thoại nhập tên có thể xuất hiện trước khi chơi lại. Các tệp chẩn đoán không được đưa vào Git.

Website thực tế có thể cần hiệu chỉnh riêng vì đồ họa, lớp giao diện, bố cục canvas và vật lý khác với bản demo.

### Hiệu chỉnh màu và vùng game bằng chuột

Nhấp đúp `run_calibrate.cmd` để hiệu chỉnh website, hoặc chạy:

```powershell
python -m flappy_auto --calibrate --profile profiles\site.json
```

Để thử quy trình với demo ngoại tuyến:

```powershell
python -m flappy_auto --demo --calibrate --profile profiles\demo.json
```

Cửa sổ tiếng Việt dùng **Tkinter** (có trong bản cài Python tiêu chuẩn trên Windows), không cần PySide6 hay Pillow. Trên Linux, nếu Python thiếu Tkinter, cài gói `python3-tk` của hệ điều hành. Hiệu chỉnh mở trình duyệt và cửa sổ riêng; không kết hợp với `--headless`, `--debug`, `--dry-run` hoặc `--capture-only`.

1. Bấm **Chụp ảnh** để cập nhật một ảnh tĩnh, hoặc bật **Ảnh trực tiếp**. Dùng **Bắt đầu**, **Space**, **Chơi lại** để thao tác game trong lúc hiệu chỉnh. Khi ống xuất hiện, nhấp vào ảnh để dừng cập nhật và lấy mẫu.
2. Chọn **Kéo chọn vùng game**, kéo trên khung ảnh gốc. Chọn **toàn bộ sân chơi**, gồm chim, ống trên và ống dưới; không chỉ khoanh quanh chim. Dòng phía trên ảnh ghi rõ gốc tọa độ là phần tử game được tìm theo selector hay toàn bộ viewport. Bấm **Toàn bộ ảnh** nếu không cần cắt.
3. Chọn **Lấy mẫu chim**, nhấp phần thân có màu chính; chọn **Lấy mẫu ống**, nhấp phần màu chính của ống. Mỗi lần nhấp lấy mẫu vùng 5 × 5 pixel và lọc nhiễu; nhấp nhiều vị trí để mở rộng khoảng màu. Tránh nền, mắt, viền và chữ điểm. **Xóa mẫu** xóa lịch sử lấy mẫu của đối tượng, giữ ngưỡng hiện tại.
4. Xem đồng thời ảnh có vòng nhận diện, **mặt nạ chim** và **mặt nạ ống**. Điều chỉnh sáu giá trị HSV cho mỗi đối tượng rồi bấm **Áp dụng ngưỡng / Thử nhận diện**. H từ 0–179; S/V từ 0–255. H thấp lớn hơn H cao biểu diễn dải đi qua 179 → 0 (màu đỏ). S/V thấp phải nhỏ hơn hoặc bằng S/V cao.
5. Tắt **Ưu tiên mỏ đỏ / mắt trắng** nếu nhân vật không có những đặc điểm này. **Màu mặc định** khôi phục toàn bộ ngưỡng và hai tùy chọn; **Mở profile** nạp lại cấu hình đã lưu cho cùng website, selector và viewport.
6. Bấm **Lưu profile…**, chọn tệp JSON, đóng cửa sổ rồi chạy bot:

```powershell
python -m flappy_auto --profile profiles\site.json --debug --runs 3 --seconds 0
# Profile demo tự khởi động máy chủ demo, không lưu URL localhost tạm thời.
python -m flappy_auto --profile profiles\demo.json --debug --seconds 40
```

`--calibrate` chỉ hiệu chỉnh và thoát, không tự chạy bot sau khi lưu. Nếu bỏ `--profile`, đường dẫn lưu gợi ý là `profiles/site.json` hoặc `profiles/demo.json`; profile có sẵn tại đường dẫn đó được nạp lại. Profile gồm phiên bản định dạng, chế độ website/demo, URL website, selector, viewport, ROI, kích thước ảnh tham chiếu theo pixel CSS, gốc tọa độ và ngưỡng màu. Hai gợi ý mỏ/mắt có thể đổi cả dải màu trong JSON. Các ngưỡng theo dõi/phục hồi nằm trong `vision.tracking` như phần bên dưới; cửa sổ hiệu chỉnh giữ nguyên các giá trị này khi lưu.

**Thứ tự ưu tiên:** tham số CLI được truyền rõ ràng → profile → mặc định hiện tại. `--url` chọn website, `--demo` chọn demo (nếu truyền cả hai thì demo được ưu tiên). `--selector`, `--viewport` và `--roi` ghi đè mục tương ứng. Dùng `--full-frame` để bỏ ROI đã lưu. Ngưỡng màu lấy từ profile. Không truyền profile vẫn giữ cách chạy và màu mặc định trước đây.

Khi ảnh game thay đổi kích thước hoặc gốc tọa độ khác profile, bot báo lỗi và yêu cầu hiệu chỉnh lại; thay viewport trên CLI không tắt kiểm tra này. Chạy lại `--calibrate --profile <tệp>` sẽ giữ ngưỡng màu, bỏ ROI cũ nếu bố cục khác để bạn chọn lại. Trong lúc chạy bot, kích thước phần tử được kiểm tra lại khoảng mỗi giây. Tọa độ được chuẩn hóa theo pixel CSS trước khi cắt ROI, kể cả khi Windows dùng DPI lớn hơn 100%. Chromium có thể bố trí phần tử hơi khác khi đổi sang `--headless`; nếu bị báo khác kích thước, chạy cùng chế độ có cửa sổ đã dùng khi hiệu chỉnh.

Thư mục `profiles/` và ảnh chẩn đoán trong `artifacts/` được Git bỏ qua. Khi lưu profile cá nhân, hãy dùng `profiles/`. [Cấu hình mẫu](examples/default-profile.json) giữ màu mặc định và chưa ràng buộc kích thước ảnh (`reference_size: null`, không có ROI); dùng cửa sổ hiệu chỉnh để lưu kích thước thật trước khi dùng lâu dài. Khi mở profile thuộc nguồn khác trong cửa sổ, hãy thoát và mở lại bằng `--calibrate --profile <tệp>` để trình duyệt tải đúng trang.

### Độ tin cậy và phục hồi nhận diện

Trong terminal và cửa sổ gỡ lỗi, `B` là độ tin cậy chim, `P` là độ tin cậy các ống còn ở phía trước hoặc đang chồng lên chim, trong khoảng 0–1. Đây là **điểm bằng chứng theo quy tắc**, không phải xác suất hay tỷ lệ nhận diện đúng đã đo. Chim được chấm theo độ phủ màu (30%), độ gọn hình dạng (25%), kích thước (20%), vị trí ngang (15%) và gợi ý mỏ/mắt (10%). Ống được chấm theo độ phủ màu hai đoạn (45%), cấu trúc khe (30%) và chiều rộng (25%); lấy điểm thấp nhất trong các ống còn liên quan đến va chạm. Bước kiểm tra theo thời gian tiếp tục loại vị trí/kích thước chim bất hợp lý và thay đổi vị trí/chiều rộng/khe của cùng một ống. Chim đứng yên quá lâu, ống đứng lại hoặc đi ngược tốc độ dự kiến cũng bị hạ trạng thái tin cậy, kể cả khi ảnh nền vẫn thay đổi.

| Trạng thái `vision` | Hành vi |
| --- | --- |
| `tracking` | Ảnh mới và nhận diện đủ tin cậy; được lập kế hoạch nhảy và học vật lý. Không có ống trước khi ống xuất hiện vẫn là tình huống hợp lệ. |
| `uncertain` | Mất dấu ngắn hoặc khe/vị trí không nhất quán; có thể vẽ chim dự đoán bằng vòng màu cam hoặc giữ hình dạng ống cũ. Tạm ngừng nhảy và học. |
| `recovering` | Hết thời gian dự đoán; tiếp tục tìm trong vùng mở rộng dần, chờ ứng viên chim chuyển động nhất quán qua thời gian xác nhận. Tạm ngừng nhảy và học. |
| `lost` | Hết hạn phục hồi hoặc gặp màn hình kết thúc/hộp thoại đã biết; ngừng điều khiển. Phiên đang chơi có thể kết thúc với lý do mất chim, mất ống hoặc video đứng. |

Ảnh dự đoán không tăng điểm, không kéo dài thời gian thấy chim và không dùng để học trọng lực, lực nhảy, độ trễ phím hay tốc độ ống. Khe ống sai không được đưa vào trung vị để dần biến thành khe mới. Khi một ống đang đến tạm mất dấu, bot vẫn giữ nó là vật cản trong khoảng cho phép. Màn hình GAME OVER hoặc hộp thoại tên được nhận diện từ ảnh mới sẽ chặn điều khiển ngay, độc lập với điểm màu của chim.

Các bộ đếm dùng giây theo đồng hồ thực, không dùng số frame. Timestamp cũ, ảnh đến trễ hoặc ảnh liên tiếp có cùng pixel không được coi là quan sát mới. Dù Chromium cung cấp timestamp tăng, ảnh lặp không thể xác nhận ứng viên, học vật lý hay kéo dài phục hồi vô hạn. Sau khi đã có ống thật hoặc chim đã chuyển động rõ (biên độ vượt 4% chiều cao ảnh), video đứng quá hạn được ghi riêng là `video stalled / stale frames`. Chế độ trước khi bắt đầu không tự kết thúc chỉ vì ảnh tĩnh.

Để chỉnh theo từng game, sửa `vision.tracking` trong profile JSON, rồi chạy với `--profile`. Có thể chỉ khai báo mục cần đổi; profile phiên bản 1 từ trước vẫn nạp được và dùng mặc định cho các mục thiếu. Xem đầy đủ trong [cấu hình mẫu](examples/default-profile.json).

| Mục cấu hình | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `min_bird_confidence`, `min_pipe_confidence` | `0.75`, `0.60` | Điểm tối thiểu để chấp nhận bằng chứng hình ảnh. |
| `prediction_seconds` | `0.12` | Thời gian tối đa vẽ dự đoán chim kể từ ảnh tin cậy cuối. |
| `pipe_memory_seconds` | `0.30` | Thời gian tối đa giữ hình dạng/vị trí ống thiếu hoặc sai khe. |
| `reacquire_seconds` | `0.05` | Khoảng xác nhận ứng viên chim bằng các ảnh mới có chuyển động. |
| `expand_search_seconds` | `0.30` | Thời gian mở rộng cửa sổ tìm kiếm tới giới hạn quanh chim cũ. |
| `lost_seconds` | `0.80` | Hạn mất dấu/video đứng trong phiên đang chơi. |
| `max_frame_age` | `0.20` | Tuổi ảnh tối đa để chấp nhận bằng chứng mới. |
| `bird_x_tolerance` | `0.05` | Sai lệch ngang ban đầu, theo tỷ lệ chiều rộng ảnh; mở rộng tối đa gấp đôi. |
| `max_bird_speed` | `2.0` | Giới hạn chuyển động dọc, theo chiều cao ảnh mỗi giây. |
| `max_gap_change` | `0.04` | Sai lệch mép khe so với hình dạng đã xác nhận, theo chiều cao ảnh. |
| `max_pipe_shift` | `0.06` | Sai lệch ngang so với vị trí ống dự kiến, theo chiều rộng ảnh. |
| `motion_window_seconds` | `0.30` | Cửa sổ lịch sử ngắn để kiểm tra chuyển động; lịch sử có giới hạn bộ nhớ. |
| `stationary_bird_seconds` | `0.22` | Khoảng quan sát tối thiểu để loại ứng viên chim chỉ đứng yên/rung nhẹ. |
| `pipe_motion_seconds` | `0.12` | Khoảng quan sát tối thiểu để kiểm tra tiến độ di chuyển của ống chưa bị cắt ở biên ảnh. |

Các giá trị phải hữu hạn, dương; điểm/tỷ lệ không quá 1. Thời gian dự đoán, giữ ống, mở rộng tìm kiếm và tuổi ảnh phải nằm trong hạn `lost_seconds`; thời gian xác nhận nhỏ hơn hạn này. Nếu liên tục thấy `uncertain` hoặc `recovering`, hãy kiểm tra ROI, màu và mặt nạ trước khi đổi ngưỡng. Ngừng nhảy khi mất nhận diện có thể khiến ván kết thúc; cơ chế phục hồi không bảo đảm sống qua mọi đoạn che khuất.

Khi dùng `--save-debug`, `telemetry.csv` có thêm `vision_state`, `fresh`, `can_control`, `can_learn` và các cột `bird_*`/`pipe_*` cho độ tin cậy, trạng thái, lý do, cờ dự đoán và thời gian mất dấu. Vòng điều khiển tiếp tục chỉ giữ ảnh mới nhất, không tạo hàng đợi ảnh cho phục hồi.

### Hiệu năng chụp và xử lý ảnh

Vùng hiển thị mặc định là **480 × 720**, tốc độ điều khiển mục tiêu là **60 FPS**. Terminal hiển thị tốc độ xử lý trong hai giây gần nhất và độ trễ của ảnh dùng để ra quyết định; phần tổng kết có FPS trung bình. `--fps` đặt giới hạn tốc độ, không bảo đảm máy luôn đạt được mức đó.

Chromium gửi các khung hình JPEG và bot chỉ giữ ảnh mới nhất. Với trang tĩnh hoặc canvas nằm ngoài vùng hiển thị, bot chuyển sang chụp ảnh trực tiếp. Tọa độ được chuẩn hóa theo pixel CSS, kể cả khi Windows bật tỷ lệ hiển thị lớn hơn 100%.

Ảnh nhận diện được thu nhỏ còn tối đa 320 pixel chiều rộng, sau đó tọa độ được quy đổi về ảnh gốc. Hình dạng khe ống được theo dõi qua nhiều khung hình để hạn chế lỗi khi chữ số điểm che ống trên. Hiển thị gỡ lỗi và mã hóa JPEG/PNG chạy nền với lượng công việc chờ giới hạn, tránh làm chậm vòng điều khiển. Việc đo thời gian dùng đồng hồ có độ phân giải cao; thời điểm khung hình của Chromium giúp giữ đúng khoảng cách thời gian giữa các ảnh trên Windows.

Cửa sổ gỡ lỗi mặc định làm mới ở 30 FPS. Để tăng tốc độ hiển thị:

```powershell
python -m flappy_auto --debug --debug-fps 60 --seconds 60
```

Để giảm tốc độ chụp mục tiêu hoặc đổi kích thước vùng hiển thị:

```powershell
python -m flappy_auto --debug --seconds 60 --fps 30 --viewport 480 720
```

Để chạy ẩn trình duyệt và không mở cửa sổ gỡ lỗi:

```powershell
python -m flappy_auto --headless --seconds 60
```

Các trình tải quảng cáo bên thứ ba đã biết trên `flappybird.io` bị chặn mặc định để giảm tải mạng và kết xuất. Dùng `--allow-ads` nếu muốn tắt bộ lọc. Menu, quảng cáo, màn hình kết thúc hoặc màu nhân vật khác vẫn có thể gây nhận diện sai. Tăng FPS không đồng nghĩa với việc chắc chắn đạt điểm cao hơn.

### Dùng Chrome đã cài trên máy

Nếu chưa tải Chromium bằng Playwright, bạn có thể chỉ định đường dẫn Chrome trên máy:

```powershell
python -m flappy_auto --demo --headless --browser-executable "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

### Chụp ảnh để kiểm tra nhận diện

```powershell
python -m flappy_auto --capture-only artifacts\site.png
```

Nếu không tìm thấy canvas, hãy chụp toàn bộ vùng hiển thị, xem ảnh rồi chọn vùng cắt bằng `--roi X Y W H`, lần lượt là tọa độ X, Y, chiều rộng và chiều cao:

```powershell
python -m flappy_auto --capture-only artifacts\full.png
python -m flappy_auto --roi 425 100 360 600 --seconds 40 --debug
```

`425 100 360 600` **chỉ là ví dụ minh họa**, không phải vị trí đã xác nhận của game trên website. Nếu tìm thấy canvas, tọa độ ROI được tính từ góc trên bên trái ảnh canvas; nếu không, tọa độ được tính từ ảnh toàn bộ vùng hiển thị.

Để lưu ảnh gỡ lỗi khoảng mỗi giây một lần, kèm dữ liệu chẩn đoán:

```powershell
python -m flappy_auto --demo --save-debug artifacts\debug --seconds 15
```

Để chỉ quan sát nhận diện, không gửi phím điều khiển; bạn cần tự bắt đầu game:

```powershell
python -m flappy_auto --demo --dry-run --debug
```

Xem toàn bộ tùy chọn:

```powershell
python -m flappy_auto --help
```

## Ghi phiên, phát lại và benchmark

Ghi toàn bộ ảnh đầu vào để tìm lỗi nhận diện mà không phải chơi lại website:

```powershell
python -m flappy_auto --demo --headless --seconds 30 --record-full artifacts\recordings
# Hoặc ghi website với profile đã hiệu chỉnh:
python -m flappy_auto --profile profiles\site.json --seconds 30 --record-full artifacts\recordings
```

Terminal in đường dẫn thư mục `session_*` mới cho mỗi lần mở. `--record-full` lưu ảnh PNG không mất dữ liệu, timestamp chụp/xử lý gốc, observation, lệnh đã gửi, thời gian quyết định, cấu hình/profile đang dùng, phiên bản Python/OpenCV/NumPy/Chromium và mã nguồn Git/SHA256. `session.json` có phiên bản định dạng; `events.jsonl` ghi từng frame/lệnh; `telemetry.csv` và `results.json` giữ thống kê. Không dùng chung `--record-full` với `--save-debug`, `--capture-only` hoặc `--calibrate`.

Ghi ảnh và telemetry chạy nền. Khi ổ đĩa chậm, bộ ghi giữ tối đa một ảnh thô chờ, một overlay chờ và 512 sự kiện gọn; ảnh bị thay thế ghi `dropped_backpressure`, số sự kiện/lệnh mất có trong `recording_stats`. Bot tiếp tục điều khiển. `complete_images` chỉ đúng nếu toàn bộ frame được lưu; ghi đầy đủ có thể tốn nhiều dung lượng. Chế độ `--save-debug` vẫn giữ bộ đệm nhẹ: tối đa 20 JPEG ở khoảng 10 FPS trước mỗi lần kết thúc/dừng, kèm overlay mỗi giây, thay vì lưu toàn bộ ảnh.

Thay `session_<tên>` bên dưới bằng thư mục thật vừa được in:

```powershell
python -m flappy_auto.replay artifacts\recordings\session_<tên> --step
# Không mở cửa sổ; xuất CSV, JSON và báo cáo Markdown:
python -m flappy_auto.replay artifacts\recordings\session_<tên> --headless --output artifacts\replays
# Thử ngưỡng màu khác hoặc detector từng ảnh, xuất ảnh có vòng nhận diện:
python -m flappy_auto.replay artifacts\recordings\session_<tên> --profile profiles\site.json --detector raw --export-images
```

Trong cửa sổ replay: **Space** tạm dừng/tiếp tục, **N** hoặc **mũi tên phải** tiến một frame và tạm dừng, **Q/Esc** thoát. `--speed 2` phát nhanh gấp đôi; `--step` mở ở trạng thái tạm dừng. Replay mặc định dùng tracker và cấu hình gốc; `--profile` chỉ thay ngưỡng nhận diện vì ảnh đã được cắt theo ROI lúc ghi.

Mỗi lần phát lại tạo thư mục báo cáo mới, gồm `frames.csv`, `report.json`, `report.md` và `overlays/` nếu yêu cầu. Cùng ảnh/cấu hình/mã nguồn/thư viện sẽ cho cùng SHA256 chuỗi observation. `all_recorded_observations_match` chỉ có ý nghĩa đối chiếu chính xác khi phiên đã đóng, đủ PNG, giữ nguyên mã nguồn/thư viện/cấu hình, không sửa timestamp và không bỏ frame. JPEG của bộ đệm nhẹ hoặc phiên thiếu ảnh vẫn phân tích được, nhưng trạng thái tracker có thể khác lần chơi gốc. Ảnh thiếu/hỏng mặc định báo lỗi; `--skip-missing` cho phép bỏ qua có thống kê. Timestamp thiếu/không hợp lệ mặc định báo lỗi; `--repair-timestamps --fallback-fps 30` sửa có ghi nhận và tắt đối chiếu chính xác. Timestamp cảm biến lặp/lùi hợp lệ vẫn được giữ để kiểm tra ảnh cũ. Phiên chẩn đoán cũ chưa có `session.json` không thuộc định dạng replay này.

**Replay chỉ đo nhận diện và ngữ cảnh lệnh đã gửi.** Thay detector/profile trên clip không tạo điểm hay tỷ lệ sống của bộ điều khiển mới. FPS replay là tốc độ tính nhận diện offline. Để so sánh bộ điều khiển có hành động ảnh hưởng lại thế giới, dùng benchmark mô phỏng:

```powershell
python -m flappy_auto.benchmark --config examples\benchmark.json --output artifacts\benchmarks
# Kiểm tra nhanh, giữ cả baseline và hai chế độ tiến trình:
python -m flappy_auto.benchmark --seeds 1 5 --runs 2 --seconds 12 --fps 30
```

Benchmark so sánh baseline `reactive` (nhảy theo độ cao/tốc độ) với `predictive` (`Strategy` hiện tại). Mỗi cặp seed/lần thử có cùng trạng thái đầu, khe ống, vật lý, nhiễu quan sát, độ trễ ảnh/phím/tính toán và ngân sách. Vật lý tích phân theo thời gian mô phỏng; thời gian CPU thực được đo riêng, không thay đổi kết quả vật lý. Đây là quan sát hình học có nhiễu trong môi trường độc lập, **không đo điểm website hoặc chất lượng OpenCV**. Chỉnh mô hình trong [cấu hình mẫu](examples/benchmark.json); file cấu hình đầy đủ được lưu lại trong mỗi báo cáo để chạy lại bằng một lệnh.

`cold` mở tiến trình Python mới cho từng ván; `warm` chạy nhiều ván cùng seed/bộ điều khiển trong một tiến trình, gọi `new_run()` để giữ thông số đã học. `process_id`, `seed`, `scenario_seed`, `trial` và SHA256 quỹ đạo cho phép kiểm tra các nhóm. Tất cả lần thử, kể cả lỗi, quá hạn worker, dừng và chưa chạy, đều được xuất trong `trials.csv`/`results.json`; không chỉ chọn ván tốt nhất. `--worker-timeout` giới hạn thời gian thực của mỗi tiến trình (mặc định 300 giây); Ctrl+C vẫn xuất phần kết quả đã nhận và các lần thử chưa chạy.

| Trạng thái ván | Ý nghĩa thống kê |
| --- | --- |
| `completed` | Va chạm thật trong mô phỏng; CLI chơi thật chỉ xác nhận khi thấy màn hình kết thúc. Có `final_score`. |
| `time_limit` | Còn sống khi hết ngân sách; `score_so_far` là điểm quan sát, `final_score` để trống. |
| `stopped` / `interrupted` | Dừng thủ công hoặc mất nhận diện/video đứng trong CLI; chưa biết điểm cuối. |
| `error` / `not_run` | Lỗi hoặc lần thử chưa khởi động; vẫn giữ trong bảng. |

`results.json` tách trung bình/trung vị/thấp nhất **điểm hoàn thành** khỏi điểm quan sát có giới hạn thời gian. Báo cáo ghi số mẫu, thời gian sống có giới hạn, FPS xử lý, tuổi ảnh, độ trễ cấu hình, thời gian quyết định trung bình/p95 và tỷ lệ **vượt** 20/50 (`>`). Tỷ lệ cận dưới đã quan sát dùng mọi lần thử đã khởi động, kèm số chưa biết; ván chưa hết giờ không mặc nhiên được tính là thất bại. `comparisons.csv` ghép từng cặp với baseline; chỉ tính `final_score_delta` khi cả hai ván hoàn thành. Chạy lâu hơn nếu muốn đánh giá ngưỡng điểm cao; lượt chạy ngắn hết giờ không chứng minh điểm tối đa.

Xem chi tiết [định dạng và giới hạn tái lập](docs/replay-benchmark.md). Toàn bộ ảnh/báo cáo trong `artifacts/` được Git bỏ qua.

## Môi trường Gymnasium cho RL

Môi trường `FlappyEnv` mô phỏng game bằng Python, nhận hai hành động thả rơi/nhảy và trả vector năm đặc trưng. Nó chạy độc lập với bot trình duyệt. Cài extra `rl` để chạy thử với agent ngẫu nhiên:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,rl]"
.\.venv\Scripts\python.exe scripts\rl_smoke.py
.\.venv\Scripts\python.exe -m pytest tests/test_rl_environment.py -v
```

Đã có seed, reward và phân biệt va chạm (`terminated`) với hết giới hạn bước (`truncated`). Smoke chỉ dùng agent ngẫu nhiên; không cần PyTorch hoặc Stable-Baselines3. Xem [hướng dẫn Gymnasium](docs/gymnasium.md) để biết API và các giới hạn mô phỏng. Nếu chỉ chạy bot, có thể cài `pip install -e .`; các test môi trường cần `[dev,rl]`.

### Huấn luyện DQN

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.3,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,rl,dqn]"
.\.venv\Scripts\python.exe scripts\train_dqn.py --config examples\dqn.json
```

Lệnh tạo thư mục `artifacts/dqn/dqn_*`, huấn luyện 200.000 bước và lưu checkpoint `.zip` kèm `.json`. Dùng `play_dqn.py` để nạp mô hình chơi thử, `evaluate_dqn.py` để so sánh với random/reactive trên seed test chưa dùng huấn luyện hoặc chọn checkpoint:

```powershell
# Thay dqn_<tên> bằng thư mục thật được in từ lệnh train.
.\.venv\Scripts\python.exe scripts\play_dqn.py artifacts\dqn\dqn_<tên>\best.zip --seed 2000
.\.venv\Scripts\python.exe scripts\evaluate_dqn.py artifacts\dqn\dqn_<tên>\best.zip
```

Training, validation và test có seed riêng; chỉ validation được chọn `best`. Báo cáo CSV/JSON tách điểm cuối ván va chạm khỏi điểm khi hết thời gian. DQN hiện chơi simulator với năm đặc trưng hình học, chưa thay bộ điều khiển website. Xem [hướng dẫn DQN](docs/dqn.md) cho cấu hình, lệnh smoke nhanh, định dạng checkpoint và cách đánh giá. Toàn bộ test, gồm huấn luyện PyTorch thật, cần `[dev,rl,dqn]`; CI cài đủ và chạy cả smoke train.

## Cấu trúc dự án

```text
FlappyBird-AutoPlayer/
├── src/flappy_auto/
│   ├── __main__.py      # Dòng lệnh và vòng lặp điều khiển
│   ├── browser.py       # Chụp ảnh, gửi phím, chọn canvas/iframe
│   ├── vision.py        # Nhận diện, theo dõi chim và ống bằng OpenCV
│   ├── profiles.py      # Profile JSON có phiên bản và kiểm tra cấu hình
│   ├── calibration.py   # Giao diện chọn vùng game, lấy mẫu và xem mặt nạ
│   ├── strategy.py      # Ước lượng vật lý và lập kế hoạch nhảy
│   ├── session.py       # Theo dõi ván chơi, đếm ống và nhận diện giao diện
│   ├── recording.py     # Ghi phiên nền và bộ đệm ảnh trước khi chết
│   ├── session_format.py # Định dạng phiên và kiểm tra timestamp/đường dẫn
│   ├── replay.py        # Phát lại nhận diện, overlay và báo cáo
│   ├── simulation.py    # Mô phỏng có phản hồi, seed và độ trễ cố định
│   ├── benchmark.py     # So sánh baseline, tiến trình cold/warm, CSV/JSON
│   ├── provenance.py    # Phiên bản mã nguồn và thư viện
│   ├── rl/              # Môi trường Gymnasium độc lập cho RL
│   ├── workers.py       # Hiển thị và xử lý ảnh nền
│   └── demo_server.py   # Máy chủ HTTP phục vụ bản demo trên máy
├── demo/
│   ├── index.html       # Trang canvas của bản demo
│   └── game.js          # Game đơn giản tương tự Flappy Bird, tự xây dựng
├── tests/               # Bộ kiểm thử bằng Pytest
├── scripts/rl_smoke.py  # Chạy một episode với agent ngẫu nhiên
├── scripts/train_dqn.py / play_dqn.py / evaluate_dqn.py
├── docs/architecture.md
├── .github/workflows/ci.yml
├── examples/default-profile.json
├── run_demo.cmd / run_site.cmd / run_calibrate.cmd
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Cách bot hoạt động

```text
Luồng khung hình Chromium → ảnh BGR mới nhất → phân tách màu HSV
                         → tìm chim và các khe ống, theo dõi qua nhiều ảnh
                         → ước lượng vật lý, tốc độ ống và độ trễ phím
                         → lập kế hoạch nhảy / thả rơi, kiểm tra an toàn
                         → gửi Space bằng Playwright → lặp lại
```

**Giới hạn hiện tại:** nhận diện dựa vào các màu cụ thể nên có thể nhầm nền hoặc trang phục nhân vật. Bộ điều khiển dùng mô hình vật lý đơn giản có khả năng tự hiệu chỉnh; kết quả thay đổi theo bố trí khe ống, độ trễ ảnh và tải kết xuất. Tự chơi lại dựa vào dòng chữ kết thúc màu đỏ và nút PLAY màu xanh của website; bản demo có thêm cách phát hiện game ngừng chuyển động hoặc mất dấu chim. Website khác có thể cần cách chơi lại riêng.

Số ống là ước lượng từ hình ảnh; dự án chưa đọc điểm bằng OCR, chưa dùng học tăng cường và không truy cập trực tiếp trạng thái nội bộ của game. Xem [tài liệu kiến trúc](docs/architecture.md) để biết cách hiệu chỉnh chi tiết.

## Đóng góp và đẩy thay đổi lên GitHub

Kho mã đã có nhánh `main`. Hãy clone kho để giữ nguyên lịch sử thay vì tạo lại kho Git từ đầu:

```powershell
cd E:\Project-OSS
git clone https://github.com/TieuBao-IT89/FlappyBird-AutoPlayer.git FlappyBird-AutoPlayer-GitHub
cd FlappyBird-AutoPlayer-GitHub
```

Nếu nhập mã từ tệp nén, chép `.github/`, `demo/`, `docs/`, `examples/`, `src/`, `tests/`, `.gitignore`, `LICENSE`, `pyproject.toml`, `README.md`, `requirements.txt`, `run_demo.cmd`, `run_site.cmd` và `run_calibrate.cmd` vào thư mục gốc của bản clone. Không chép thêm các thư mục bao ngoài của tệp nén vào cấu trúc dự án.

Không đưa `.venv/`, `venv/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info/`, `artifacts/`, `profiles/`, `.env`, `dist/` và `build/` vào Git. Các mục này đã nằm trong `.gitignore`; không dùng tùy chọn ép thêm chúng.

Kiểm tra các thay đổi và tệp đã đưa vào vùng chờ trước khi commit:

```powershell
git status --short
git add .
git diff --cached --stat
git status --short
git commit -m "docs: update project documentation"
git push -u origin main
```


Một số hướng phát triển tiếp: tối ưu bộ điều khiển bằng benchmark, hỗ trợ nhiều giao diện game, đọc điểm từ ảnh bằng OCR, xây dựng giao diện quản lý và bổ sung kiểm thử trình duyệt vào CI.

## Sử dụng có trách nhiệm

Dự án phục vụ học tập và kiểm thử trong môi trường kiểm soát được. Nên ưu tiên bản demo ngoại tuyến. Hãy xem quy định của website trước khi tự động hóa, tránh gửi yêu cầu quá mức và **không gửi điểm tự động lên bảng xếp hạng công khai**. Dự án độc lập với flappybird.io và không phân phối hình ảnh nhân vật của website.

## Giấy phép

Phát hành theo giấy phép **MIT** — xem [LICENSE](LICENSE). Mã nguồn và tài nguyên của bản demo trong kho được tự xây dựng.
