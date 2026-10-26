# FlappyBird-AutoPlayer 🐤

**Bot thử chơi Flappy Bird bằng hình ảnh, không đọc biến game hoặc can thiệp mã JavaScript của trò chơi.**

Dự án sử dụng **Python, OpenCV và Playwright**, gồm bot tự chơi và một bản game ngoại tuyến để kiểm thử trên máy. Khả năng chơi trên website công khai vẫn mang tính **thử nghiệm**; bạn có thể cần hiệu chỉnh ngưỡng màu và vùng chụp khi giao diện trò chơi thay đổi.

## Tính năng chính

- Mở trình duyệt Chromium bằng Playwright.
- Tự chọn phần tử `<canvas>` hiển thị lớn nhất, kể cả trong iframe.
- Nhận luồng khung hình từ Chromium và chỉ xử lý ảnh mới nhất bằng OpenCV, tránh tồn đọng ảnh cũ.
- Nhận diện chim và cặp ống bằng mặt nạ màu HSV; mặc định dùng chim vàng/cam, ống xanh lá.
- Hiệu chỉnh trực quan bằng chuột: chọn vùng game, lấy mẫu màu, xem mặt nạ và lưu/nạp profile JSON.
- Ước lượng trọng lực, lực nhảy và tốc độ ống từ hình ảnh để lập kế hoạch nhấn **Space**.
- Chặn lệnh nhảy có quỹ đạo dự kiến đâm vào ống trên, kể cả khi chuyển từ khe cao xuống khe thấp.
- Tự ước lượng độ trễ phím và giữ các thông số đã hiệu chỉnh giữa những ván trong cùng phiên chạy.
- Nhận diện kết thúc ván, tự chơi lại và đếm số ống đã vượt qua từ hình ảnh.
- Tùy chọn lưu dữ liệu chẩn đoán và bộ đệm ảnh khoảng hai giây trước khi chết.
- Hỗ trợ cửa sổ gỡ lỗi, chụp ảnh, quan sát không điều khiển, bắt đầu thủ công và tùy chỉnh vùng chụp.
- Có **game demo ngoại tuyến**, bộ kiểm thử bằng Pytest, CI trên GitHub Actions và giấy phép MIT.

## Cài đặt nhanh trên Windows 11 bằng PowerShell

Cài Python 3.11 hoặc 3.12 và Git. Mở terminal của VS Code tại thư mục dự án, rồi chạy:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
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

`--calibrate` chỉ hiệu chỉnh và thoát, không tự chạy bot sau khi lưu. Nếu bỏ `--profile`, đường dẫn lưu gợi ý là `profiles/site.json` hoặc `profiles/demo.json`. Profile gồm phiên bản định dạng, chế độ website/demo, URL website, selector, viewport, ROI, kích thước ảnh tham chiếu theo pixel CSS, gốc tọa độ và ngưỡng màu. Hai gợi ý mỏ/mắt có thể đổi cả dải màu trong JSON. Chưa có cơ chế tìm lại chim hoặc đánh giá độ tin cậy; phần đó thuộc issue #2.

**Thứ tự ưu tiên:** tham số CLI được truyền rõ ràng → profile → mặc định hiện tại. `--url` chọn website, `--demo` chọn demo (nếu truyền cả hai thì demo được ưu tiên). `--selector`, `--viewport` và `--roi` ghi đè mục tương ứng. Dùng `--full-frame` để bỏ ROI đã lưu. Ngưỡng màu lấy từ profile. Không truyền profile vẫn giữ cách chạy và màu mặc định trước đây.

Khi ảnh game thay đổi kích thước hoặc gốc tọa độ khác profile, bot báo lỗi và yêu cầu hiệu chỉnh lại; thay viewport trên CLI không tắt kiểm tra này. Chạy lại `--calibrate --profile <tệp>` sẽ giữ ngưỡng màu, bỏ ROI cũ nếu bố cục khác để bạn chọn lại. Trong lúc chạy bot, kích thước phần tử được kiểm tra lại khoảng mỗi giây. Tọa độ được chuẩn hóa theo pixel CSS trước khi cắt ROI, kể cả khi Windows dùng DPI lớn hơn 100%. Chromium có thể bố trí phần tử hơi khác khi đổi sang `--headless`; nếu bị báo khác kích thước, chạy cùng chế độ có cửa sổ đã dùng khi hiệu chỉnh.

Thư mục `profiles/` và ảnh chẩn đoán trong `artifacts/` được Git bỏ qua. Khi lưu profile cá nhân, hãy dùng `profiles/`. [Cấu hình mẫu](examples/default-profile.json) giữ màu mặc định và chưa ràng buộc kích thước ảnh (`reference_size: null`, không có ROI); dùng cửa sổ hiệu chỉnh để lưu kích thước thật trước khi dùng lâu dài. Khi mở profile thuộc nguồn khác trong cửa sổ, hãy thoát và mở lại bằng `--calibrate --profile <tệp>` để trình duyệt tải đúng trang.

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
│   ├── recording.py     # Lưu dữ liệu chẩn đoán và ảnh trước khi chết
│   ├── workers.py       # Hiển thị và xử lý ảnh nền
│   └── demo_server.py   # Máy chủ HTTP phục vụ bản demo trên máy
├── demo/
│   ├── index.html       # Trang canvas của bản demo
│   └── game.js          # Game đơn giản tương tự Flappy Bird, tự xây dựng
├── tests/               # Bộ kiểm thử bằng Pytest
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


Một số hướng phát triển tiếp: đánh giá độ tin cậy và phục hồi nhận diện, hỗ trợ nhiều giao diện game, đọc điểm từ ảnh bằng OCR, so sánh các bộ điều khiển, xây dựng giao diện quản lý và bổ sung kiểm thử trình duyệt vào CI.

## Sử dụng có trách nhiệm

Dự án phục vụ học tập và kiểm thử trong môi trường kiểm soát được. Nên ưu tiên bản demo ngoại tuyến. Hãy xem quy định của website trước khi tự động hóa, tránh gửi yêu cầu quá mức và **không gửi điểm tự động lên bảng xếp hạng công khai**. Dự án độc lập với flappybird.io và không phân phối hình ảnh nhân vật của website.

## Giấy phép

Phát hành theo giấy phép **MIT** — xem [LICENSE](LICENSE). Mã nguồn và tài nguyên của bản demo trong kho được tự xây dựng.
