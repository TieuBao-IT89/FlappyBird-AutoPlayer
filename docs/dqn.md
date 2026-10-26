# Huấn luyện DQN với Stable-Baselines3

Đợt này thêm pipeline huấn luyện mạng PyTorch trên `FlappyEnv`, lưu/nạp checkpoint `.zip`, chơi thử một episode và đánh giá trên seed tách riêng. Môi trường trả năm đặc trưng hình học; bot website vẫn dùng OpenCV và `Strategy`. DQN hiện chơi simulator Python, chưa điều khiển website từ ảnh.

## Cài đặt trên Windows 11

Tại thư mục gốc của dự án, dùng Python 3.11 trong `.venv`. Bản CPU đủ chạy mạng MLP nhỏ; cài PyTorch CPU trước để không tải phụ thuộc CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.3,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,rl,dqn]"
.\.venv\Scripts\python.exe scripts\rl_smoke.py
.\.venv\Scripts\python.exe -m pytest -q
```

`rl` tiếp tục chỉ thêm Gymnasium. Extra `dqn` thêm Stable-Baselines3 và PyTorch; CLI bot không import các thư viện huấn luyện. Khi chưa cài `dqn`, các test huấn luyện thật được đánh dấu skip; CI cài đủ extra và chạy cả chúng. Không cần TensorBoard, GPU hoặc tài khoản dịch vụ bên ngoài.

## Huấn luyện

```powershell
.\.venv\Scripts\python.exe scripts\train_dqn.py --config examples\dqn.json --output artifacts\dqn
```

Đây là 200.000 bước mô phỏng, không phải 200.000 giây thực. Mặc định dùng CPU, một luồng PyTorch, `MlpPolicy` hai lớp 64 nút, replay buffer 50.000 mẫu, batch 64, learning rate 0.0005, gamma 0.99, bắt đầu cập nhật sau 2.000 bước, mỗi 4 bước cập nhật một lần và cập nhật target network mỗi 1.000 bước. Epsilon giảm từ 1 xuống 0.05 trong 35% đầu ngân sách. Các giá trị đầy đủ nằm trong [examples/dqn.json](../examples/dqn.json).

Terminal in thư mục `dqn_*` mới và tiến độ ở mỗi mốc validation 20.000 bước. Tốc độ phụ thuộc CPU. Để chỉ kiểm tra pipeline nhanh, chưa nhằm học cách chơi tốt:

```powershell
.\.venv\Scripts\python.exe scripts\train_dqn.py --timesteps 128 --learning-starts 16 --eval-freq 64 --output artifacts\dqn-smoke
```

`--timesteps` phải lớn hơn `--learning-starts`. SB3 thu thập theo nhóm `train_freq`; tổng bước thực tế có thể được làm tròn lên theo nhóm này, và luôn được ghi lại. Số bước không đồng nghĩa với số lần cập nhật gradient; xem `gradient_updates` trong metadata. Chạy ngắn có thể chưa cải thiện so với mạng khởi tạo.

## Seed và chọn checkpoint

Cấu hình mặc định có ba nhóm không giao nhau:

| Nhóm | Seed | Vai trò |
| --- | --- | --- |
| Training | 0–31 | Sinh episode để thu thập replay buffer và cập nhật mạng |
| Validation | 1000–1003 | Đánh giá định kỳ và chọn checkpoint |
| Test | 2000–2019 | Đánh giá cuối sau khi mô hình đã được cố định |

`seed=42` trong cấu hình là seed khởi tạo mạng, lấy mẫu replay và hành động khám phá. Nó tách khỏi seed cấp độ. Wrapper `SeedSchedule` xoay vòng các seed training khi SB3 tự reset: chỉ số bắt đầu là `seed % số_seed_training`, rồi tăng một sau mỗi episode. Mỗi lần vào một seed, RNG của game được reset về seed đó; cùng seed xác định cùng chuỗi khe ống, không phải một chuỗi level vô hạn mới sau mỗi lần chết.

Trước khi học và sau mỗi `eval_freq`, policy quyết định tất định được chạy trên **validation**. Chọn checkpoint theo trung bình `score_so_far`, nếu hòa thì theo trung bình reward. Ngân sách của các lượt validation giống nhau. Ván hết giờ được ghi đúng là time limit; tiêu chí chọn là điểm quan sát, không giả làm điểm cuối ván.

Seed test không được gọi trong huấn luyện hoặc chọn mô hình. CLI đánh giá từ chối seed trùng training **hoặc validation**. Khi so sánh nhiều cấu hình, tiếp tục chỉ dùng validation để chọn; giữ test cho lần đánh giá cuối. Một đợt chạy với một seed khởi tạo mạng chưa chứng minh tính ổn định qua nhiều lần huấn luyện độc lập.

## Các file đầu ra

Mỗi lần chạy tạo một thư mục mới:

| File | Nội dung |
| --- | --- |
| `config.json`, `code.json` | Cấu hình đầy đủ; Git, hash source RL, môi trường và phiên bản thư viện |
| `episodes.monitor.csv` | Reward, độ dài, điểm và seed của từng episode training hoàn thành |
| `validation.csv` | Từng episode validation ở từng mốc, gồm cả mạng khởi tạo ở bước 0 |
| `checkpoint_<bước>.zip` + `.json` | Checkpoint riêng ở từng mốc validation, không ghi đè ZIP cũ |
| `selection.json` | Mốc đang được chọn và chỉ số validation |
| `best.zip` + `best.json` | Bản sao checkpoint được chọn khi đợt chạy kết thúc hoặc bị ngắt |
| `last.zip` + `last.json` | Mạng ở thời điểm kết thúc hoặc bị ngắt, có thể khác best |
| `training.json` | Trạng thái completed/interrupted/error, số bước thực tế, số lần cập nhật và thời gian |

Mỗi checkpoint có thuật toán, cấu hình, seed, số bước, số lần cập nhật, SHA256 mô hình và mã nguồn environment. Khi load, pipeline kiểm tra metadata, hash, phiên bản schema và observation/action space. Giữ cả cặp `.zip`/`.json` khi sao chép mô hình. Chỉ nạp checkpoint từ nguồn bạn tin cậy; định dạng SB3 có dữ liệu Python serialize.

Nếu mọi mốc validation không cải thiện, `best` có thể vẫn là mạng bước 0; metadata thể hiện chính xác điều này. `last` vẫn lưu các trọng số đã học. Ctrl+C lưu `last`, sao chép `best` nếu đã chọn được và ghi trạng thái interrupted; CLI trả mã 130. Lỗi huấn luyện ghi error và giữ checkpoint phần đã có khi có thể ghi file. Các checkpoint dùng để chạy suy luận; đợt này chưa hỗ trợ tiếp tục chính xác trạng thái replay buffer/lịch học của một lần train đã ngắt.

`artifacts/`, `models/`, `checkpoints/` và các file `.zip` được Git bỏ qua. Không commit model hoặc replay buffer vào source; báo cáo CSV/JSON nhỏ có thể chia sẻ riêng.

## Nạp mô hình để chơi thử

Thay đường dẫn dưới đây bằng thư mục được in từ lệnh train:

```powershell
.\.venv\Scripts\python.exe scripts\play_dqn.py artifacts\dqn\dqn_<tên>\best.zip --seed 2000
# Xuất từng observation, hành động, reward và trạng thái để xem lại:
.\.venv\Scripts\python.exe scripts\play_dqn.py artifacts\dqn\dqn_<tên>\best.zip --seed 2000 --trace artifacts\dqn-play.csv
```

Lệnh chơi một episode trong bộ nhớ, không mở cửa sổ game; kết quả JSON in điểm, reward, số bước, thời gian sống mô phỏng và trạng thái. `--max-steps` đổi giới hạn bước. Trace CSV chỉ tạo file mới, báo lỗi nếu đường dẫn đã tồn tại. Chơi thử có thể dùng bất kỳ seed không âm để debug; kết quả đó không tự được coi là đánh giá trên seed chưa thấy.

## Đánh giá và so sánh baseline

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_dqn.py artifacts\dqn\dqn_<tên>\best.zip --output artifacts\dqn-evaluation
# Hoặc một tập seed mới, vẫn không trùng training/validation:
.\.venv\Scripts\python.exe scripts\evaluate_dqn.py artifacts\dqn\dqn_<tên>\best.zip --seeds 3000 3001 3002
```

Mặc định chạy toàn bộ seed test trong metadata, cùng ngân sách `max_steps` đã lưu, lần lượt với DQN, agent ngẫu nhiên có seed và luật reactive cố định dựa trên vị trí/vận tốc/khe ống. Các policy dùng cùng seed và vật lý, không đọc danh sách ống tương lai. DQN dùng `predict(..., deterministic=True)` để bỏ khám phá khi đánh giá.

`trials.csv` lưu từng lượt ngay sau khi hoàn tất; `report.json` và `report.md` ghi tổng hợp và metadata/hash checkpoint. Lượt lỗi vẫn có dòng và làm CLI trả mã lỗi. Khi Ctrl+C, giữ lượt đang dừng với trạng thái interrupted, các lượt chưa khởi động là not_run, xuất báo cáo rồi trả 130.

Phân biệt `completed` (va chạm thật) với `time_limit` (còn sống khi hết bước). `score_so_far` của ván hết giờ là cận dưới, `final_score` để trống. Báo cáo tách trung bình/trung vị/thấp nhất điểm cuối trên ván hoàn thành với điểm quan sát có giới hạn thời gian, kèm số mẫu. Tỷ lệ vượt 20/50 dùng dấu `>` và ghi số chưa biết; các lượt chưa chạy không vào mẫu số đã thử. Không suy điểm cuối từ ván bị cắt thời gian hoặc chỉ công bố seed có điểm cao nhất.

## Giới hạn và tài liệu gốc

Reward và va chạm giữ nguyên `FlappyEnv` của mốc trước. Điểm được ghi khi mép phải ống qua tâm chim, sớm hơn lúc toàn thân thoát ống; cùng một bước có thể vừa ghi điểm vừa va chạm. Mô hình hiện nhận vị trí/vận tốc trực tiếp, chưa qua pipeline ảnh có nhiễu hoặc độ trễ thật. Đạt điểm cao trong simulator không chứng minh điều khiển website tốt hơn. Bước sau cần thêm đánh giá nhiều seed huấn luyện và cầu nối quan sát/độ trễ trước khi đưa AI sang bot browser.

Tham khảo tài liệu chính thức: [DQN và lưu/nạp mô hình của SB3](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html), [callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html), [cài đặt PyTorch](https://pytorch.org/get-started/locally/).
