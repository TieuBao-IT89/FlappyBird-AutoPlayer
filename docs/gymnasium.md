# Môi trường Gymnasium cho Flappy Bird

`FlappyEnv` là môi trường Python độc lập, dùng làm nền cho agent RL. Nó chạy trong bộ nhớ, không mở Chromium hoặc website. Trang này hướng dẫn môi trường và agent ngẫu nhiên; phần huấn luyện và nạp mô hình nằm trong [hướng dẫn DQN](dqn.md).

## Cài đặt và chạy thử

Tại thư mục gốc dự án, dùng Python trong môi trường ảo:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,rl]"
.\.venv\Scripts\python.exe scripts\rl_smoke.py
.\.venv\Scripts\python.exe -m pytest tests/test_rl_environment.py -v
.\.venv\Scripts\python.exe -m pytest -q
```

Extra `rl` chỉ thêm `gymnasium>=1.1,<2`. Không cần PyTorch hoặc Stable-Baselines3. Nếu chỉ chạy bot trình duyệt, vẫn có thể cài `pip install -e .` như trước; Gymnasium chỉ được import khi dùng `flappy_auto.rl`. Để chạy toàn bộ test, cài cả `[dev,rl]`; CI cũng dùng hai extra này.

Smoke script seed cả môi trường lẫn bộ lấy mẫu hành động bằng `42`, chạy đến va chạm hoặc giới hạn bước rồi đóng môi trường. Kết quả có dạng:

```text
Episode complete: {'score': 0, 'steps': ...}; reward=...
```

Các số trên chỉ minh họa định dạng. Agent chọn ngẫu nhiên thường chết sớm; kết quả smoke không đo năng lực AI.

Kiểm tra API bằng công cụ chính thức của Gymnasium:

```powershell
.\.venv\Scripts\python.exe -c "from gymnasium.utils.env_checker import check_env; from flappy_auto.rl import FlappyEnv; check_env(FlappyEnv(), skip_render_check=True); print('Environment OK')"
```

Môi trường chưa có chế độ render, nên bỏ kiểm tra render. `check_env` vẫn kiểm tra spaces, kiểu dữ liệu, reset, step và tính lặp lại theo seed; kiểm tra này nằm trong unit tests. Tham khảo [API Env](https://gymnasium.farama.org/api/env/) và [Environment Checking](https://gymnasium.farama.org/api/utils/#environment-checking) của Farama.

## Sử dụng trong Python

```python
from flappy_auto.rl import FlappyEnv

env = FlappyEnv(max_steps=3600, randomize_gaps=True)
try:
    observation, info = env.reset(seed=42)
    env.action_space.seed(42)
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

Gọi `reset()` trước bước đầu tiên và sau mỗi episode kết thúc. `max_steps` phải là số nguyên dương, `randomize_gaps` phải là boolean. `reset(seed=...)` seed bộ sinh khe ống; muốn tái lập cả chuỗi hành động ngẫu nhiên thì seed riêng `action_space`. `options` được nhận theo API Gymnasium, hiện chưa có tùy chọn tùy biến reset.

## Hành động và quan sát

Action space là `Discrete(2)`: `0` thả rơi, `1` đặt vận tốc dọc thành lực nhảy hướng lên. Không có cooldown riêng; agent có thể chọn nhảy ở mỗi bước.

Observation là `Box(shape=(5,), dtype=float32)`, gồm:

| Vị trí | Đặc trưng | Miền |
| --- | --- | --- |
| 0 | Tọa độ Y của chim chia 600 | [0, 1], cắt biên |
| 1 | Vận tốc dọc chia 400; dương là đi xuống | [-2, 2], cắt biên |
| 2 | Khoảng cách từ chim tới **mép phải** ống đích, chia 360 | [-1, 3], cắt biên |
| 3 | Mép trên khe ống chia 600 | [0, 1] |
| 4 | Mép dưới khe ống chia 600 | [0, 1] |

Ống đích là ống đầu tiên còn ở phía trước hoặc chồng lên thân chim, đến khi mép phải đi qua mép trái thân chim. Observation trả hình học của ống đó, không đưa toàn bộ danh sách ống tương lai cho agent. Đây là đặc trưng trực tiếp từ simulator, chưa có pipeline OpenCV hoặc quan sát pixel.

## Vật lý, điểm và reward

Mỗi `step` mô phỏng 1/60 giây, không chờ theo đồng hồ thực. Hằng số xuất phát từ `demo/game.js`: màn 360 × 600, sàn Y = 528, chim X = 105/bán kính 12, trọng lực 740 px/s², vận tốc nhảy −260 px/s, ống rộng 52 di chuyển 100 px/s, khe cao 168 và khoảng cách ống 210 px.

Sau khi áp dụng hành động, cập nhật vận tốc rồi cập nhật vị trí bằng Euler bán ẩn như demo. Va chạm được kiểm tra ở cuối bước với bán kính chim và khe ống; đây là mô phỏng đơn giản ở bước cố định, chưa có kiểm tra va chạm liên tục giữa hai thời điểm. Danh sách ống giữ ba phần tử; khi ống đầu ra khỏi màn hình, tái sử dụng vị trí bằng cách sinh ống mới phía sau.

`randomize_gaps=True` chọn tâm khe trong [190, 360) bằng RNG của Gymnasium. `False` dùng `255 + sin(n × 1.7) × 85`, với `n` là chỉ số sinh ống bắt đầu từ 0. Chuỗi này dùng cùng công thức với demo; logic đánh số ống mới được đơn giản hóa, không phải bản sao chính xác mọi diễn biến JavaScript. Khi reset, chim ở Y = 270, vận tốc bằng 0, các ống ở X = 370, 580, 790.

| Sự kiện | Reward |
| --- | --- |
| Mỗi bước | +0.01 |
| Mép phải một ống đi qua tâm X của chim lần đầu | +1 |
| Va chạm | −1 |

Các thành phần cộng vào nhau: bước va chạm thông thường là −0.99; bước qua một ống là +1.01. `info` gồm `score` (số ống đã ghi điểm) và `steps`; mỗi ống chỉ ghi điểm một lần. Mốc ghi điểm qua tâm chim được giữ từ demo, sớm hơn lúc toàn thân chim thoát ống. Vì vậy ghi điểm và va chạm có thể xảy ra trong cùng một bước; không nên đồng nhất `score` với số ống đã thoát hoàn toàn an toàn.

Va chạm trả `terminated=True`. Đạt `max_steps` khi còn sống trả `truncated=True`; va chạm đúng bước giới hạn được ưu tiên là terminated. `score` khi truncated chỉ là điểm lúc bị cắt thời gian, chưa phải điểm cuối nếu chơi tiếp. Mặc định 3600 bước tương đương tối đa 60 giây mô phỏng.

## Quan hệ với bot và benchmark hiện tại

Đây là module riêng trong `flappy_auto/rl/`; `strategy.py`, `vision.py` và vòng điều khiển trình duyệt tiếp tục hoạt động như trước. Simulator benchmark trong `simulation.py` có cấu hình, nhiễu, độ trễ và kiểm tra va chạm riêng; không tự chuyển sang môi trường Gymnasium này. Điểm giữa hai môi trường chưa phải phép so sánh cùng điều kiện, và điểm RL không đại diện cho điểm website.

Pipeline DQN hiện có trong [hướng dẫn huấn luyện](dqn.md): dùng extra `dqn` để huấn luyện, lưu/nạp checkpoint và so sánh với baseline trên seed tách riêng. Lệnh `rl_smoke.py` của tài liệu này vẫn là agent ngẫu nhiên, không yêu cầu thư viện huấn luyện.
