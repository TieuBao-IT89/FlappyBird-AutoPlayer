# Kết quả mẫu issue #3

Các đợt khác: [DQN simulator 200k](dqn-200k/README.md), [DQN browser và rule](dqn-browser/README.md).

Đợt kiểm tra ngày 26/10/2026: 24 lượt mô phỏng, 3 seed × 2 ván × 2 bộ điều khiển × 2 chế độ. Công bố **toàn bộ 24 lượt** trong [issue3-trials.csv](issue3-trials.csv), gồm cả ván va chạm và hết giờ; không có lỗi hoặc lượt chưa chạy trong đợt này. Kiểm thử tự động riêng bao phủ lỗi I/O, worker lỗi/quá hạn/gián đoạn và dữ liệu thiếu.

Chạy lại từ thư mục gốc bằng cấu hình đúng của đợt này:

```powershell
python -m flappy_auto.benchmark --config examples/benchmark.json --output artifacts/benchmarks
```

| Bộ điều khiển | Chế độ | Va chạm / hết giờ | Điểm hoàn thành TB / trung vị / thấp nhất | Đã quan sát vượt 20 / tổng lượt |
| --- | --- | --- | --- | --- |
| predictive | cold | 0 / 6 | — / — / — (n=0) | 6 / 6 |
| predictive | warm | 0 / 6 | — / — / — (n=0) | 6 / 6 |
| reactive | cold | 3 / 3 | 20.67 / 22.00 / 18.00 (n=3) | 5 / 6 |
| reactive | warm | 3 / 3 | 20.67 / 22.00 / 18.00 (n=3) | 5 / 6 |

Mọi ván có ngân sách 40 giây mô phỏng. 18 ván hết giờ có `final_score` để trống; điểm quan sát của chúng không được đưa vào trung bình điểm cuối. Không có cặp predictive/reactive nào cùng hoàn thành, nên chưa có `final_score_delta` cho đợt này. Không ván nào đã vượt 50; 18 ván hết giờ còn chưa biết có vượt 50 nếu chạy tiếp hay không.

Đây là mô phỏng hình học có nhiễu, **không phải điểm website hoặc đánh giá OpenCV**. Không dùng kết quả này để khẳng định bot chơi website tốt hơn. Cấu hình, quy tắc thống kê và giới hạn nằm trong [hướng dẫn benchmark](../replay-benchmark.md).

Hai lần chạy cấu hình này cho cùng kết quả trạng thái, điểm, thời gian sống và SHA256 quỹ đạo ở cả 24 lượt. PID và thời gian CPU có thể khác. CSV là đợt chạy trên commit sau:

- Commit: `fcade311413fe8dc4a3a6038da5aa50d5c522b9e` (working tree sạch lúc bắt đầu).
- SHA256 package: `81f9d115b91dddb57900affcfc97c09fed53aa1c5db989c9f1f42de159724b85`.
- Python 3.11.9; NumPy 2.4.6; OpenCV 4.14.0; Windows-10-10.0.26200-SP0.

Kiểm tra browser/replay riêng trên cùng mã: ghi đủ 263/263 PNG từ demo Chromium trong 8 giây, không bỏ ảnh/sự kiện; hai lần replay khớp mọi observation gốc và cùng SHA256 `acb87c8382cdee82924638c7cca4f025040b48aa8872dacca53d579fc185b860`. Ảnh phiên lớn nằm trong artifacts đã bỏ qua Git. Cửa sổ replay đã được kiểm tra với tiến từng frame, tiếp tục và thoát.
