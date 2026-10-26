# DQN: đợt huấn luyện 200.000 bước

Ngày 26/10/2026, một đợt DQN CPU với seed khởi tạo 42, 200.000 bước và **49.500 cập nhật gradient**. Checkpoint được chọn ở **bước 140.000** bằng validation, trước khi đánh giá test. Thời gian train khoảng 413 giây trên máy phát triển; đây không phải cam kết tốc độ cho máy khác.

Training levels 0–31, validation 1000–1003, test 2000–2019; mỗi episode tối đa 3600 bước, tương đương 60 giây mô phỏng. Cấu hình đầy đủ ở [config.json](config.json), tất cả 44 lượt validation ở [validation.csv](validation.csv), tất cả 60 lượt test/baseline ở [trials.csv](trials.csv). Không lọc seed xấu hoặc ván hết giờ.

| Policy | Lượt test | Va chạm / hết giờ | Điểm quan sát TB | Điểm cuối TB (n) | Đã vượt 20 |
| --- | --- | --- | --- | --- | --- |
| dqn | 20 | 6 / 14 | 22.60 | 10.00 (n=6) | 14 / 20 |
| random | 20 | 20 / 0 | 0.00 | 0.00 (n=20) | 0 / 20 |
| reactive | 20 | 0 / 20 | 28.00 | — (n=0) | 20 / 20 |

DQN học tốt hơn agent ngẫu nhiên trong đợt này, nhưng **chưa vượt luật reactive cố định**. 14/20 ván DQN còn sống tại giới hạn 60 giây; các ván đó dừng ở 28 điểm quan sát, chưa biết điểm cuối. 22,6 là trung bình điểm quan sát gồm ván hết giờ, không phải trung bình điểm cuối ván. 6 ván DQN va chạm có điểm 11, 9, 3, 8, 13, 16; trung bình 10. Không ván nào đã vượt 50; 14 ván DQN và 20 ván reactive bị cắt thời gian còn chưa biết có vượt ngưỡng đó nếu chạy tiếp.

Mốc validation 140.000 đạt điểm quan sát trung bình 25,25; các mốc sau giảm (xem CSV). Vì vậy best.zip khác last.zip. Không dùng kết quả test để chọn lại checkpoint hoặc cấu hình. Đây là một seed huấn luyện trong simulator có đặc trưng hình học trực tiếp; chưa khẳng định ổn định qua nhiều lần train hoặc cải thiện điểm website.

## Tái lập

Từ thư mục gốc, sau khi cài extra dqn:

```powershell
python scripts/train_dqn.py --config docs/benchmarks/dqn-200k/config.json --output artifacts/dqn
# Dùng thư mục mới được in từ lệnh train:
python scripts/evaluate_dqn.py artifacts/dqn/dqn_<tên>/best.zip --output artifacts/dqn-evaluation
```

ZIP mô hình và log đầy đủ nằm trong artifacts đã bỏ qua Git. [checkpoint.json](checkpoint.json) giữ metadata/hash của mô hình được đánh giá. Thư viện, phần cứng hoặc phiên bản khác có thể làm quá trình train khác; không yêu cầu thời gian xử lý hoặc byte ZIP khớp.

- Source tương ứng: `1718aafbfd75d7f88e31ab7370ea0c0a856ad22f`; hash source RL: `37a5216c3ca6530b6d366edcfda9d17eee34869e63923ba1df3be95bed1f5bac`.
- Train chạy trước commit này nên provenance gốc giữ Git base/dirty; đã kiểm tra hash source RL khớp với commit tương ứng.
- Python 3.11.9, Gymnasium 1.4.0, SB3 2.9.0, PyTorch 2.14.1+cpu, NumPy 2.4.6.
- SHA256 checkpoint: `8395ab6a9effd3baf01736693ca0160a060911d9697292b532f623a28ddc89c7`.

Hướng dẫn chi tiết: [DQN](../../dqn.md).
