# Ghi chú gửi Ly: đưa notebook E1 vào repo (26/10/2026)

Notebook E1 của bạn đáp ứng đủ yêu cầu của đề (3 mô hình, vòng lặp huấn luyện tự viết, đường cong, confusion matrix, mẫu lỗi, biểu đồ chi phí). Mình chuyển nó vào repo theo cấu trúc chung của E1–E3. **Code của bạn được giữ nguyên**: kiến trúc 3 mô hình, `train_one_epoch`, `evaluate_one_epoch`, `predict` và toàn bộ hàm vẽ chỉ được chuyển sang file module, không viết lại. Chỉ có các thay đổi dưới đây, mỗi thay đổi đều có lý do.

## Thay đổi và lý do

1. **Chia train/val bằng `common/split.py` thay cho `random_split`.** Chia phân tầng 45.000/5.000 (500 ảnh mỗi lớp), seed 42, có mã băm SHA-256 để kiểm tra. *Lý do:* E1, E2, E3 phải dùng cùng một split (đề E3 yêu cầu so sánh với baseline E1 trên cùng split). `random_split` của bạn cho tập val khác, nên số của E1 và E2 không so sánh trực tiếp được. Tập test (10.000 ảnh chính thức) giữ nguyên.
2. **Mean/std tính từ 45.000 ảnh train của split chung** (hàm `common/cifar.py::channel_stats`), E2 cũng dùng hàm này. *Lý do:* chỉ dùng train để tính (không rò rỉ từ val/test) và cả nhóm dùng đúng một cách tính, đúng như mục 2.4 của báo cáo.
3. **Tăng cường dữ liệu: RandomCrop(32, padding=4) + lật ngang, chỉ cho tập train.** Áp dụng *sau* Normalize (đệm 0 = màu trung bình), giống hệt E2. Train, val, test là ba `Dataset` riêng. *Lý do:* nếu E1 không augment mà E2 có augment thì khoảng cách CNN–Transformer lẫn với ảnh hưởng của augmentation. Ngoài ra, trong notebook cũ train/val dùng chung một `transform` (gán lên `full_train_dataset`), nên nếu thêm augmentation vào đó thì val cũng bị augment. Tách ba Dataset tránh lỗi này. Có tùy chọn `--no-aug` nếu muốn chạy lại bản không augment làm đối chứng.
4. **Quy tắc báo cáo: số chính là test accuracy ở epoch cuối** (không chọn checkpoint). Checkpoint có val loss thấp nhất (cách của bạn) vẫn được lưu và báo cáo thêm trong cột "best". *Lý do:* E2 và A2 cũng báo cáo epoch cuối, nên không có bước chọn lọc nào tác động lên bảng tổng hợp.
5. **Seed và số lần chạy:** gọi `set_seed` trước *mỗi* mô hình (trước đây seed một lần ở đầu nên khởi tạo của MLP phụ thuộc việc Softmax chạy trước); chạy seed 0, 1, 2 và báo cáo trung bình ± độ lệch chuẩn; 30 epoch như E2. Giữ nguyên Adam, lr 1e-3, batch 64 của bạn.
6. **Cấu trúc file:** `E1/e1lib/{models,data,train,analysis,summary,run}.py`, notebook mới chỉ gọi module. Mỗi lần chạy ghi `E1/results/<model>_s<seed>.json` với cùng schema như E2 (cấu hình, mã băm split, history, final, best, tham số, thời gian, môi trường) và file dự đoán `.npz` để vẽ lại confusion matrix và mẫu lỗi không cần huấn luyện lại. Bỏ `device=device` toàn cục trong lớp mô hình (lớp nào cũng được `.to(device)` ngoài), và file `*_best.pt` không còn ghi vào thư mục hiện tại.
7. **Notebook gốc** để nguyên trong `E1/notebooks/archive/E1_original.ipynb` (có cả output của bạn).

## Việc của bạn
- Đọc `git diff` các file trong `E1/e1lib/` so với notebook gốc, nhất là `train.py` (đổi duy nhất: giữ trọng số epoch cuối và trả thêm checkpoint tốt nhất) và `data.py`. Bạn phải giải thích được từng thay đổi khi bảo vệ.
- Chạy: `python -m unittest discover -s E1/tests -v`, rồi `python -m e1lib.run --model mlp --seed 0 --fake --smoke`, rồi `python -m e1lib.run --all`, và gửi lại traceback nếu có lỗi. Mình chưa chạy được code torch ở môi trường của mình.
- Chạy `python -m common.cifar` trên máy bạn và máy Đạt: hai mã băm split phải trùng nhau.
- Kết quả cũ trong notebook (val 36,5% / 50,7% / 73,7%) dùng split khác, augmentation khác và 10 epoch nên **không dùng lại**; số mới sẽ thay thế.
