# CO5085 – Học sâu và ứng dụng trong thị giác máy tính

Repository chính thức chứa các bài tập E1–E3 và các bài tập lớn A1, A2: mã nguồn, kết quả, báo cáo và các trang web dùng để chấm.

🌐 **Website báo cáo kết quả (GitHub Pages):** [https://broistg.github.io/deep-learning-CO5085/](https://broistg.github.io/deep-learning-CO5085/)

---

## 👥 Thành viên nhóm

| Họ và tên | MSSV | Vai trò |
| :--- | :---: | :--- |
| Lê Tiến Đạt | 2310653 | GitHub Pages, E2, A1, A2 |
| Nguyễn Lê Thảo Ly | 2312010 | E1, E3, A2 |

---

## Tiến độ

| Phần | Nội dung | Trạng thái | Notebook | Báo cáo / trang web |
|---|---|---|---|---|
| [E1](E1/) | Softmax, MLP và CNN (MiniVGG) trên CIFAR-10 | hoàn thành (3 mô hình × 3 seed) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/deep-learning-CO5085/blob/main/E1/notebooks/E1.ipynb) | báo cáo bài tập, [trang E1](https://broistg.github.io/deep-learning-CO5085/exercises/e1.html) |
| [E2](E2/) | Multi-head self-attention (tự viết so với `nn.MultiheadAttention`) và bốn kiểu tokenization trên CIFAR-10 | hoàn thành (30 lần chạy) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/deep-learning-CO5085/blob/main/E2/notebooks/E2.ipynb) | báo cáo bài tập, [trang E2](https://broistg.github.io/deep-learning-CO5085/exercises/e2.html) |
| E3 | LSTM và GRU cho ảnh | **đang thực hiện** | sẽ bổ sung | sẽ bổ sung |
| [A1](A1/) | CNN và Transformer trên Food-101 (ResNet-50 so với DeiT-S, có và không tiền huấn luyện) | **đang thực hiện** (đã chốt quy trình và đo chi phí, các lần chạy chưa xong) | sẽ bổ sung | trang A1.1 và báo cáo bản nháp |
| A2 | SegFormer-B0 so với DeepLabV3+ (MobileNetV2) trên ADE20K | **đang thực hiện** (đã chốt quy trình, đã viết báo cáo mốc A2.1; mã nguồn và các lần chạy chưa có) | sẽ bổ sung | báo cáo A2.1; A2.2 chưa có |

E1–E3 dùng chung một quy trình: phép chia CIFAR-10 cố định và phân tầng (45.000 huấn luyện / 5.000 kiểm định / 10.000 kiểm thử, seed 42, mã băm SHA-256 trong `common/cifar10_split.json`), thống kê chuẩn hóa chỉ tính trên tập huấn luyện, augmentation (cắt ngẫu nhiên và lật ngang) chỉ trên tập huấn luyện, 30 epoch, seed 0, 1, 2 và độ chính xác kiểm thử ở epoch cuối. Vòng lặp huấn luyện được viết tay (không dùng lớp trainer có sẵn).

---

## Tái lập kết quả

Mỗi phần đã hoàn thành có một notebook mỏng (không chứa logic, chỉ gọi mã trong kho) với hai chế độ, chọn bằng biến `MODE` ở ô đầu tiên:

- `"load"` (mặc định): dùng kết quả đã lưu trong kho, sinh lại mọi bảng và hình, rồi kiểm tra chúng trùng với báo cáo. Mất vài phút, chạy được trên Colab miễn phí chỉ với CPU.
- `"rerun"`: huấn luyện lại toàn bộ từ đầu vào một thư mục `results/` mới (kết quả đã công bố được giữ lại để đối chiếu). Cần GPU, có thể dừng và chạy tiếp khi bị ngắt kết nối.

Trọng số và logit kiểm thử quá lớn để đưa vào git, nên được đính kèm trong GitHub Release `weights-v1`; mã băm SHA-256 nằm trong `common/release_manifest.json`, và notebook tự tải rồi kiểm tra khi cần (cách thiết lập: `common/README.md`).

### Chạy trên máy cá nhân

```bash
git clone https://github.com/broistg/deep-learning-CO5085.git
cd deep-learning-CO5085
# cài PyTorch + torchvision theo phiên bản CUDA của máy (không cố định trong repo), sau đó:
pip install -r requirements/e.txt
pip install -e .

python -m common.cifar                       # tải CIFAR-10, kiểm tra phép chia cố định (in mã băm)
python -m unittest discover -s E2/tests -v   # kiểm thử tính đúng đắn
python -m e2lib.plan                         # toàn bộ lần chạy E2 (có thể dừng và chạy tiếp)
python -m e2lib.analyze                      # bảng và hình
```

CIFAR-10 được tải từ Hugging Face Hub (`uoft-cs/cifar10`, qua thư viện `datasets`), với TorchVision là nguồn dự phòng. Dữ liệu chỉ được chấp nhận khi mã băm SHA-256 của nó trùng với bản TorchVision (bản Hugging Face lưu tập huấn luyện theo thứ tự khác; mã nguồn sắp xếp lại về thứ tự TorchVision mà phép chia cố định dựa vào).

---

## Cấu trúc thư mục

```
common/        mã dùng chung: nạp CIFAR-10 và phép chia cố định, checkpoint, tái lập, tiện ích cho notebook, đóng gói Release
E1/            e1lib (mô hình, vòng lặp huấn luyện, phân tích), kiểm thử, notebook, kết quả
E2/            e2lib (attention, tokenizer, huấn luyện, phân tích), kiểm thử, notebook, kết quả
A1/            a1lib (dữ liệu, mô hình, huấn luyện, kế hoạch chạy và worker), cấu hình, kiểm thử, kết quả
docs/          GitHub Pages: trang tổng quan và một trang cho mỗi bài tập
reports/       mã LaTeX của các báo cáo (bài tập E1–E3, A1, A2)
requirements/  danh sách thư viện (không cố định PyTorch vì Colab có bản CUDA riêng)
```

---

## 📖 Xem tài liệu nội bộ và quy định

Chi tiết quy chuẩn code, quy cách làm việc, cách commit và pull request: Xem tại [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Dữ liệu và giấy phép

- CIFAR-10: Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
- Food-101 (A1): Bossard và cộng sự, ECCV 2014. Ảnh thuộc về Foodspotting, không thuộc ETH Zürich, và giấy phép ghi trên thẻ Hugging Face là "unknown". Nhóm chỉ dùng dữ liệu cho môn học và không phân phối lại ảnh hay bộ nhớ đệm ảnh; kho chỉ chứa mã nguồn, cấu hình, kết quả và trọng số, không chứa ảnh.
- ADE20K (A2): MIT CSAIL, dùng bản SceneParsing cho môn học.

---

## Sử dụng công cụ AI

Nhóm lên ý tưởng và tự viết bản gốc của mã. Công cụ AI (ChatGPT / Claude) hỗ trợ cải thiện và rà soát mã, xây dựng bộ kiểm thử và bản nháp báo cáo. Nhóm đã đọc, hoàn thiện báo cáo, kiểm tra toàn bộ mã và nội dung, chịu trách nhiệm về chúng và có thể giải thích, bảo vệ chúng; chi tiết nằm trong phụ lục "Sử dụng công cụ AI" của từng báo cáo.