# CO5085 - Deep Learning Coursework

Repository chính thức cho các bài tập (Exercises) và bài tập lớn (Assignments) môn học **Học sâu và ứng dụng trong thị giác máy tính (CO5085)**.

🌐 **Website báo cáo kết quả (GitHub Pages):** [https://broistg.github.io/deep-learning-CO5085/](https://broistg.github.io/deep-learning-CO5085/)

---

## 👥 Thành viên nhóm

| Họ và tên | MSSV | Vai trò |
| :--- | :---: | :--- |
| Lê Tiến Đạt | 2310653 | GitHub Pages, E2, E4, A1, A2 |
| Nguyễn Lê Thảo Ly | 2312010 | E1, E3, A1, A2 |

---

## 📌 Nội dung môn học

### 1. Exercises (Bài tập thực hành)
* **[E1: Small Image Classification](https://broistg.github.io/deep-learning-CO5085/exercises/e1.html):** Phân loại trên tập ảnh nhỏ (MNIST / Fashion-MNIST / CIFAR-10) với Softmax Classifier, MLP và CNN. So sánh accuracy, capacity, và phân tích dự đoán lỗi.
* **[E2: Multi-Head Self-Attention (MSA)](https://broistg.github.io/deep-learning-CO5085/exercises/e2.html):** Tự hiện thực lại Multi-Head Self-Attention (MSA) từ đầu, sử dụng PyTorch. Thử nghiệm các cơ chế tokenization (patch, pixel/row, CNN-stem, CLS + patches, ...) và phân loại trên cùng tập ảnh nhỏ.
* **[E3: Sequence Models (LSTM / GRU)](https://broistg.github.io/deep-learning-CO5085/exercises/e3.html):** Hiện thực và so sánh Sequence model (LSTM và GRU) với baseline Feed-Forward/CNN.
* **[E4: Generative Models](https://broistg.github.io/deep-learning-CO5085/exercises/e4.html):** Hiện thực các mô hình sinh (VAE, GAN và Diffusion model) trên tập ảnh nhỏ. Phân tích latent space và so sánh sự đánh đổi giữa chất lượng mẫu với năng lực tính toán.

### 2. Assignments (Bài tập lớn)
* **[A1: CNN vs. Transformer on Large Dataset](https://broistg.github.io/deep-learning-CO5085/assignments/a1.html):**
  * **A1.1:** Lựa chọn tập dữ liệu ảnh lớn & quy trình thực nghiệm (Protocol).
  * **A1.2:** So sánh CNN và Transformer dưới các chế độ huấn luyện: From-scratch vs Pretrained; Frozen backbone vs Partial freeze vs Full finetuning.
* **[A2: Computer Vision Pipeline & Paper Analysis](https://broistg.github.io/deep-learning-CO5085/assignments/a2.html):**
  * **A2.1:** Lựa chọn đề tài (detection, segmentation, ...) tập dữ liệu, và một bài báo liên quan.
  * **A2.2:** Xây dựng pipeline và báo cáo phân tích bài báo (phương pháp, thực nghiệm (có thể so sánh với thực nghiệm của nhóm), hạn chế).

---

## 🛠️ Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/broistg/deep-learning-CO5085.git
cd deep-learning-CO5085

# Khởi tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```
---

## 📖 Xem tài liệu nội bộ và quy định

Chi tiết quy chuẩn code, quy cách làm việc, cách commit và pull request: Xem tại [`CONTRIBUTING.md`](CONTRIBUTING.md).