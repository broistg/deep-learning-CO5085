# Team Working Rules & Collaboration Guidelines

Tài liệu này định hình quy tắc làm việc, phân luồng nhánh Git và chuẩn code nhằm đảm bảo tiến độ và tính nhất quán cho toàn bộ dự án.

---

## 1. Quy ước phân nhánh (Git Branching Model)

* **`main`**: Nhánh bảo vệ (Protected Branch). Tuyệt đối **không commit trực tiếp** lên nhánh này. Chỉ merge code thông qua Pull Request (PR) sau khi đã chạy kiểm tra và có ít nhất 1 thành viên approve.
* **`dev`**: Nhánh tích hợp các tính năng chuẩn bị nộp bài.
* **Nhánh tính năng (Feature branches):** Tạo nhánh mới từ `dev` theo cú pháp:
  * Bài tập: `feature/e1-softmax-mlp-cnn`, `feature/e2-msa-tokenization`
  * Đồ án: `feature/a1-data-preprocessing`, `feature/a2-model-pipeline`
  * Sửa lỗi / Cập nhật docs: `fix/e1-cuda-out-of-memory`, `docs/update-gh-pages`

---

## 2. Quy ước Commit Message

Tuân thủ chuẩn **Conventional Commits**: `<type>(<scope>): <mô tả ngắn bằng tiếng Anh hoặc tiếng Việt>`

* `feat`: Thêm mô hình, kiến trúc hoặc tính năng mới (ví dụ: `feat(e2): implement scratch multihead self-attention module`)
* `fix`: Sửa lỗi thuật toán hoặc code (ví dụ: `fix(e4): correct kl-divergence loss in vae`)
* `docs`: Thêm hoặc chỉnh sửa báo cáo, tài liệu trên GitHub Pages (ví dụ: `docs(a1): update fine-tuning accuracy comparison table`)
* `refactor`: Tái cấu trúc code mà không đổi logic (ví dụ: `refactor(utils): modularize data loading pipeline`)
* `exp`: Lưu notebook hoặc kết quả thử nghiệm thực tế (ví dụ: `exp(e3): log gru vs lstm convergence plots`)

---

## 3. Quản lý Dữ liệu lớn & Checkpoint (RẤT QUAN TRỌNG)

1. **Tuyệt đối KHÔNG commit tập dữ liệu (`.jpg`, `.png`, `.zip`, `.tar.gz`)** hoặc trọng số mô hình lớn (`.pt`, `.pth`, `.onnx`, `.ckpt`) vào Git repository.
2. Tất cả thư mục dữ liệu phải đặt tên là `data/` và checkpoint đặt tên là `checkpoints/` (đã được cấu hình trong `.gitignore`).
3. Dùng Google Drive, HuggingFace Hub hoặc Kaggle Datasets để lưu trữ checkpoint/dữ liệu và dẫn link tải trong phần báo cáo/notebook.
4. Xóa kết quả output quá nặng trước khi commit các file `.ipynb` (Clear outputs), chỉ giữ lại bảng thống kê và biểu đồ nhỏ cần thiết.

---

## 4. Quy tắc Viết Code & Thực nghiệm (Coding & Experiment Standards)

* **Cố định Random Seed:** Mọi script/notebook phải có hàm cố định seed (`torch.manual_seed(42)`, `np.random.seed(42)`) ở đầu file để bảo đảm tính tái lập (Reproducibility).
* **Quản lý thiết bị linh hoạt:** Luôn dùng `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
* **Logging rõ ràng:** Bắt buộc lưu lịch sử huấn luyện (loss, accuracy qua từng epoch) ra file `.json` hoặc `.csv` để phục vụ vẽ biểu đồ so sánh trên web báo cáo.

---

## 5. Quy trình Nộp bài & Cập nhật Website (GitHub Pages)

1. Mỗi khi hoàn thành code của một phần (ví dụ `E1`), thành viên phụ trách phải đồng thời cập nhật file báo cáo tương ứng trong thư mục `docs/`.
2. Mọi biểu đồ và ảnh trực quan hóa phải được lưu vào thư mục `docs/assets/<tên-bài>/` với độ phân giải rõ nét và dung lượng tối ưu.
3. Trước ngày nộp bài (deadline) ít nhất **24 giờ**, toàn bộ PR liên quan phải được merge vào `main` để kiểm tra hiển thị trên GitHub Pages trước khi nộp link cho giảng viên.