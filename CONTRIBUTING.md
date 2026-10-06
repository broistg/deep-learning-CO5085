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
* `fix`: Sửa lỗi thuật toán hoặc code (ví dụ: `fix(e3): fix tensor dimension in row-by-row image sequence`)
* `docs`: Thêm hoặc chỉnh sửa báo cáo, tài liệu trên GitHub Pages (ví dụ: `docs(a1): update fine-tuning accuracy comparison table`)
* `refactor`: Tái cấu trúc code mà không đổi logic (ví dụ: `refactor(utils): modularize data loading pipeline`)
* `exp`: Lưu notebook hoặc kết quả thử nghiệm thực tế (ví dụ: `exp(e3): log gru vs lstm convergence plots`)

---

## 3. Cấu trúc thư mục & Quản lý Dữ liệu lớn (RẤT QUAN TRỌNG)

1. **Cấu trúc thư mục nộp bài chuẩn:**
   * Mã nguồn bài tập và đồ án phải tổ chức theo đúng thư mục: `E1/`, `E2/`, `E3/`, `A1/`, `A2/`, mỗi thư mục kèm `README.md` hướng dẫn môi trường và lệnh chạy.
   * File báo cáo PDF được lưu trữ tập trung tại thư mục `reports/`.
2. **Tuyệt đối KHÔNG commit tập dữ liệu (`.jpg`, `.png`, `.zip`, `.tar.gz`)** hoặc trọng số mô hình lớn (`.pt`, `.pth`, `.onnx`, `.ckpt`) vào Git repository.
3. Tất cả thư mục dữ liệu phải đặt tên là `data/` và checkpoint đặt tên là `checkpoints/` (đã được cấu hình trong `.gitignore`).
4. Dùng Google Drive, HuggingFace Hub hoặc Kaggle Datasets để lưu trữ checkpoint/dữ liệu và dẫn link tải trong phần báo cáo/notebook.
5. Xóa kết quả output quá nặng trước khi commit các file `.ipynb` (Clear outputs), chỉ giữ lại bảng thống kê và biểu đồ nhỏ cần thiết.

---

## 4. Quy tắc Viết Code & Thực nghiệm (Coding & Experiment Standards)

* **Tự viết vòng lặp huấn luyện PyTorch (Bắt buộc theo specs):** Viết đầy đủ `forward → loss → backward() → optimizer.step()`. Không chỉ gọi các API cấp cao như `trainer.fit()` hay thư viện auto-training.
* **Dùng chung tập ảnh nhỏ cho E1-E3:** Thống nhất dùng cùng một tập ảnh nhỏ (MNIST, Fashion-MNIST hoặc CIFAR-10) xuyên suốt E1, E2, E3 để đảm bảo tính đối chứng.
* **Cố định Random Seed:** Mọi script/notebook phải có hàm cố định seed (`torch.manual_seed(42)`, `np.random.seed(42)`) ở đầu file để bảo đảm tính tái lập (Reproducibility).
* **Quản lý thiết bị linh hoạt:** Luôn dùng `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
* **Logging rõ ràng:** Bắt buộc lưu lịch sử huấn luyện (loss, accuracy qua từng epoch) ra file `.json` hoặc `.csv` để phục vụ vẽ biểu đồ so sánh trên web báo cáo.

---

## 5. Quy trình Nộp bài, Báo cáo PDF & GitHub Pages

1. **Báo cáo PDF bắt buộc:**
   * **E1-E3:** Bắt buộc dùng **một file PDF báo cáo chung duy nhất** tại `reports/exercises.pdf`. Đến hạn W9 bổ sung E1 và E2; đến hạn W10 bổ sung E3. Không tách file PDF riêng lẻ.
   * **A1 & A2:** Mỗi bài có 1 báo cáo PDF riêng đặt tại `reports/assignment1.pdf` và `reports/assignment2.pdf`.
2. **Cập nhật Website (GitHub Pages):**
   * Mỗi khi hoàn thành code của một phần, thành viên phụ trách phải đồng thời cập nhật file báo cáo HTML tương ứng trong `docs/` và đảm bảo liên kết tải PDF và link source code hoạt động chính xác.
   * Mọi biểu đồ và ảnh trực quan hóa phải được lưu vào thư mục `docs/assets/` với độ phân giải rõ nét và dung lượng tối ưu.
3. **Thời hạn (Deadline) & Quy định nộp trễ:**
   * Mọi mốc nộp bài có hạn chót lúc **23:59 GMT+7** (chấm theo commit trên nhánh `main`).
   * **Nộp trễ:** Trừ **−20%** điểm của mốc đó mỗi tuần, nộp trễ tối đa 7 ngày.
4. **Liêm chính học thuật & Quy định sử dụng AI:**
   * Tuân thủ quy định học thuật của Nhà trường.
   * Công cụ AI chỉ được dùng để hỗ trợ (tìm tài liệu, gợi ý, sửa lỗi cú pháp/ngữ pháp) và phải ghi rõ trong báo cáo. Không nộp sản phẩm do AI tạo phần lớn.
   * Mọi thành viên phải hiểu sâu và bảo vệ được toàn bộ mã nguồn cũng như số liệu thực nghiệm đã nộp.