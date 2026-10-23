# Failure Analysis — Lab 18: Production RAG

**Học viên:** Nguyễn Hải Long · **MSSV:** 2A202602471
**Lượt đánh giá:** 20 câu, pipeline và RAGAS chạy live ngày 21/10/2026. Nguồn: `reports/ragas_report.json`. Lượt này chạy trước khi bổ sung hai OCR sidecar; các điểm số dưới đây chưa đo lại trên corpus đã mở rộng.

## Kết quả

| Metric | Naive baseline | Production | Chênh lệch |
|---|---:|---:|---:|
| Faithfulness | 0.8333 | 0.9042 | +0.0708 |
| Answer relevancy | 0.6637 | 0.8697 | +0.2061 |
| Context precision | 0.9222 | 0.9917 | +0.0694 |
| Context recall | 0.9211 | 0.9000 | -0.0211 |

Năm module đạt 38/38 test trong môi trường có đủ dependency ở lượt chạy trước. Faithfulness ≥0.85, cả bốn metric ≥0.75, enrichment combined mode một API call/chunk và bảng latency thỏa bốn điều kiện bonus công khai (+10 tối đa) **trên corpus của lượt đo đó**; điểm chính thức vẫn do giám khảo chấm. Sau lượt đo, hai PDF scan được nạp bằng OCR sidecar và kiểm thử loader liên quan đã pass. Cần chạy lại pipeline/RAGAS 20 câu khi Docker Qdrant sẵn sàng để xác nhận các ngưỡng bonus trên corpus mới.

## Bottom-5: chẩn đoán theo Error Tree

### 1. Hoàn trả khóa học 25 triệu sau 8 tháng

- **Kết quả:** 25.000.000 VNĐ, bằng 100% chi phí, vì nghỉ trước cam kết 1 năm. Nội dung khớp đáp án tham chiếu; RAGAS faithfulness vẫn chỉ 0.3333.
- **Error Tree:** Output đúng số và điều kiện → tài liệu `hoan_chi_dao_tao.md` có mốc 1 năm và tỷ lệ 100% → claim “8 tháng < 1 năm” và phép nhân 100% cần được đối chiếu với bước suy luận được đưa vào context → điểm thấp có thể đến từ cách judge tách claim hoặc context cụ thể, chưa đủ căn cứ gọi đây là hallucination.
- **Cách xử lý:** Pipeline đã thêm phép tính có điều kiện vào context từ dữ kiện truy xuất. Cần lưu trace claim/context từng câu để kiểm tra vì RAGAS vẫn chấm thấp; không sửa tay câu trả lời hoặc điểm.

### 2. Nghỉ không lương 20 ngày

- **Kết quả:** CEO phê duyệt, đúng vế được hỏi; faithfulness 0.5000. Đáp án tham chiếu còn nhắc nhân viên tự đóng phần bảo hiểm khi nghỉ trên 14 ngày.
- **Error Tree:** Output đúng thẩm quyền nhưng thiếu nghĩa vụ đi kèm → `nghi_phep_khong_luong.md` có cả ngưỡng 16–30 ngày và ngưỡng bảo hiểm >14 ngày → vấn đề ở khâu tổng hợp câu trả lời, không phải thiếu quy định trong corpus.
- **Cách xử lý:** Với câu hỏi về nghỉ không lương, prompt hoặc quy tắc trả lời nên yêu cầu nêu nghĩa vụ bảo hiểm liên quan nếu số ngày vượt ngưỡng. Không thêm nghĩa vụ không có trong nguồn.

### 3. Chu kỳ đổi mật khẩu

- **Kết quả:** 120 ngày theo v2.0, đúng chính sách hiện hành; context recall 0.5000. Ground truth còn giải thích bản v1 cũ là 90 ngày.
- **Error Tree:** Output đúng → retriever loại `mat_khau_v1.md` đã bị thay thế để tránh trả sai → evaluator kỳ vọng cả thông tin lịch sử trong ground truth → thiếu ngữ cảnh bản cũ làm giảm recall dù câu hỏi chỉ hỏi quy định hiện hành.
- **Cách xử lý:** Giữ ưu tiên v2.0 khi trả lời. Khi lập bộ eval, tách câu hỏi “hiện hành” khỏi câu hỏi “so sánh phiên bản”, hoặc đưa bản cũ vào context có nhãn hết hiệu lực mà không dùng để trả lời hiện hành.

### 4. Độ dài tối thiểu mật khẩu

- **Kết quả:** 12 ký tự theo v2.0, đúng hiện hành; context recall 0.5000. Ground truth nhắc thêm mốc 8 ký tự của v1 đã bị thay thế.
- **Error Tree:** Output đúng → metadata filter bỏ bản cũ → RAGAS so với ground truth gồm cả thông tin v1 → giảm recall do lệch phạm vi đánh giá, không phải trả sai 12 ký tự.
- **Cách xử lý:** Dùng cùng quy tắc phiên bản trong test set và retrieval; chỉ so sánh v1/v2 khi người dùng hỏi lịch sử thay đổi.

### 5. Ngày phép cộng theo thâm niên

- **Kết quả:** thêm một ngày sau mỗi 3 năm, đúng `nghi_phep_nam_v2024.md`; context recall 0.5000. Ground truth có thêm quy định cũ v2023 là 5 năm.
- **Error Tree:** Output đúng → retriever bỏ `nghi_phep_nam_v2023.md` để tránh xung đột → ground truth yêu cầu thông tin cũ → evaluator tính thiếu một vế lịch sử.
- **Cách xử lý:** Giữ câu trả lời hiện hành 3 năm; bổ sung bài eval riêng về so sánh chính sách 2023/2024 thay vì ép câu hỏi hiện hành phải truy xuất cả bản đã thay thế.

## Kiểm tra lỗi số học và latency

Ở lượt trước, model từng cộng mức phí tháng 300.000 VNĐ với phí ước tính 5 ngày 50.000 VNĐ thành 350.000 VNĐ. Đó là sai: với giả định tháng 30 ngày, phí phải trả cho 5 ngày quá hạn là khoảng **50.000 VNĐ**, còn 300.000 VNĐ chỉ là mức cho cả tháng. Pipeline hiện tính từ số tiền, mốc hạn và tỷ lệ trong tài liệu, rồi dùng câu trả lời xác định cho trường hợp này; test hồi quy `tests/test_pipeline_calculation.py` kiểm tra không cộng đôi. Lượt RAGAS cuối đã chạy lại toàn bộ sau sửa.

| Bước | Giây |
|---|---:|
| Chunking | 0.17 |
| Enrichment | 37.32 |
| Indexing BM25 + Dense | 56.39 |
| Tạo câu trả lời 20 câu | 524.51 |
| RAGAS | 38.95 |
| **Tổng** | **657.34** |

Nút thắt là retrieval/reranking và sinh câu trả lời trên CPU trong 20 câu, nhất là truy vấn nhiều vế. Bảng này là thời gian đo của lượt chạy cuối; thời gian sẽ thay đổi theo máy và dịch vụ API.
