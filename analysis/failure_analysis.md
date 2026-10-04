# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Hải Long  
**Khóa:** K4 - Track 3A  
**MSSV:** 2A202602471  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:-------------:|:----------:|:--:|
| Faithfulness | 0.8333 | 0.7083 | -0.1250 |
| Answer Relevancy | 0.6637 | 0.6836 | +0.0199 |
| Context Precision | 0.9222 | 0.9028 | -0.0194 |
| Context Recall | 0.9211 | 0.8889 | -0.0322 |

---

## Bottom-5 Failures

### #1
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.00)
- **Error Tree:** Output sai (Không tìm thấy) → Context retrieved có chứa chunk chính sách mật khẩu v2.0? (Thiếu chunk v2.0 do reranker ưu tiên v1.0 hoặc bị cắt chunk) → Root cause ở Retrieval / Reranking.
- **Root cause:** Khi query ngắn và thiếu context phiên bản ("v2.0"), BM25 và Dense retrieve cả `mat_khau_v1.md` và `mat_khau_v2.md`. Cross-encoder reranker có thể đã lọc nhầm hoặc prompt LLM quá khắt khe khi thấy xung đột giữa 90 ngày và 120 ngày nên trả về "Không tìm thấy".
- **Suggested fix:** Thêm metadata filter ưu tiên tài liệu có version mới nhất (`version: 2.0`), hoặc bổ sung query rewriting để nhận diện "hiện hành / mới nhất".

### #2
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.00)
- **Error Tree:** Output sai → Context đúng? (Context chỉ có thông tin bảng lương hoặc chính sách thử việc riêng lẻ, thiếu phép tính suy luận đa tài liệu) → Multi-hop reasoning failure.
- **Root cause:** Câu hỏi yêu cầu thông tin từ 2 nguồn tài liệu khác nhau: `bang_luong_2024.md` (mức lương Junior tối đa 20tr) và `thu_viec.md` (lương thử việc = 85% lương chính thức). Pipeline retrieval chỉ lấy top-3 chunks đơn lẻ nên không gom đủ cả 2 chunks cùng lúc.
- **Suggested fix:** Tăng retrieval top-k trước khi rerank (top 30), hoặc triển khai Multi-hop / Agentic query decomposition tách thành 2 sub-queries: "Lương tối đa Junior?" và "Tỷ lệ lương thử việc?".

### #3
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên có 9 năm thâm niên được 18 ngày phép (15 + 3). Không tìm thấy thông tin về lương.
- **Worst metric:** Faithfulness (0.00)
- **Error Tree:** Output thiếu 1 vế (vế lương) → Context đúng? (Context chỉ retrieve được chunk về chính sách nghỉ phép `nghi_phep_nam_v2024.md`, không retrieve được `bang_luong_2024.md`) → Retrieval Recall bị hụt đối với multi-aspect query.
- **Root cause:** Câu hỏi kép kết hợp 2 chủ đề hoàn toàn độc lập ("ngày phép" và "mức lương"). Dense embedding và BM25 bị lệch trọng số về phía từ khóa "nghỉ phép thâm niên", đẩy các chunk về bảng lương ra khỏi top 20 candidate.
- **Suggested fix:** Áp dụng Sub-query decomposition hoặc Query splitting trong module tiền xử lý truy vấn: tách câu hỏi phức thành "Số ngày phép của Senior 9 năm thâm niên" và "Mức lương Senior".

### #4
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất. Vì đây là một chiếc laptop có giá 30 triệu, trưởng phòng sẽ phê duyệt đơn hàng.
- **Worst metric:** Faithfulness (0.00)
- **Error Tree:** Output sai thẩm quyền phê duyệt ("trưởng phòng" thay vì "Giám đốc phòng ban/Director") → Context đúng? (Context có bảng hạn mức mua sắm) → LLM ánh xạ nhầm mốc hạn mức thẩm quyền.
- **Root cause:** Tài liệu `mua_sam.md` quy định nhiều phân cấp thẩm quyền (dưới 5tr: Trưởng phòng; 5-50tr: Giám đốc bộ phận; trên 50tr: CEO). LLM đọc chunk nhưng bị nhầm lẫn ranh giới giữa Trưởng phòng và Giám đốc phòng ban khi tổng hợp.
- **Suggested fix:** Hạ temperature xuống 0.0, tinh chỉnh prompt yêu cầu LLM trích dẫn chính xác dòng quy định thẩm quyền kèm số tiền cụ thể trước khi đưa ra kết luận.

### #5
- **Question:** Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Expected:** Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng, chi trả cùng kỳ lương.
- **Got:** Phụ cấp ăn trưa hàng tháng là **1.000.000 VNĐ/tháng**.
- **Worst metric:** Faithfulness (0.00 theo đánh giá RAGAS do thiếu vế 'chi trả cùng kỳ lương')
- **Error Tree:** Output đúng nội dung cốt lõi nhưng thiếu chi tiết phụ → Context có đủ → Đánh giá RAGAS chấm điểm khắt khe về completeness.
- **Root cause:** Câu trả lời ngắn gọn, đúng số tiền 1.000.000 VNĐ nhưng LLM bỏ qua câu phụ "chi trả cùng kỳ lương", khiến metric RAGAS đối chiếu với ground truth bị lệch.
- **Suggested fix:** Bổ sung vào system prompt chỉ dẫn: "Cung cấp câu trả lời đầy đủ bao gồm số tiền, thời gian và phương thức chi trả nếu có trong context".

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Không. LLM trả về *"Không tìm thấy"* trong khi tài liệu có đầy đủ dữ liệu.
2. **Context đúng?** → Không. Context đưa vào LLM chỉ chứa chunk quy định chung về thời gian thử việc 60 ngày (`thu_viec.md`), hoàn toàn thiếu chunk bảng lương Junior 20 triệu (`bang_luong_2024.md`).
3. **Query rewrite OK?** → Truy vấn gốc là câu hỏi gộp (composite query) yêu cầu tính toán: `Mức lương thử việc = 85% × Lương chính thức Junior`. Retriever thông thường chỉ tìm theo semantic similarity tổng thể nên không gom được cả 2 văn bản.
4. **Fix ở bước:**  
   - Cần bổ sung bước **Query Decomposition** (tách truy vấn trước khi tìm kiếm) hoặc **Multi-Hop Retrieval**.
   - Thêm tính năng **Calculation/Tool use** cho LLM khi gặp các bài toán tính tỷ lệ phần trăm thù lao.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tích hợp kỹ thuật **Sub-Query Decomposition**: Khi nhận câu hỏi phức hợp, tự động phân tách thành các câu hỏi con độc lập, retrieve song song trên Qdrant và BM25, sau đó hợp nhất contexts trước khi rerank.
- Tinh chỉnh **Metadata Filtering** để phân biệt chính sách hết hiệu lực (v2023) và chính sách hiện hành (v2024).
