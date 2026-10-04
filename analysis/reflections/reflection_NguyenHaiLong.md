# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Hải Long  
**Khóa:** K4 - Track 3A  
**MSSV:** 2A202602471  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | "Threshold 0.85 giúp gom các câu có ngữ nghĩa tương đồng vào cùng chunk, không ngắt đôi ý giữa chừng như basic chunking cắt thô theo ký tự/đoạn." |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | "Tạo parent (2048 chars) bao quát ngữ cảnh và children (256 chars) cho retrieval chính xác cao; khi match child thì trả về parent để LLM đọc ngữ cảnh rộng." |
| Structure-Aware chunking | M1 | `chunk_structure_aware()` | "Phân tích cú pháp Markdown header (#, ##, ###) để phân vùng section logic, giữ nguyên vẹn bảng biểu, danh sách và quy chế." |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "RRF score(d) = Σ 1/(k + rank + 1) với k=60; cân bằng giữa từ khóa chính xác (lexical BM25 tiếng Việt) và ngữ nghĩa sâu (dense vector BAAI/bge-m3)." |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | "Đưa 20 candidates qua model bge-reranker-v2-m3 để chấm điểm cross-attention query-document, chọn top 3 chính xác nhất, loại bỏ tài liệu nhiễu." |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | "Đánh giá 4 chỉ số: Faithfulness (độ trung thực), Answer Relevancy (mức liên quan câu trả lời), Context Precision (tỉ lệ chunk đúng trong retrieved), Context Recall (tỉ lệ thông tin cần thiết được tìm thấy)." |
| Diagnostic Tree & Failure Analysis | M4 | `failure_analysis()` | "Tự động phân loại nguyên nhân lỗi theo cây chẩn đoán (LLM hallucination, thiếu chunk liên quan, nhiễu chunk) và đề xuất cách khắc phục tương ứng." |
| Contextual embeddings & Enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` | "Kỹ thuật contextual prepend theo Anthropic benchmark giúp giảm 49% retrieval failure; chế độ combined mode gọi 1 API duy nhất cho 4 tác vụ tiết kiệm 75% chi phí API." |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `ModuleNotFoundError: No module named 'rank_bm25'`, `sentence_transformers` khi chạy pytest ban đầu.
  - Cảnh báo xung đột dependency: `langgraph 1.2.11 requires langchain-core<2,>=1.4.7, but you have langchain-core 0.2.43`.
  - IDE Pylance báo gạch đỏ `Import "openai" could not be resolved` do hệ thống Windows có 2 phiên bản Python (3.14 và 3.12).
- **Nguyên nhân gốc rễ & Cách debug:**
  - Hệ thống máy tính có Python 3.14 làm mặc định của OS, trong khi môi trường cài đặt gói bài lab là Python 3.12. Đã tạo file `.vscode/settings.json` trỏ rõ `python.defaultInterpreterPath` về Python 3.12.
  - Chạy `pip install -r requirements.txt` để cài đặt đầy đủ bộ thư viện cho Python 3.12. Thông báo `langgraph` là cảnh báo xung đột của môi trường chung nhưng không ảnh hưởng vì Lab 18 không phụ thuộc vào `langgraph`.
  - Cấu hình file `.env` chuẩn endpoint `https://openrouter.ai/api/v1` cùng model `openai/gpt-4o-mini`, tinh chỉnh `config.py` và `m4_eval.py` truyền model rõ ràng tránh lỗi 404 từ OpenRouter.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Đã củng cố kiến thức về cách hoạt động của RRF (Reciprocal Rank Fusion) và sự khác biệt giữa Bi-Encoder (Dense retrieval) và Cross-Encoder (Reranker).
  - Nắm vững quy trình đánh giá định lượng hệ thống RAG bằng framework RAGAS.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành tại Lab 18, lập kế hoạch cụ thể áp dụng vào project cá nhân:

### Project: Trợ lý AI Hỏi - Đáp Nội Bộ Doanh Nghiệp (Internal Enterprise Q&A Copilot)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng basic paragraph chunking (split 500 ký tự) + Dense retrieval đơn thuần (ChromaDB) + LLM Prompt trực tiếp.
- **Vấn đề / Bottlenecks đang gặp:**
  - Retrieval recall kém với các câu hỏi chứa mã số điều luật, từ khóa chuyên ngành hoặc từ viết tắt.
  - Context precision thấp: Lấy top 5 chunks thì thường có 2-3 chunks lạc đề làm loãng prompt, gây ra hiện tượng hallucination.
  - Thiếu hệ thống metric định lượng chất lượng câu trả lời một cách tự động.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chuyển sang kết hợp **Hierarchical chunking** (parent 2048 chars, child 256 chars) cho văn bản dạng tài liệu chính sách, và **Structure-Aware** cho tài liệu kỹ thuật có bảng biểu.
2. **Search retrieval:** Áp dụng **Hybrid Search (BM25 có word segmentation tiếng Việt bằng underthesea + Dense BAAI/bge-m3)** kết hợp thuật toán **RRF** ($k=60$) để giải quyết triệt để vấn đề tìm kiếm từ khóa chính xác.
3. **Reranking:** Tích hợp mô hình Cross-encoder `BAAI/bge-reranker-v2-m3` để re-rank từ top 20 candidate xuống top 3-4 chunks đưa vào LLM context, giúp tăng vọt Context Precision và giảm latency so với việc nhét quá nhiều chunk vào context window.
4. **Evaluation:** Thiết lập bộ benchmark tự động bằng **RAGAS** với 4 metrics chuẩn, chạy định kỳ trong CI/CD pipeline mỗi khi cập nhật tri thức hoặc đổi prompt template.
5. **Enrichment:** Sử dụng **Combined Mode Enrichment** (tóm tắt + tạo câu hỏi giả định HyQA + contextual prepend) ở giai đoạn tiền xử lý tài liệu trước khi nạp vào vector store.

#### 3. Timeline triển khai
- **Tuần 1:** Refactor pipeline chunking (Hierarchical) và thiết lập Hybrid Search BM25 + Qdrant.
- **Tuần 2:** Tích hợp Cross-Encoder Reranker, kiểm thử đo lường độ trễ (latency breakdown) và độ chính xác top-k.
- **Tuần 3:** Tự động hóa bộ test set 50 câu hỏi nội bộ và đánh giá định lượng bằng RAGAS.
- **Tuần 4:** Triển khai thử nghiệm (pilot) với người dùng nội bộ và tinh chỉnh prompt template dựa trên Diagnostic Tree.
