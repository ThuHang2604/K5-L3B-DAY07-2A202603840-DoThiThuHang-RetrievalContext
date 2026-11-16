## Bài làm của học viên — Đỗ Thị Thu Hằng (2A202603840, L3B)

**Đề tài:** agent Gemini (function calling) trả lời câu hỏi về tài liệu **PostgreSQL 16** (36 trang section hợp lệ, 418 chunk chiến lược B từ Day 06), embedding `intfloat/multilingual-e5-base`, lưu bằng PostgreSQL + pgvector. So sánh 3 phương pháp retrieval (TF-IDF, dense, query rewriting + hybrid RRF), thử các cấu hình cải thiện trên dev, chốt rồi chạy test một lần.

**Báo cáo:** [report/individual/2A202603840_DoThiThuHang.md](report/individual/2A202603840_DoThiThuHang.md)

### Kết quả chính (test: 7 câu có đáp án + 2 câu thiếu bằng chứng)

| Cấu hình | Evidence Hit@3 | Context Evidence | Grounded answer (trung bình /2) | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|
| Baseline (dense, budget 800) | 7/7 | 1/7 | 0,57 | 2/2 | 3/7 |
| Đã chọn (dense + chunk lân cận, budget 9000) | 7/7 | 7/7 | 1,71 | 2/2 | 1/7 |

Cỡ mẫu test nhỏ; xem giới hạn trong báo cáo.

### Cài đặt (Windows CMD, Python 3.11 trở lên, Docker)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install "psycopg[binary]" python-dotenv sentence-transformers google-genai
copy .env.example .env
```

Điền `.env` (không commit): `DATABASE_URL`, `GEMINI_API_KEY`, `AGENT_MODEL=gemini-3.1-flash-lite`, `EMBEDDING_MODEL=intfloat/multilingual-e5-base`.

Lưu ý: trên Windows, cài `torch` bằng Python từ Microsoft Store có thể lỗi vì đường dẫn quá dài (Long Path). Tác giả đã chạy các script bằng Python trong `.venv` của repo Day 06 (đã có sẵn thư viện). Có thể bật Long Path hoặc dùng đường dẫn ngắn.

**Cần có database Day 06 đang chạy** (pgvector, cổng 55433, database `day6_lab`, bảng `chunks` có 418 dòng `strategy = 'B'`). Bật từ repo Day 06: `cd starter` rồi `docker compose -p day6_student_starter up -d --wait`. Nếu DB trống, chạy lại `python -m src.lab6 ingest` ở repo Day 06.

### Chạy lại (từ thư mục gốc repo)

```cmd
python starter\export_day6.py
python starter\convert_day6.py --chunks starter\data\day6_export\day6_chunks.json --questions starter\data\day6_export\day6_questions.json --output starter\data\converted
python starter\build_cases.py
python starter\baseline.py --chunks starter\data\converted\chunks.json --questions starter\data\converted\questions.json --split dev --output starter\results\dev-tfidf
python starter\dense.py --chunks starter\data\converted\chunks.json --questions starter\data\converted\questions.json --split dev --output starter\results\dev-dense
python starter\dense_ctx.py --chunks starter\data\converted\chunks.json --questions starter\data\converted\questions.json --split dev --budget 9000 --expand 1 --top_k 3 --output starter\results\dev-expand-b9000
python starter\rewrite.py --chunks starter\data\converted\chunks.json --questions starter\data\converted\questions.json --split dev --output starter\results\dev-rewrite
python starter\run_agent.py --split dev --output starter\results\agent-dev-lite
```

Chỉ chạy test sau khi đã chốt cấu hình trên dev, mỗi cấu hình một lần:

```cmd
python starter\run_agent.py --split test --expand 1 --budget 9000 --output starter\results\agent-test-chosen
python starter\run_agent.py --split test --expand 0 --budget 800 --output starter\results\agent-test-baseline
python starter\dense_ctx.py --chunks starter\data\converted\chunks.json --questions starter\data\converted\questions.json --split test --budget 9000 --expand 1 --top_k 3 --output starter\results\test-retrieval-chosen
python starter\dense_ctx.py --chunks starter\data\converted\chunks.json --questions starter\data\converted\questions.json --split test --budget 800 --expand 0 --top_k 3 --output starter\results\test-retrieval-baseline
```

Chấm thủ công và tổng hợp: nhãn ghi trong `starter/fill_reviews.py` (đọc từng câu bằng `show_answers.py`, kiểm tra context bằng `check_context.py`), sau đó:

```cmd
python starter\fill_reviews.py
python starter\agent_metrics.py --cases starter\data\cases.json --traces starter\results\agent-test-chosen\traces_normalized.json --reviews starter\results\agent-test-chosen\reviews.json --split test --output starter\results\agent-test-chosen\summary.json
python starter\agent_metrics.py --cases starter\data\cases.json --traces starter\results\agent-test-baseline\traces_normalized.json --reviews starter\results\agent-test-baseline\reviews.json --split test --output starter\results\agent-test-baseline\summary.json
```

Test code (mock, không phải đánh giá agent thật). Trên Windows nên bật chế độ UTF-8 vì một test mẫu đọc file không chỉ định encoding:

```cmd
set PYTHONUTF8=1
python -m unittest discover -s tests -v
```

### Dữ liệu và kết quả

| Đường dẫn | Nội dung |
|---|---|
| `benchmark/questions.jsonl` | 26 câu có đáp án (19 dev, 7 test) kèm evidence |
| `benchmark/no_evidence.json` | 4 câu thiếu bằng chứng (n01, n02 dev; n03, n04 test) và cách kiểm tra |
| `data/manifest.json`, `data/legal_notice.txt` | nguồn, phiên bản, giấy phép của tài liệu PostgreSQL 16 |
| `starter/data/` | chunk đã chuyển schema, `cases.json` |
| `starter/results/` | trace, nhãn, `summary.json` của từng lần chạy (kể cả các lần lỗi do hạn mức Gemini) |
| `starter/*.py` | code: xuất dữ liệu, dense, lexical, rewrite, agent, đánh giá |

Không commit `.env` hay API key. Chạy lại từ đầu cần Gemini API key riêng; gói miễn phí có hạn mức theo model nên có thể phải dùng `--resume`.

---
# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
