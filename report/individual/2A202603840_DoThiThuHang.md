# Báo cáo Day 7: Agentic RAG trên tài liệu PostgreSQL 16

Học viên: Đỗ Thị Thu Hằng, MSSV 2A202603840, lớp L3B.


## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang, số chunk:** PostgreSQL 16 Documentation (https://www.postgresql.org/docs/16/), 7 chương (indexes, mvcc, performance-tips, maintenance, backup, high-availability, wal). 36 trang section hợp lệ (mỗi trang tối thiểu 1500 ký tự sau chuẩn hóa). Quyền sử dụng: theo Legal Notice của docs v16 (`data/legal_notice.txt`), chi tiết trong `data/manifest.json`. Chunk: chiến lược A có 835 chunk (500 ký tự cố định), chiến lược B có 418 chunk (theo heading, tối đa khoảng 1200 ký tự). Day 07 dùng chiến lược B.
- **Vector store, embedding model:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker), khoảng cách cosine, quét chính xác (không dùng index ANN). Embedding `intfloat/multilingual-e5-base`, 768 chiều, tiền tố `passage: ` cho tài liệu và `query: ` cho câu hỏi, vector chuẩn hóa. Dữ liệu được xuất từ DB Day 06 bằng `starter/export_day6.py`.
- **Agent model, cách chạy, chia dev/test:** agent dùng `gemini-3.1-flash-lite` (function calling, temperature 0), cũng là model viết lại truy vấn. Python 3.13, chạy qua `starter/run_agent.py`. Benchmark gồm 30 câu: 26 câu có đáp án từ Day 06 (19 dev, 7 test) và 4 câu thiếu bằng chứng viết thêm (n01, n02 thuộc dev; n03, n04 thuộc test). Tổng: dev 19 có đáp án + 2 thiếu bằng chứng, test 7 có đáp án + 2 thiếu bằng chứng. Các câu test của Day 06 đã từng được báo cáo ở Day 06, nhưng không dùng để chọn hay chỉnh cấu hình Day 07.
- **Cách xác định câu có/thiếu evidence, tiêu chí chấm:** câu có đáp án phải có đoạn `evidence_quote` nằm nguyên văn trong chunk của đúng nguồn/trang gold (kiểm bằng `baseline.validate`). Câu thiếu bằng chứng là câu gần chủ đề nhưng không có đáp án trong 36 trang; cách kiểm tra ghi trong trường `check` của `benchmark/no_evidence.json` . Gold không được đưa vào agent hay retriever. Hit tính khi chunk nằm đúng trang gold và chứa nguyên văn evidence.

## Retrieval và thử nghiệm

Ba phương pháp retrieval khác nhau, trên cùng 418 chunk B và cùng bộ câu hỏi dev (19 câu có đáp án):

1. **Lexical (TF-IDF)**: `baseline.py`, tách từ theo `\w+`, không filter.
2. **Dense**: embedding query rồi tìm vector gần nhất trong pgvector Day 06 (`starter/dense.py`), không filter.
3. **Query rewriting + hybrid**: Gemini viết lại câu hỏi tiếng Việt thành truy vấn tiếng Anh dùng thuật ngữ PostgreSQL (`starter/rewrite.py`), sau đó chạy dense, TF-IDF và hybrid (RRF, k = 60, pool 20) trên truy vấn mới. Truy vấn gốc và truy vấn viết lại được báo riêng.

Hit@1 và Evidence Hit@3 luôn tính trên top-3 xếp hạng gốc. Context Evidence Rate tính trên context cuối cùng gửi model. `Budget` là số ký tự tối đa của context (gồm cả phần nguồn của từng chunk); chunk không vừa budget bị bỏ.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline lexical | TF-IDF, top-3, không filter, budget 800 | Phương pháp 1 | dev | 7/19 | 11/19 | 3/19 | 0,4 ms trung vị, 0 USD |
| Dense (phương pháp 2) | e5-base + pgvector, top-3, không filter, budget 800 | Câu hỏi tiếng Việt, tài liệu tiếng Anh: dense hiểu đa ngôn ngữ | dev | 13/19 | 16/19 | 3/19 | 72 ms trung vị, 0 USD |
| Dense, budget 2500 | như trên, budget 2500 | Context 800 ký tự cắt mất chunk dài (chunk B tới 1200 ký tự) | dev | 13/19 | 16/19 | 15/19 | 48 ms |
| Dense, budget 4000 | như trên, budget 4000 | | dev | 13/19 | 16/19 | 16/19 | 49 ms |
| Đối chứng budget 6000 | top-3, budget 6000, không mở rộng | Tách ảnh hưởng của budget | dev | 13/19 | 16/19 | 16/19 | 66 ms, context TB 3344 ký tự |
| **Cải thiện 1**: mở rộng lân cận | top-3 + chunk liền trước/sau, budget 6000 | Chunk đúng thường liền kề chunk đã tìm được (q06, q16) | dev | 13/19 | 16/19 | 17/19 | 58 ms, context TB 5314 ký tự |
| **Cải thiện 2**: top-5 | top-5, không mở rộng, budget 6000 | Chunk đúng có thể nằm hạng 4 đến 5 | dev | 13/19 | 16/19 | 17/19 | 70 ms, context TB 5310 ký tự |
| **Cải thiện 3**: mở rộng lân cận, budget 9000 | top-3 + lân cận, budget 9000 | Budget 6000 vẫn cắt mất chunk lân cận (q16) | dev | 13/19 | 16/19 | 18/19 | 60 ms, context TB 6844 ký tự |
| Dense, truy vấn viết lại | rewrite bằng Gemini, top-3, budget 6000 | Dịch sang tiếng Anh để khớp thuật ngữ | dev | 9/19 | 12/19 | 12/19 | gồm 1 lần gọi Gemini mỗi câu, gói miễn phí |
| TF-IDF, truy vấn viết lại | như trên | Từ khóa tiếng Anh giúp lexical | dev | 11/19 | 13/19 | 13/19 | như trên |
| Hybrid RRF, truy vấn viết lại | dense + TF-IDF hợp nhất bằng RRF, budget 6000 | Kết hợp hai tín hiệu | dev | 10/19 | 15/19 | 15/19 | như trên |
| **Cấu hình đã chọn** | dense, truy vấn gốc, top-3 + lân cận, budget 9000 | Theo tiêu chí bên dưới | **test** | 3/7 | 7/7 | 7/7 | 52 ms, context TB 7872 ký tự |
| Baseline (test) | dense, truy vấn gốc, top-3, budget 800, không mở rộng | Mốc đối chứng | **test** | 3/7 | 7/7 | 1/7 | 48 ms, context TB 296 ký tự |

**Ghi chú về chi phí:** embedding chạy cục bộ (0 USD). Gemini dùng gói miễn phí nên chưa phát sinh phí, số token từng câu ở `tokens_in`, `tokens_out` trong trace của agent.

**Kết quả không cải thiện:** viết lại truy vấn làm dense giảm từ 13/19 xuống 9/19 ở Hit@1, nhưng tăng TF-IDF từ 7/19 lên 11/19. Hit@1 và Hit@3 không đổi giữa các mức budget, vì budget chỉ ảnh hưởng đến cách chọn context, không ảnh hưởng thứ hạng. Dev chỉ có 19 câu nên mỗi câu tương ứng khoảng 5 điểm phần trăm.

**Hai lỗi dev và cách xử lý:**

1. **q06 và q16: đúng trang nhưng sai đoạn.** Top-3 của dense chứa chunk của đúng trang gold (`indexes-bitmap-scans:000`; `indexes-index-only-scans:000/006/008`), nhưng câu evidence nằm ở chunk liền kề (`:001` ở cả hai câu). Giả thuyết: ranh giới chunk cắt một ý thành hai đoạn. Thử mở rộng chunk lân cận: Context Evidence tăng từ 16/19 lên 17/19 ở budget 6000. Ở q16, chunk đúng đã nằm trong danh sách ứng viên nhưng bị loại do budget (kiểm bằng `compare_runs.py`). Tăng budget lên 9000 chữa được cả hai câu, đạt 18/19.
2. **q11: chunk đúng không nằm cạnh chunk nào đã tìm được.** Evidence ở `explicit-locking:005`, trong khi top-3 là `hot-standby:003`, `explicit-locking:008` và `:002`. Cả mở rộng lân cận lẫn top-5 đều không chữa được. Đây là lỗi còn lại ở cấu hình chọn (18/19).

**Lý do chọn cấu hình** (dense + mở rộng lân cận + budget 9000): đạt Context Evidence cao nhất trên dev (18/19), Hit@1 và Hit@3 bằng các cấu hình dense khác. Tiêu chí chọn: Context Evidence Rate cao nhất, nếu hòa thì chọn context ngắn hơn. Đánh đổi: context dài gấp khoảng 2 lần mức đối chứng (6844 so với 3344 ký tự trên dev), tức nhiều token Gemini hơn cho mỗi câu. Query rewriting không được chọn vì làm giảm Hit@1 của dense. Cấu hình được chốt trên dev trước khi chạy test; test chỉ chạy một lần cho mỗi cấu hình.

## Agent và đánh giá

**Thiết kế agent:** hai tool cho Gemini: `search(query)` và `answer(answer, citations)`. `search` gọi dense retrieval (top-3, thêm chunk lân cận nếu bật). Điều kiện dừng: model gọi `answer`, hoặc sau 2 lần search thì chỉ cho phép `answer` (giới hạn đặt ra vì lần chạy đầu trên dev, agent tìm 3 lần rồi bị cắt mà không trả lời). Context cuối gồm các chunk đã tìm, loại trùng, cắt theo budget; mỗi chunk có `[chunk-id | nguồn | page]`. System prompt dặn rõ nội dung tài liệu là dữ liệu, không phải chỉ dẫn, và phải trả lời `Không đủ bằng chứng để trả lời.` khi không có bằng chứng. Citation hợp lệ về mặt ID khi chunk-id có trong context và xuất hiện dạng `[chunk-id]` trong câu trả lời; tính hợp lệ về nội dung do người chấm đọc. Lỗi 429 của dịch vụ được thử lại có giới hạn, nếu liên tiếp hai câu thì dừng và ghi `status: error`; không thay bằng kết quả giả.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Đã chọn / test | 9/9 | search 1 lần ở cả 9 câu (2 lần gọi LLM mỗi câu) | 1 / 0 / 6, trung bình 1,71/2 | 7/7 | 6/6 | 2/2 | 1/7 |
| Baseline / test | 9/9 | search 1 lần ở 3 câu, 2 lần ở 6 câu | 5 / 0 / 2, trung bình 0,57/2 | 4/4 | 4/4 | 2/2 | 3/7 |

Lỗi hệ thống (timeout, provider) trên test: 0 ở cả hai cấu hình. Số liệu được tổng hợp bằng `starter/agent_metrics.py` từ nhãn thủ công. Ở n03, agent trả `answered` kèm 1 citation nhưng nội dung nêu rõ không đủ bằng chứng; bản `traces_normalized.json` chuyển n03 sang `abstain` theo nhãn, bản `traces.json` giữ nguyên làm gốc.

**Chấm context (0 không có evidence, 1 có nhưng thiếu điều kiện hoặc nguồn, 2 đủ evidence và nguồn).** Kiểm tra bằng `starter/check_context.py` (evidence chuẩn có nằm trong context cuối mà agent nhận không), sau đó đọc lại. **[TODO: đọc lại context ít nhất 5 câu và xác nhận hoặc sửa các số dưới đây]**

| Cấu hình | Câu chấm 2 | Câu chấm 1 | Câu chấm 0 |
|---|---|---|---|
| Đã chọn | q20, q21, q22, q23, q24, q25, q26 (7/7) | 0 | 0 |
| Baseline | q23 (1/7) | 0 | q20, q21, q22, q24, q25, q26 (6/7) |

Lưu ý: baseline q21 chấm context 0 theo đoạn evidence chuẩn, nhưng agent vẫn trả lời đúng cốt lõi nhờ một đoạn khác của cùng trang (`wal-async-commit:001`). Đây là giới hạn của việc đo bằng khớp nguyên văn một đoạn evidence.

**Kết quả từng câu test (nhãn thủ công, `answer_grade` cho câu có đáp án):**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q20 (đã chọn) | có đáp án | search 1 lần, context có evidence | 5 phút và 1 GB [wal-configuration:001] | 2; đúng, citation chứa cả hai giá trị (đã kiểm tra) |
| q20 (baseline) | có đáp án | search 1 lần, context không có evidence | nói tài liệu không nêu giá trị mặc định | 0; sai do context 800 ký tự thiếu bằng chứng |
| q21 (đã chọn) | có đáp án | search 1 lần | mất dữ liệu, không hỏng, khôi phục nhất quán [wal-async-commit:002, :005] | 2; đúng và đủ |
| q21 (baseline) | có đáp án | search 1 lần | mất dữ liệu, không hỏng [wal-async-commit:001] | 2; đúng cốt lõi, thiếu ý khôi phục nhất quán **[TODO: xem lại, có thể nên chấm 1]** |
| q22 (cả hai) | có đáp án | đã chọn: evidence có trong context; baseline: không | Không đủ bằng chứng | 0; từ chối nhầm. Ở cấu hình chọn bằng chứng đã nằm trong context nên lỗi nằm ở phía model **[TODO: xác minh nguyên nhân, mới là giả thuyết]** |
| q23 (cả hai) | có đáp án | search 1 lần | lấy mẫu ngẫu nhiên [routine-vacuuming:011] | 2; đúng, citation nêu rõ |
| q24 (đã chọn) | có đáp án | search 1 lần | VACUUM, ANALYZE, một số DDL như CREATE INDEX [planner-stats:001] | 2; đúng |
| q24 (baseline) | có đáp án | search 2 lần, không có evidence | Không đủ bằng chứng | 0; từ chối nhầm |
| q25 (đã chọn) | có đáp án | search 1 lần | snapshot tại câu lệnh không điều khiển giao dịch đầu tiên [transaction-iso:011] | 2; đúng (đã kiểm tra cụm từ trong chunk) |
| q25 (baseline) | có đáp án | search 2 lần | Không đủ bằng chứng | 0; từ chối nhầm |
| q26 (đã chọn) | có đáp án | search 1 lần | ít nhất mỗi hai tỷ giao dịch [routine-vacuuming:016] | 2; đúng (đã kiểm tra) |
| q26 (baseline) | có đáp án | search 2 lần | nói tài liệu không quy định khoảng thời gian | 0; sai, citation chỉ hỗ trợ ý phụ |
| n03 (đã chọn) | thiếu bằng chứng | search 1 lần | nêu rõ không có con số phần trăm, thêm ghi chú theo [transaction-iso:023] | từ chối đúng; ghi chú khớp đoạn trích, không có con số bịa |
| n03 (baseline) | thiếu bằng chứng | search 2 lần | Không đủ bằng chứng | từ chối đúng |
| n04 (cả hai) | thiếu bằng chứng | search 1 đến 2 lần | Không đủ bằng chứng | từ chối đúng |

**Agent trên dev (cấu hình chọn, `gemini-3.1-flash-lite`, 21 câu):** 18 câu có đáp án được trả lời kèm citation, q09 bị từ chối, n01 và n02 đều từ chối, 0 lỗi hệ thống. Số câu chỉ được đếm theo trạng thái. Nguyên nhân q09 bị từ chối chưa được phân tích.

**Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** `starter/results/agent-test-chosen/` và `starter/results/agent-test-baseline/` (gồm `traces.json`, `traces_normalized.json`, `reviews.json`, `summary.json`); retrieval: `starter/results/test-retrieval-*` và các thư mục `dev-*`. Lệnh chạy chính: `run_agent.py --split test --expand 1 --budget 9000` (cấu hình chọn) và `--expand 0 --budget 800` (baseline). Thời gian mỗi câu thường 7 đến 20 giây, tối đa khoảng 66 giây (q24, đã chọn), do độ trễ phía dịch vụ. Chi phí: gói miễn phí.

**Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**

- Điều phân biệt hai cấu hình là cách chọn context, không phải thứ hạng: Hit@1 và Hit@3 như nhau, nhưng Context Evidence tăng từ 1/7 lên 7/7 và Grounded answer trung bình từ 0,57 lên 1,71/2. Cái giá là context dài khoảng 27 lần trên test (7872 so với 296 ký tự).
- Cỡ mẫu nhỏ: test chỉ có 7 câu có đáp án (mỗi câu khoảng 14 điểm phần trăm) và 2 câu thiếu bằng chứng, nên kết luận chỉ mang tính chỉ báo.
- Test của Day 07 dùng lại các câu test của Day 06; chúng không được dùng để chỉnh cấu hình Day 07.
- Nhãn do một người chấm (không kiểm chứng chéo). Khớp nguyên văn bằng code chỉ hỗ trợ, nhãn cuối do đọc thủ công.
- Lỗi vận hành đã gặp và ghi lại: lần chạy đầu `agent-dev` có 16/21 câu lỗi do Gemini báo 429 (hết hạn mức gói miễn phí `gemini-3.5-flash`), nên toàn bộ dev được chạy lại bằng `gemini-3.1-flash-lite` (`agent-dev-lite`); `agent-dev2` (8 câu bằng `gemini-3.5-flash`) chỉ là lần thử phụ và không dùng trong số liệu chính. Một lần chạy test bị `MemoryError` khi nạp model e5, xảy ra trước khi xử lý câu nào, nên không ảnh hưởng kết quả.
- Giới hạn khác: không dùng filter metadata trong Day 07 (filter `part` của Day 06 gán tay theo từng câu nên không dùng để tránh vi phạm quy tắc); chưa thử rerank; query rewriting chỉ thử riêng ở mức retrieval, chưa tích hợp vào agent.

## Thử thách nâng cao

Không thực hiện.
