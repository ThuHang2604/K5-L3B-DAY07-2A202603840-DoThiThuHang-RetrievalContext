import json
from pathlib import Path

LABELS = {
    "agent-test-chosen": {
        "q20": (2, [True], "Đúng 5 phút và 1 GB; đoạn wal-configuration:001 nêu giá trị mặc định (đã kiểm tra)."),
        "q21": (2, [True, True], "Đúng: mất dữ liệu, không hỏng; nêu cả khôi phục nhất quán."),
        "q22": (0, [], "Từ chối nhầm câu có đáp án; cần xem trace để biết nguyên nhân."),
        "q23": (2, [True], "Đúng: lấy mẫu ngẫu nhiên; citation nêu rõ."),
        "q24": (2, [True], "Đúng: VACUUM, ANALYZE, một số DDL như CREATE INDEX."),
        "q25": (2, [True], "Đúng: snapshot từ câu lệnh không điều khiển giao dịch đầu tiên (đã kiểm tra)."),
        "q26": (2, [True], "Đúng: ít nhất mỗi hai tỷ giao dịch (đã kiểm tra)."),
        "n03": (True, [], "Nêu rõ không đủ bằng chứng; ghi chú thêm khớp đoạn trích, không có con số bịa."),
        "n04": (True, [], "Từ chối đúng."),
    },
    "agent-test-baseline": {
        "q20": (0, [True], "Sai: nói tài liệu không nêu giá trị mặc định (context 800 ký tự thiếu bằng chứng)."),
        "q21": (2, [True], "Đúng cốt lõi: mất dữ liệu, không hỏng; thiếu ý khôi phục nhất quán."),
        "q22": (0, [], "Từ chối nhầm câu có đáp án."),
        "q23": (2, [True], "Đúng: lấy mẫu ngẫu nhiên."),
        "q24": (0, [], "Từ chối nhầm câu có đáp án."),
        "q25": (0, [], "Từ chối nhầm câu có đáp án."),
        "q26": (0, [True], "Sai: nói tài liệu không quy định khoảng thời gian; citation chỉ hỗ trợ ý phụ."),
        "n03": (True, [], "Từ chối đúng."),
        "n04": (True, [], "Từ chối đúng."),
    },
}

cases = {c["id"]: c for c in json.load(open("starter/data/cases.json", encoding="utf-8"))}
for folder, labels in LABELS.items():
    base = Path(f"starter/results/{folder}")
    traces = json.load(open(base / "traces.json", encoding="utf-8"))
    reviews, normalized = [], []
    for t in traces:
        qid = t["query_id"]
        value, support, reason = labels[qid]
        if cases[qid]["answerable"] and len(support) != len(t["citations"]):
            raise SystemExit(f"{folder}/{qid}: cần {len(t['citations'])} nhãn citation")
        r = {"query_id": qid, "citation_support": support, "reason": reason}
        t2 = dict(t)
        if cases[qid]["answerable"]:
            r["answer_grade"] = value
        else:
            r["abstention_correct"] = value
            if value and t["status"] == "answered":  # trả lời có nêu rõ thiếu bằng chứng: chuẩn hóa trong bản sao
                t2["status"], t2["citations"] = "abstain", []
                t2["normalized_note"] = "Chuyển answered -> abstain theo nhãn thủ công; trace gốc giữ trong traces.json"
        reviews.append(r)
        normalized.append(t2)
    (base / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "traces_normalized.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    print("đã ghi", base)