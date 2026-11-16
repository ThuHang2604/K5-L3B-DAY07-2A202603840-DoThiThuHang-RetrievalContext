import json
from pathlib import Path

cases = {c["id"]: c for c in json.load(open("starter/data/cases.json", encoding="utf-8"))}
for folder in ("agent-test-chosen", "agent-test-baseline"):
    traces = json.load(open(f"starter/results/{folder}/traces.json", encoding="utf-8"))
    out = []
    for t in traces:
        r = {"query_id": t["query_id"], "citation_support": [None] * len(t["citations"]), "reason": ""}
        if cases[t["query_id"]]["answerable"]:
            r["answer_grade"] = None
        else:
            r["abstention_correct"] = None
        out.append(r)
    path = Path(f"starter/results/{folder}/reviews.json")
    if path.exists():
        print("đã có, bỏ qua:", path)
        continue
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("đã tạo", path)