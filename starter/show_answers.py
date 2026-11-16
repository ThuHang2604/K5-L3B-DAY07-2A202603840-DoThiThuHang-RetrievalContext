import json
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "agent-dev2"
only = set(sys.argv[2:])
traces = json.load(open(f"starter/results/{folder}/traces.json", encoding="utf-8"))
chunks = {c["id"]: c for c in json.load(open("starter/data/converted/chunks.json", encoding="utf-8"))}
qs = {q["id"]: q for q in json.load(open("starter/data/converted/questions.json", encoding="utf-8"))}

for t in traces:
    if only and t["query_id"] not in only:
        continue
    q = qs.get(t["query_id"])
    print("=" * 90)
    print(t["query_id"], t["status"], "|", t["question"])
    if q:
        print("ĐÁP ÁN CHUẨN:", q["expected_answer"])
    print("TRẢ LỜI:", t["answer"])
    for cid in t["citations"]:
        txt = chunks[cid]["text"][:500].replace("\n", " ") if cid in chunks else "(không có trong corpus)"
        print(f"  [{cid}] {txt}")