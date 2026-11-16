import json
from pathlib import Path

qs = json.loads(Path("starter/data/converted/questions.json").read_text(encoding="utf-8"))
ne = json.loads(Path("benchmark/no_evidence.json").read_text(encoding="utf-8"))
cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True} for q in qs]
cases += [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": False} for q in ne]
Path("starter/data/cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(cases), "cases:", sum(c["answerable"] for c in cases), "có đáp án,", sum(not c["answerable"] for c in cases), "thiếu bằng chứng")