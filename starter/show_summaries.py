import json
from pathlib import Path

rows = []
for f in sorted(Path("starter/results").glob("*/summary.json")):
    s = json.loads(f.read_text(encoding="utf-8"))
    rows.append((f.parent.name, s.get("strategy"), s.get("split"), s.get("n"),
                 s.get("top_k", 3), s.get("expand", 0), s.get("budget_chars", 800),
                 s.get("hit1"), s.get("hit3"), s.get("context_hit"),
                 round(s.get("avg_context_chars", 0))))

print(f"{'thư mục':24} {'pp':6} {'split':5} {'n':>3} {'k':>2} {'exp':>3} {'budget':>6} {'H@1':>4} {'H@3':>4} {'ctx':>4} {'chars':>6}")
for r in rows:
    print(f"{r[0]:24} {r[1]:6} {r[2]:5} {r[3]:>3} {r[4]:>2} {r[5]:>3} {r[6]:>6} {r[7]:>4} {r[8]:>4} {r[9]:>4} {r[10]:>6}")