import json

qs = {q["id"]: q for q in json.load(open("starter/data/converted/questions.json", encoding="utf-8"))}
for folder in ("agent-test-chosen", "agent-test-baseline"):
    traces = json.load(open(f"starter/results/{folder}/traces.json", encoding="utf-8"))
    print(folder)
    for t in traces:
        q = qs.get(t["query_id"])
        if not q:
            continue
        searches = [e for e in t["events"] if e.get("event") == "search"]
        ctx = searches[-1]["context"] if searches else ""
        has = any(g["evidence"] in ctx for g in q["gold"])
        print(f"  {t['query_id']}: evidence trong context cuối = {has} | truy vấn agent: {[e['query'] for e in searches]}")