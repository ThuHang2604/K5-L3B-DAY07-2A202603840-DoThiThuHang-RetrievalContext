import json

chunks = json.load(open("starter/data/converted/chunks.json", encoding="utf-8"))
traces = json.load(open("starter/results/dev-dense-b4000/traces.json", encoding="utf-8"))

for t in traces:
    if t["hit3"]:
        continue
    g = t["gold"][0]
    holders = [c["id"] for c in chunks if g["evidence"] in c["text"] and c["page"] == g["page"]]
    print(t["query_id"], "| evidence nằm ở:", holders, "| top3:", [r["id"] for r in t["results"]])