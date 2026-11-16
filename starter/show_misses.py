import json
t = json.load(open("starter/results/dev-dense-b4000/traces.json", encoding="utf-8"))
for x in t:
    if not x["hit3"]:
        print(x["query_id"], x["question"])
        print("  gold:", [g["page"] for g in x["gold"]])
        print("  top3:", [(r["id"], r["page"]) for r in x["results"]])
        print()