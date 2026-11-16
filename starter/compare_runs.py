import json

runs = {"ctrl": "dev-ctrl-b6000", "expand": "dev-expand-b6000", "top5": "dev-top5-b6000"}
data = {k: {t["query_id"]: t for t in json.load(open(f"starter/results/{v}/traces.json", encoding="utf-8"))}
        for k, v in runs.items()}

for qid in data["ctrl"]:
    flags = {k: data[k][qid]["context_hit"] for k in runs}
    if len(set(flags.values())) > 1 or not all(flags.values()):
        print(qid, flags)
        for k in ("expand", "top5"):
            t = data[k][qid]
            found = [c["id"] for c in t["results"] if c["hit"]]
            dropped = [i for i in found if i not in t["context_ids"]]
            print(f"   {k}: chunk đúng trong ứng viên={found}, bị loại do budget={dropped}")