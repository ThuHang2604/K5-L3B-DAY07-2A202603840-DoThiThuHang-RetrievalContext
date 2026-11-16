import json

t = json.load(open("starter/results/agent-dev/traces.json", encoding="utf-8"))
for x in t:
    events = [e.get("event") for e in x["events"]]
    print(x["query_id"], x["status"], "| events:", events, "| llm_calls:", x["llm_calls"],
          "| tokens_in:", x["tokens_in"])
    if x["status"] == "error":
        print("   LỖI:", x.get("error", "")[:300])
    elif x["status"] == "abstain":
        print("   câu trả lời cuối:", x["answer"][:100])