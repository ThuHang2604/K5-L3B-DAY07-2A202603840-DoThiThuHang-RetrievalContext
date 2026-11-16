"""Dense + (tuỳ chọn) mở rộng chunk lân cận / top-k lớn hơn.
Hit@1, Hit@3 luôn tính trên top-3 xếp hạng gốc; context_hit tính trên context cuối."""
import argparse
import json
import statistics
import time
from pathlib import Path

from baseline import hit, pack_context, validate
from dense import DenseRetriever


def neighbor_ids(cid):
    prefix, idx = cid.rsplit(":", 1)
    n = int(idx)
    return [f"{prefix}:{n - 1:03d}", f"{prefix}:{n + 1:03d}"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True, type=Path)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budget", type=int, default=4000)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--expand", type=int, default=0, help="1 = thêm chunk liền trước/sau")
    args = p.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in questions if q.get("split") == args.split]
    validate(chunks, selected)
    by_id = {c["id"]: c for c in chunks}
    retriever = DenseRetriever()

    traces = []
    for q in selected:
        t0 = time.perf_counter()
        results = retriever.search(q["question"], top_k=args.top_k)
        candidates = list(results)
        if args.expand:
            seen = {r["id"] for r in results}
            for r in results:
                for nid in neighbor_ids(r["id"]):
                    if nid in by_id and nid not in seen:
                        seen.add(nid)
                        candidates.append({**by_id[nid], "score": None, "strategy": "dense-neighbor"})
        context, ids = pack_context(candidates, args.budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        marked = [{**c, "hit": hit(c, q["gold"])} for c in candidates]
        top3 = marked[:3]
        traces.append({
            "query_id": q["id"], "split": q["split"], "question": q["question"],
            "strategy": "dense", "top_k": args.top_k, "expand": args.expand,
            "budget_chars": args.budget, "gold": q["gold"], "results": marked,
            "context": context, "context_ids": ids, "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(top3 and top3[0]["hit"]),
            "hit3": any(c["hit"] for c in top3),
            "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
        })

    summary = {"strategy": "dense", "split": args.split, "n": len(traces),
               "top_k": args.top_k, "expand": args.expand, "budget_chars": args.budget,
               "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
               "context_hit": sum(t["context_hit"] for t in traces),
               "avg_context_chars": statistics.mean(t["context_chars"] for t in traces),
               "median_ms": statistics.median(t["elapsed_ms"] for t in traces),
               "service_cost_usd": 0}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()