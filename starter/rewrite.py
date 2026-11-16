"""Query rewriting bằng Gemini, rồi dense / TF-IDF / hybrid RRF trên truy vấn đã viết lại.
Chỉ câu hỏi được gửi cho Gemini; gold chỉ dùng để chấm sau khi truy xuất."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from baseline import TfidfIndex, hit, pack_context, validate
from dense import DenseRetriever
from dense_ctx import neighbor_ids

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gemini-3.1-flash-lite")
RRF_K, POOL = 60, 20
PROMPT = (
    "Rewrite the following Vietnamese question about the PostgreSQL 16 documentation "
    "into ONE concise English search query. Use exact PostgreSQL technical terms and "
    "keywords likely to appear in the documentation. Output only the query on a single "
    "line, with no explanation, no quotes and no alternatives.\n\nQuestion: "
)


def rewrite(client, question):
    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=REWRITE_MODEL, contents=PROMPT + question,
                config=types.GenerateContentConfig(temperature=0))
            lines = (r.text or "").strip().splitlines()
            if not lines or not lines[0].strip():
                raise ValueError("Gemini trả về truy vấn rỗng")
            return lines[0].strip().strip('"')
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def rrf(dense_pool, tfidf_pool, by_id):
    scores = {}
    for pool in (dense_pool, tfidf_pool):
        for rank, c in enumerate(pool, 1):
            scores[c["id"]] = scores.get(c["id"], 0) + 1 / (RRF_K + rank)
    top = sorted(scores, key=lambda i: (-scores[i], i))[:3]
    return [{**by_id[i], "score": scores[i], "strategy": "hybrid"} for i in top]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True, type=Path)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budget", type=int, default=6000)
    p.add_argument("--expand", type=int, default=0)
    args = p.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in questions if q.get("split") == args.split]
    validate(chunks, selected)
    by_id = {c["id"]: c for c in chunks}

    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.output / "rewrites.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for q in selected:
        if q["id"] not in cache:
            t0 = time.perf_counter()
            cache[q["id"]] = {"query": rewrite(client, q["question"]),
                              "model": REWRITE_MODEL,
                              "ms": (time.perf_counter() - t0) * 1000}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(q["id"], "->", cache[q["id"]]["query"], flush=True)

    dense = DenseRetriever()
    tfidf = TfidfIndex(chunks)
    methods = {"dense_rw": [], "tfidf_rw": [], "hybrid_rw": []}
    for q in selected:
        rq = cache[q["id"]]["query"]
        t0 = time.perf_counter()
        dense_pool = dense.search(rq, top_k=POOL)
        tfidf_pool = tfidf.search(rq, top_k=POOL)
        top = {"dense_rw": dense_pool[:3], "tfidf_rw": tfidf_pool[:3],
               "hybrid_rw": rrf(dense_pool, tfidf_pool, by_id)}
        retrieval_ms = (time.perf_counter() - t0) * 1000
        for name, top3 in top.items():
            candidates = list(top3)
            if args.expand:
                seen = {c["id"] for c in top3}
                for c in top3:
                    for nid in neighbor_ids(c["id"]):
                        if nid in by_id and nid not in seen:
                            seen.add(nid)
                            candidates.append({**by_id[nid], "score": None, "strategy": "neighbor"})
            context, ids = pack_context(candidates, args.budget)
            marked = [{**c, "hit": hit(c, q["gold"])} for c in candidates]
            methods[name].append({
                "query_id": q["id"], "split": q["split"], "question": q["question"],
                "rewritten_query": rq, "strategy": name, "budget_chars": args.budget,
                "expand": args.expand, "gold": q["gold"], "results": marked,
                "context": context, "context_ids": ids, "context_chars": len(context),
                "retrieval_ms": retrieval_ms, "rewrite_ms": cache[q["id"]]["ms"],
                "hit1": bool(marked[:3] and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked[:3]),
                "context_hit": any(c["hit"] and c["id"] in ids for c in marked)})

    for name, traces in methods.items():
        summary = {"strategy": name, "split": args.split, "n": len(traces),
                   "rewrite_model": REWRITE_MODEL, "budget_chars": args.budget,
                   "expand": args.expand,
                   "hit1": sum(t["hit1"] for t in traces),
                   "hit3": sum(t["hit3"] for t in traces),
                   "context_hit": sum(t["context_hit"] for t in traces),
                   "avg_context_chars": statistics.mean(t["context_chars"] for t in traces),
                   "median_retrieval_ms": statistics.median(t["retrieval_ms"] for t in traces),
                   "median_rewrite_ms": statistics.median(t["rewrite_ms"] for t in traces),
                   "service_cost_usd": "free tier (chưa tính phí)"}
        d = args.output / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{name:10} hit1={summary['hit1']:>2}/{summary['n']} hit3={summary['hit3']:>2} "
              f"ctx={summary['context_hit']:>2} chars={summary['avg_context_chars']:.0f}")


if __name__ == "__main__":
    main()