"""Dense retrieval trên pgvector Day 06 (chiến lược B), đánh giá cùng cách với baseline."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from baseline import evaluate, validate

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
FILTER_KEYS = {"part", "chapter"}


class DenseRetriever:
    def __init__(self, strategy="B"):
        self.conn = psycopg.connect(os.environ["DATABASE_URL"])
        self.model = SentenceTransformer(MODEL_NAME)
        self.model.max_seq_length = 512
        self.strategy = strategy

    def search(self, query, *, filters=None, top_k=3):
        qvec = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        v = "[" + ",".join(f"{float(x):.6f}" for x in qvec) + "]"
        where, params = "strategy = %s", [self.strategy]
        for key, val in (filters or {}).items():
            if key in FILTER_KEYS:
                where += f" AND {key} = %s"
                params.append(val)
        sql = ("SELECT chunk_id, document_id, source, page, part, chapter, section_title, text, "
               "1 - (embedding <=> %s::vector) AS score "
               f"FROM chunks WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s")
        rows = self.conn.execute(sql, [v] + params + [v, top_k]).fetchall()
        return [{"id": r[0], "source_id": r[1], "source": r[2], "page": r[3],
                 "metadata": {"part": r[4], "chapter": r[5], "section_title": r[6]},
                 "text": r[7], "score": float(r[8]), "strategy": "dense"} for r in rows]


def main():
    p = argparse.ArgumentParser(description="Dense retrieval trên một split")
    p.add_argument("--chunks", required=True, type=Path)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budget", type=int, default=800)
    args = p.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in questions if q.get("split") == args.split]
    validate(chunks, selected)

    t0 = time.perf_counter()
    retriever = DenseRetriever()
    load_seconds = time.perf_counter() - t0
    traces = evaluate(retriever, selected, budget=args.budget)
    for t in traces:
        t["strategy"] = "dense"

    summary = {"strategy": "dense", "split": args.split, "n": len(traces),
               "budget_chars": args.budget,
               "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
               "context_hit": sum(t["context_hit"] for t in traces),
               "avg_context_chars": statistics.mean(t["context_chars"] for t in traces),
               "median_ms": statistics.median(t["elapsed_ms"] for t in traces),
               "model_load_seconds": load_seconds, "service_cost_usd": 0}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()