"""Xuất chunk B và benchmark từ Day 06 sang JSON array cho convert_day6.py."""
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OUT = ROOT / "starter" / "data" / "day6_export"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Chunk chiến lược B từ pgvector của Day 06
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    rows = conn.execute(
        "SELECT chunk_id, document_id, source, page, part, chapter, section_title, text "
        "FROM chunks WHERE strategy = 'B' ORDER BY chunk_id"
    ).fetchall()

chunks = [
    {
        "chunk_id": r[0],
        "document_id": r[1],
        "source": r[2],
        "page": r[3],
        "text": r[7],
        "metadata": {"part": r[4], "chapter": r[5], "section_title": r[6]},
    }
    for r in rows
]

# 2. Câu hỏi từ benchmark/questions.jsonl (mỗi dòng là một câu)
qpath = ROOT / "benchmark" / "questions.jsonl"
questions = [
    json.loads(line)
    for line in qpath.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

(OUT / "day6_chunks.json").write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "day6_questions.json").write_text(
    json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"{len(chunks)} chunk, {len(questions)} câu")