"""Build the committed fixture that checks retrieval against the RAG repo's own rankings.

    uv sync --extra embed
    uv run python scripts/build_retrieval_fixture.py --rag ../rag-support-assistant

Reads the RAG repo's 200 questions and, for each, the top 8 chunks its chosen
hybrid config froze for generation (`results/contexts/`), plus the top 10
articles its hybrid, BM25-only and dense-only runs ranked on test
(`results/retrieval/test/`). Embeds each question once with bge-small at the
snapshot's pinned revision (downloads the 33M model on first use) and writes
tests/fixtures/rag_equivalence/. The tests then need no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from support_triage_agents.retrieval import BgeSmallEncoder, load_config
from support_triage_agents.vendor import RAG_SNAPSHOT, REPO_ROOT, git_output

OUT = REPO_ROOT / "tests" / "fixtures" / "rag_equivalence"
CONTEXTS = "results/contexts/fixed-title-hybrid-bge-small-convex0.7.jsonl"
TOP_ARTICLES = {
    "hybrid": "results/retrieval/test/fixed-title-hybrid-bge-small-convex0.7.jsonl",
    "bm25": "results/retrieval/test/fixed-title-bm25.jsonl",
    "dense": "results/retrieval/test/fixed-title-dense-bge-small.jsonl",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rag", type=Path, required=True, help="rag-support-assistant checkout")
    args = parser.parse_args(argv)
    head = git_output(args.rag, "rev-parse", "HEAD")
    if head != RAG_SNAPSHOT.pinned_commit:
        print(f"{args.rag} is at {head}, expected {RAG_SNAPSHOT.pinned_commit}", file=sys.stderr)
        return 1
    questions = {}
    for split in ("dev", "test"):
        for row in _jsonl(args.rag / "data" / "tallowbrook" / f"questions_{split}.jsonl"):
            questions[row["question_id"]] = row["question"]
    contexts = {row["question_id"]: row for row in _jsonl(args.rag / CONTEXTS)}
    tops: dict[str, dict[str, list[str]]] = {}
    for name, rel in TOP_ARTICLES.items():
        for rec in _jsonl(args.rag / rel):
            tops.setdefault(rec["item_id"], {})[name] = rec["meta"]["top_articles"]
    encoder = BgeSmallEncoder(load_config())
    rows, vectors = [], []
    for qid in sorted(contexts):
        rows.append(
            {
                "question_id": qid,
                "question": questions[qid],
                "top8_chunks": [c["chunk_id"] for c in contexts[qid]["chunks"]],
                "top10_articles": tops.get(qid, {}),
            }
        )
        vectors.append(encoder.encode(questions[qid]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.save(OUT / "query_vectors.npy", np.stack(vectors).astype(np.float32))
    meta = {
        "rag_commit": head,
        "n_questions": len(rows),
        "files": [CONTEXTS, *TOP_ARTICLES.values()],
    }
    (OUT / "SOURCE.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} questions to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
