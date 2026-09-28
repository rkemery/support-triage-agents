# RAG retrieval snapshot (vendored)

A pinned copy of the frozen retrieval index exported by `rag-support-assistant` (commit in `MANIFEST.json`). The Researcher agent searches it. `SNAPSHOT_README.md` is the exporter's own description.

- `chunks.jsonl`: 367 chunks of the 151 help-center articles, in embedding row order.
- `embeddings.npy`: one L2-normalized `BAAI/bge-small-en-v1.5` vector per chunk.
- `config.json`: chunking, model revision, query prompt, BM25 settings and the retrieval config chosen on the RAG dev split (hybrid, convex fusion, alpha 0.7 on dense).

Chunk text comes from the Tallowbrook dataset, CC-BY-4.0.

`uv run python scripts/sync_data.py` checks every file against its sha256.
