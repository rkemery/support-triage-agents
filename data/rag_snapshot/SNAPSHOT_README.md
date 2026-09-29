# Retrieval snapshot

A frozen copy of the retrieval index for other repos to vendor, written by `make export-snapshot`.

- `chunks.jsonl`: one chunk per line, in the row order of the embeddings. `text` is what an answer model should see, `embed_text` is exactly what the encoders saw (the article title, then the chunk).
- `embeddings.npy`: float32, one L2-normalized `BAAI/bge-small-en-v1.5` vector per chunk (revision in `config.json`). Embed queries with the same model and revision, prefixed with `query_prompt`, and rank by dot product.
- `config.json`: chunking, model and revision, query prompt, BM25 settings (to rebuild BM25 over `embed_text`), the retrieval config chosen on dev, and the Tallowbrook dataset commit.
- `MANIFEST.json`: sha256 of the three files above.

The chunking is the one chosen on dev. The embedding model is bge-small whichever model won, so a consumer needs only that 33M model. On the RAG test split, dense retrieval with this chunking and model reaches nDCG@10 0.782, and hybrid convex fusion with BM25 (alpha 0.7 on dense) 0.811. See the main README for intervals.

Data license: CC-BY-4.0 (Tallowbrook dataset).
