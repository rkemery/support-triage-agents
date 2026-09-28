"""Retrieval must reproduce the RAG repo's rankings from the same snapshot.

The fixture holds bge-small query vectors for the RAG repo's 200 questions,
the top 8 chunks its chosen hybrid config froze for each, and the top 10
articles its hybrid, BM25-only and dense runs ranked for the 130 scored test
questions (scripts/build_retrieval_fixture.py). No model is needed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from support_triage_agents.retrieval import (
    BM25,
    CachedQueryEncoder,
    HybridRetriever,
    QueryEmbeddingMiss,
    build_retriever,
    load_config,
    tokenize,
)

FIXTURE = Path(__file__).parent / "fixtures" / "rag_equivalence"


class FixtureEncoder:
    def __init__(self) -> None:
        rows = [json.loads(line) for line in (FIXTURE / "questions.jsonl").open()]
        vectors = np.load(FIXTURE / "query_vectors.npy")
        self.rows = rows
        self._by_text = {row["question"]: vectors[i] for i, row in enumerate(rows)}

    def encode(self, query: str) -> np.ndarray:
        return self._by_text[query]


@pytest.fixture(scope="module")
def setup() -> tuple[HybridRetriever, FixtureEncoder]:
    encoder = FixtureEncoder()
    return build_retriever(encoder), encoder


def _articles(chunk_ids: list[int], retriever: HybridRetriever) -> list[str]:
    seen: dict[str, None] = {}
    for i in chunk_ids:
        seen.setdefault(retriever.chunks[i].article_id, None)
    return list(seen)[:10]


def test_hybrid_top8_chunks_match_the_rag_repo(setup):
    retriever, encoder = setup
    exact = same_set = 0
    for row in encoder.rows:
        top8 = [h.chunk.chunk_id for h in retriever.search(row["question"], top_k=8)]
        exact += top8 == row["top8_chunks"]
        same_set += set(top8) == set(row["top8_chunks"])
    assert len(encoder.rows) == 200
    assert same_set == 200
    # One question swaps two chunks whose scores tie to float precision.
    assert exact >= 199


def test_hybrid_and_dense_article_rankings_match_on_test(setup):
    retriever, encoder = setup
    scored = [row for row in encoder.rows if "hybrid" in row["top10_articles"]]
    assert len(scored) == 130
    exact = {"hybrid": 0, "dense": 0}
    for row in scored:
        q, gold = row["question"], row["top10_articles"]
        hybrid = [retriever.chunks.index(h.chunk) for h in retriever.search(q, top_k=50)]
        dense = retriever._dense(q)
        ranked = {"hybrid": hybrid, "dense": sorted(dense, key=lambda i: -dense[i])}
        for name, order in ranked.items():
            mine = _articles(order, retriever)
            assert mine[:3] == gold[name][:3], (name, row["question_id"])
            assert set(mine[:9]) <= set(gold[name]), (name, row["question_id"])
            exact[name] += mine == gold[name]
    # The one difference in each is a v1/v2 pair of near-identical articles at rank 9 or 10.
    assert exact == {"hybrid": 129, "dense": 129}


def test_bm25_scores_rank_the_same_articles(setup):
    """BM25 alone ties often (a v1 and v2 article can share every query term), and Qdrant
    breaks ties in its own order, so this checks the top article and the top-10 set."""
    retriever, encoder = setup
    scored = [row for row in encoder.rows if "bm25" in row["top10_articles"]]
    same_top1 = same_set = 0
    for row in scored:
        sparse = retriever._sparse(row["question"])
        mine = _articles(list(sparse), retriever)
        gold = row["top10_articles"]["bm25"]
        same_top1 += mine[0] == gold[0]
        same_set += set(mine) == set(gold)
    assert same_top1 == 130
    assert same_set >= 120


def test_tokenizer_matches_the_rag_repo_rules():
    assert tokenize("The Plus fee is $2.50, isn't it?") == ["plu", "fee", "2.50"]


def test_bm25_prefers_documents_with_rarer_query_terms():
    bm25 = BM25(["card fee", "card replacement", "card card card"])
    scores = bm25.scores("replacement card")
    assert int(np.argmax(scores)) == 1


def test_cached_encoder_replays_without_a_model(tmp_path):
    config = load_config()
    calls = []

    class Inner:
        def encode(self, query: str) -> np.ndarray:
            calls.append(query)
            return np.ones(384, dtype=np.float32) / np.sqrt(384)

    path = tmp_path / "q.jsonl"
    live = CachedQueryEncoder(path, config, Inner())
    first = live.encode("lost card")
    live.encode("lost card")
    assert calls == ["lost card"]
    replay = CachedQueryEncoder(path, config, None)
    assert np.array_equal(replay.encode("lost card"), first)
    with pytest.raises(QueryEmbeddingMiss):
        replay.encode("never seen")


@pytest.mark.download
def test_bge_encoder_reproduces_the_fixture_vectors():
    from support_triage_agents.retrieval import BgeSmallEncoder

    encoder = FixtureEncoder()
    bge = BgeSmallEncoder(load_config())
    for row in encoder.rows[:5]:
        assert np.allclose(bge.encode(row["question"]), encoder.encode(row["question"]), atol=1e-5)
