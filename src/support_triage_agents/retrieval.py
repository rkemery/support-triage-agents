"""Hybrid search over the vendored RAG snapshot, reproducing the RAG repo's chosen config.

The RAG repo runs this search through LlamaIndex and Qdrant. This module does
the same arithmetic with numpy over the frozen snapshot, so the agents repo
needs neither:

- Dense: dot product of L2-normalized `bge-small-en-v1.5` vectors (the query
  gets the model's query prompt), top 50 chunks.
- BM25: k1 = 1.2, b = 0.75, IDF ln(1 + (N - df + 0.5) / (df + 0.5)), over the
  chunks' `embed_text`. Tokens are lowercased words and numbers, NLTK English
  stopwords dropped, Porter-stemmed. A query term counts once per occurrence.
  Top 50 chunks with a nonzero score (Qdrant's sparse search returns only
  those).
- Convex fusion, as LlamaIndex's `relative_score_fusion`: min-max normalize
  each list over what it returned, a chunk missing from a list scores 0
  there, fused = alpha * dense + (1 - alpha) * bm25 with alpha 0.7 from the
  RAG dev split.

`tests/test_retrieval.py` checks the result against the RAG repo's own
rankings for its 200 questions, from committed query vectors.

Query vectors come from a `QueryEncoder`. Live runs embed new queries with
bge-small on CPU (the `embed` extra). Every vector is also written to a
committed cache, so a replay or the offline demo needs no model.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from nltk.stem import PorterStemmer

from support_triage_agents.data import SNAPSHOT_DIR

FIRST_STAGE_DEPTH = 50
TOKEN_PATTERN = r"[a-z0-9]+(?:[.,:][0-9]+)*"
_TOKEN = re.compile(TOKEN_PATTERN)
# NLTK's 179-word English stopword list, inlined (the same list the RAG repo inlines).
STOPWORDS = frozenset(
    """
    i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself
    yourselves he him his himself she she's her hers herself it it's its itself they them
    their theirs themselves what which who whom this that that'll these those am is are was
    were be been being have has had having do does did doing a an the and but if or because as
    until while of at by for with about against between into through during before after above
    below to from up down in out on off over under again further then once here there when
    where why how all any both each few more most other some such no nor not only own same so
    than too very s t can will just don don't should should've now d ll m o re ve y ain aren
    aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn
    isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn
    wasn't weren weren't won won't wouldn wouldn't
    """.split()  # noqa: SIM905 (a word list reads better as text)
)
_stemmer = PorterStemmer()


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower().replace("\u2019", "'"))
    return [_stem(w) for w in words if w not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article_id: str
    title: str
    effective_date: str
    chunk_index: int
    text: str
    embed_text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25:
    """BM25 scores of one query against every chunk, as Qdrant's sparse dot product gives them."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        if not texts:
            raise ValueError("cannot fit BM25 on an empty corpus")
        docs = [tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / n
        idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        # term -> {doc index: weight}
        self._postings: dict[str, dict[int, float]] = {}
        for i, tokens in enumerate(docs):
            norm = k1 * (1 - b + b * len(tokens) / avgdl)
            for term, tf in Counter(tokens).items():
                weight = idf[term] * tf * (k1 + 1) / (tf + norm)
                self._postings.setdefault(term, {})[i] = weight
        self.n_docs = n

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term, count in Counter(tokenize(query)).items():
            for i, weight in self._postings.get(term, {}).items():
                out[i] += count * weight
        return out


class QueryEncoder(Protocol):
    def encode(self, query: str) -> np.ndarray: ...


class QueryEmbeddingMiss(LookupError):
    """No cached vector for a query and no model to embed it with."""


def _vec_to_b64(vector: np.ndarray) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def _b64_to_vec(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32).copy()


class CachedQueryEncoder:
    """Query vectors from a JSONL cache, keyed by model, revision, prompt and query text.

    With `inner=None` a miss raises `QueryEmbeddingMiss`, which is how replays
    and the demo run with no model installed. New vectors are appended to the
    file, one line each, so the cache can be committed and diffed.
    """

    def __init__(self, path: Path, config: dict[str, Any], inner: QueryEncoder | None) -> None:
        self.path = path
        self._model_id = (
            f"{config['embedding_model']}@{config['embedding_revision']}|{config['query_prompt']}"
        )
        self._inner = inner
        self._vectors: dict[str, np.ndarray] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._vectors[row["key"]] = _b64_to_vec(row["vector"])

    def key(self, query: str) -> str:
        return hashlib.sha256(f"{self._model_id}\n{query}".encode()).hexdigest()

    def encode(self, query: str) -> np.ndarray:
        key = self.key(query)
        if key in self._vectors:
            return self._vectors[key]
        if self._inner is None:
            raise QueryEmbeddingMiss(
                f"no cached embedding for query {query[:60]!r} in {self.path}. Embedding it "
                "needs the bge-small model: uv sync --extra embed, then run live."
            )
        vector = np.asarray(self._inner.encode(query), dtype=np.float32)
        self._vectors[key] = vector
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            row = {"key": key, "query": query, "vector": _vec_to_b64(vector)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return vector


class BgeSmallEncoder:
    """bge-small-en-v1.5 at the snapshot's pinned revision, on CPU. Needs the `embed` extra."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "embedding new queries needs sentence-transformers: uv sync --extra embed"
            ) from exc
        self._prompt = config["query_prompt"]
        self._model = SentenceTransformer(
            config["embedding_model"], revision=config["embedding_revision"], device="cpu"
        )

    def encode(self, query: str) -> np.ndarray:
        vector = self._model.encode(
            [query],
            prompt=self._prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.astype(np.float32)


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    hi, lo = max(scores.values()), min(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, hi)
    return {i: (s - lo) / (hi - lo) for i, s in scores.items()}


class HybridRetriever:
    """Dense plus BM25 with convex fusion over the snapshot's chunks."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        config: dict[str, Any],
        encoder: QueryEncoder,
        depth: int = FIRST_STAGE_DEPTH,
    ) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {embeddings.shape[0]} vectors")
        chosen = config["dev_chosen_config"]
        if chosen["mode"] != "hybrid" or chosen["fusion"] != "convex" or chosen["reranker"]:
            raise ValueError(f"snapshot config {chosen} is not hybrid convex without a reranker")
        self.chunks = list(chunks)
        self.embeddings = embeddings.astype(np.float32)
        self.alpha = float(chosen["alpha"])
        bm25 = config["bm25"]
        self.bm25 = BM25([c.embed_text for c in chunks], k1=bm25["k1"], b=bm25["b"])
        self.encoder = encoder
        self.depth = depth

    def _dense(self, query: str) -> dict[int, float]:
        q = np.asarray(self.encoder.encode(query), dtype=np.float32)
        sims = self.embeddings @ q
        top = np.argsort(-sims, kind="stable")[: self.depth]
        return {int(i): float(sims[i]) for i in top}

    def _sparse(self, query: str) -> dict[int, float]:
        scores = self.bm25.scores(query)
        top = [int(i) for i in np.argsort(-scores, kind="stable")[: self.depth] if scores[i] > 0]
        return {i: float(scores[i]) for i in top}

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        dense = self._dense(query)
        sparse = self._sparse(query)
        if not sparse:
            order = list(dense.items())
        else:
            d_norm, s_norm = _minmax(dense), _minmax(sparse)
            candidates = list(dense) + [i for i in sparse if i not in dense]
            order = [
                (i, self.alpha * d_norm.get(i, 0.0) + (1 - self.alpha) * s_norm.get(i, 0.0))
                for i in candidates
            ]
            order.sort(key=lambda pair: pair[1], reverse=True)
        return [Hit(self.chunks[i], score) for i, score in order[:top_k]]


def load_chunks(snapshot_dir: Path = SNAPSHOT_DIR) -> list[Chunk]:
    with (snapshot_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def load_config(snapshot_dir: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    return json.loads((snapshot_dir / "config.json").read_text(encoding="utf-8"))


def build_retriever(encoder: QueryEncoder, snapshot_dir: Path = SNAPSHOT_DIR) -> HybridRetriever:
    config = load_config(snapshot_dir)
    embeddings = np.load(snapshot_dir / "embeddings.npy")
    return HybridRetriever(load_chunks(snapshot_dir), embeddings, config, encoder)
