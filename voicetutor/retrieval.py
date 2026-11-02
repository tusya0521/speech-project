"""Hybrid retrieval: BM25 (keywords) + dense embeddings (meaning), fused with reciprocal rank fusion."""
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .ingest import Chunk, tokenize


@lru_cache(maxsize=1)
def _embedder():
    # The model is shared (read-only); each session's index holds its own vectors.
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=config.EMBED_MODEL)


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    model = _embedder()
    vecs = list(model.query_embed(texts) if query else model.passage_embed(texts))
    m = np.array(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-9)


@dataclass
class Hit:
    chunk: Chunk
    score: float      # fused RRF score (for ranking)
    cosine: float     # dense similarity (for confidence)


class Index:
    """In-memory index for one session's document."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])
        self.vectors = embed([c.text for c in chunks])

    def search(self, query: str, k: int = None) -> tuple[list[Hit], float]:
        """Return (top-k hits, confidence). Confidence = best cosine similarity in [0, 1]."""
        k = k or config.TOP_K
        if not self.chunks or not query.strip():
            return [], 0.0
        # Whisper often hyphenates terms ("tri-phone"); also try the joined spelling and keep the better match.
        variants = list(dict.fromkeys([query, re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", "", query)]))
        sims = self.vectors @ embed(variants, query=True).T
        best = int(sims.max(axis=0).argmax())
        query, cos = variants[best], sims[:, best]
        bm = self.bm25.get_scores(tokenize(query))
        fused = rrf([np.argsort(-cos), np.argsort(-bm)], config.RRF_K)
        hits = [Hit(self.chunks[i], s, float(cos[i])) for i, s in fused[:k]]
        return hits, float(cos.max())

    def neighbours(self, chunk: Chunk, n: int) -> list[Chunk]:
        """The chunk plus following chunks on the same or next page (for 'read the section on X')."""
        out = [chunk]
        for c in self.chunks[chunk.idx + 1:]:
            if len(out) >= n or c.page > chunk.page + 1:
                break
            out.append(c)
        return out


def rrf(rankings: list, k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
