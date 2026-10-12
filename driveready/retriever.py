"""Hybrid retrieval: BM25 keyword search + dense (embedding) search, fused with RRF.

Why two retrievers?
  * BM25 (Lesson 1: tokens and term statistics) is excellent at exact words
    such as "60 kph", "RA 10913" or "yellow line".
  * Dense vectors (Lesson 4: word vectors) match meaning even when the
    question uses different words ("Can I use my phone at a red light?" vs.
    "Anti-Distracted Driving Act").
Reciprocal Rank Fusion (RRF) combines the two ranked lists without needing to
calibrate their very different score scales:  score(d) = sum 1 / (k + rank(d)).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .ingest import Chunk

# Small English + Filipino stop-word list; the manual is mostly English.
STOPWORDS = set("""
a an the and or of to in on at for by with from is are was were be been being it its this that
these those as if then than so such into over under up down out about can could should would
may might must shall will do does did not no yes you your yours i me my we our they their them
he she his her what which who whom when where why how all any each other some more most very
ang ng sa mga na at ay si ni kay
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps tokens like 'ra-10913', '60kph', 'r.a.'."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused score (higher is better)
    bm25_rank: int | None
    dense_rank: int | None


# ---------------------------------------------------------------------------
# Dense backends
# ---------------------------------------------------------------------------
class DenseIndex:
    """Cosine-similarity search over a matrix of L2-normalised vectors."""

    def __init__(self, matrix: np.ndarray, encode_query):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.clip(norms, 1e-9, None)
        self._encode_query = encode_query

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        q = np.asarray(self._encode_query(query), dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)
        sims = self.matrix @ q
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order]


def build_ollama_dense(chunks: list[Chunk], client) -> np.ndarray:
    """Embed every chunk with Ollama (nomic-embed-text). nomic expects task prefixes."""
    vecs = client.embed([f"search_document: {c.section}. {c.text}" for c in chunks])
    return np.asarray(vecs, dtype=np.float32)


def ollama_query_encoder(client):
    return lambda q: client.embed([f"search_query: {q}"])[0]


def gensim_available() -> bool:
    try:
        import gensim  # noqa: F401
        return True
    except ImportError:
        return False


def train_fasttext(chunks: list[Chunk], seed: int = 42):
    """Train FastText on the manual itself (same method as the course notebook,
    Part 5). Character n-grams let it handle words it never saw in training.
    Optional: needs `pip install gensim` (wheels exist for Python <= 3.12)."""
    from gensim.models import FastText

    sentences = [tokenize(c.text) for c in chunks]
    return FastText(sentences=sentences, vector_size=100, window=5, min_count=2,
                    sg=1, epochs=40, min_n=3, max_n=5, seed=seed, workers=1)


def fasttext_encoder(model):
    """Average the word vectors of a text (zero vector if it has no tokens)."""
    def encode(text: str) -> np.ndarray:
        toks = tokenize(text)
        if not toks:
            return np.zeros(model.vector_size, dtype=np.float32)
        return np.mean([model.wv[t] for t in toks], axis=0)
    return encode


class LSAModel:
    """Latent Semantic Analysis: TF-IDF matrix reduced with SVD (pure NumPy).

    A classical "vector semantics" method that needs no downloads and no extra
    packages, so it is the always-available offline fallback.
    """

    def __init__(self, texts: list[str], dim: int = 100):
        docs = [tokenize(t) for t in texts]
        vocab = sorted({w for d in docs for w in d})
        self.index = {w: i for i, w in enumerate(vocab)}
        df = np.zeros(len(vocab))
        for d in docs:
            for w in set(d):
                df[self.index[w]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
        X = np.vstack([self._tfidf(d) for d in docs])
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        self.components = vt[:min(dim, vt.shape[0])].T          # vocab x dim
        self.vector_size = self.components.shape[1]

    def _tfidf(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self.index), dtype=np.float32)
        for w in tokens:
            i = self.index.get(w)
            if i is not None:
                v[i] += 1.0
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, text: str) -> np.ndarray:
        return (self._tfidf(tokenize(text)) @ self.components).astype(np.float32)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense: DenseIndex | None = None,
                 dense_name: str = "none"):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(f"{c.section} {c.text}") for c in chunks])
        self.dense = dense
        self.dense_name = dense_name

    # individual retrievers (also used by the evaluation script) -------------
    def bm25_search(self, query: str, n: int = config.CANDIDATES) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def dense_search(self, query: str, n: int = config.CANDIDATES) -> list[tuple[int, float]]:
        return self.dense.search(query, n) if self.dense else []

    # fused search -----------------------------------------------------------
    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Hit]:
        """mode: 'hybrid' (default), 'bm25' or 'dense'."""
        bm = self.bm25_search(query) if mode in ("hybrid", "bm25") else []
        de = self.dense_search(query) if mode in ("hybrid", "dense") else []
        scores: dict[int, float] = {}
        bm_rank = {idx: r for r, (idx, _) in enumerate(bm, start=1)}
        de_rank = {idx: r for r, (idx, _) in enumerate(de, start=1)}
        for idx, r in bm_rank.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (config.RRF_K + r)
        for idx, r in de_rank.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (config.RRF_K + r)
        order = sorted(scores, key=scores.get, reverse=True)[:k]
        return [Hit(self.chunks[i], scores[i], bm_rank.get(i), de_rank.get(i)) for i in order]
