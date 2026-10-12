"""Build and load the searchable knowledge base (chunks + vectors).

Dense backend selection ("auto"):
  1. Ollama nomic-embed-text, if Ollama is running and the model is pulled.
  2. LSA (TF-IDF + SVD in NumPy): offline fallback, no downloads or extra packages.
  FastText (gensim) is an optional extra backend used in the evaluation.
Vectors are cached in data/processed/ so the app starts instantly.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from . import config
from .ingest import Chunk, build_chunks, load_chunks, save_chunks
from .llm import OllamaClient
from .retriever import (
    DenseIndex,
    HybridRetriever,
    LSAModel,
    build_ollama_dense,
    fasttext_encoder,
    gensim_available,
    ollama_query_encoder,
    train_fasttext,
)


def kb_fingerprint(chunks: list[Chunk]) -> str:
    """Short SHA-256 of all chunk ids + texts; shown in the app and in reports."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:12]


def _meta_path(backend: str):
    return config.PROCESSED_DIR / f"dense_{backend}.json"


def _cache_valid(backend: str, fp: str) -> bool:
    npy = config.PROCESSED_DIR / f"dense_{backend}.npy"
    meta = _meta_path(backend)
    if not (npy.exists() and meta.exists()):
        return False
    return json.loads(meta.read_text()).get("fingerprint") == fp


def _write_meta(backend: str, fp: str, extra: dict | None = None) -> None:
    _meta_path(backend).write_text(json.dumps({"fingerprint": fp, **(extra or {})}, indent=2))


def build_dense(chunks: list[Chunk], backend: str, client: OllamaClient | None = None) -> DenseIndex:
    """Build (or load from cache) the dense index for one backend."""
    fp = kb_fingerprint(chunks)
    npy = config.PROCESSED_DIR / f"dense_{backend}.npy"
    if backend == "ollama":
        client = client or OllamaClient()
        if not _cache_valid("ollama", fp):
            matrix = build_ollama_dense(chunks, client)
            np.save(npy, matrix)
            _write_meta("ollama", fp, {"model": client.embed_model, "dim": int(matrix.shape[1])})
        return DenseIndex(np.load(npy), ollama_query_encoder(client))
    if backend == "fasttext":
        from gensim.models import FastText

        if _cache_valid("fasttext", fp) and config.FASTTEXT_MODEL.exists():
            model = FastText.load(str(config.FASTTEXT_MODEL))
        else:
            model = train_fasttext(chunks)
            model.save(str(config.FASTTEXT_MODEL))
            enc = fasttext_encoder(model)
            np.save(npy, np.vstack([enc(f"{c.section} {c.text}") for c in chunks]).astype(np.float32))
            _write_meta("fasttext", fp, {"model": "gensim FastText (trained on manual)",
                                         "dim": model.vector_size})
        return DenseIndex(np.load(npy), fasttext_encoder(model))
    if backend == "lsa":
        model = LSAModel([f"{c.section} {c.text}" for c in chunks])   # fast (< 1 s), so never cached
        return DenseIndex(np.vstack([model.encode(f"{c.section} {c.text}") for c in chunks]), model.encode)
    raise ValueError(f"unknown dense backend: {backend}")


def resolve_backend(client: OllamaClient | None = None) -> str:
    if config.DENSE_BACKEND != "auto":
        return config.DENSE_BACKEND
    client = client or OllamaClient()
    return "ollama" if client.has_model(client.embed_model) else "lsa"


def load_retriever(backend: str | None = None, client: OllamaClient | None = None) -> HybridRetriever:
    """Load chunks and the chosen dense backend; falls back to LSA on error."""
    chunks = load_chunks()
    backend = backend or resolve_backend(client)
    try:
        dense = build_dense(chunks, backend, client)
    except Exception as e:  # Ollama stopped mid-way, etc.
        if backend == "lsa":
            raise
        print(f"[knowledge] {backend} backend failed ({e}); using lsa")
        backend, dense = "lsa", build_dense(chunks, "lsa")
    return HybridRetriever(chunks, dense, dense_name=backend)


def rebuild_all(client: OllamaClient | None = None) -> dict:
    """Re-read the PDFs, re-chunk, and rebuild every available dense index."""
    chunks = build_chunks()
    if not chunks:
        raise RuntimeError("No PDFs found in data/raw/. Run: python scripts/download_docs.py")
    save_chunks(chunks)
    built = ["lsa"]
    build_dense(chunks, "lsa")
    if gensim_available():
        build_dense(chunks, "fasttext")
        built.append("fasttext")
    client = client or OllamaClient()
    if client.has_model(client.embed_model):
        build_dense(chunks, "ollama", client)
        built.append("ollama")
    return {"chunks": len(chunks), "fingerprint": kb_fingerprint(chunks), "dense_backends": built}
