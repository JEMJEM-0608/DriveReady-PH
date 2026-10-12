"""Build the knowledge base: read PDFs -> chunks -> BM25 + dense vectors.

Usage:  python scripts/build_index.py
Writes data/processed/chunks.jsonl, dense_fasttext.npy and (if Ollama has
nomic-embed-text) dense_ollama.npy, plus reports/kb_stats.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from driveready import config  # noqa: E402
from driveready.ingest import load_chunks  # noqa: E402
from driveready.knowledge import rebuild_all  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    info = rebuild_all()
    chunks = load_chunks()
    lengths = [len(c.text.split()) for c in chunks]
    stats = {
        **info,
        "build_seconds": round(time.perf_counter() - t0, 2),
        "per_document": {d: sum(1 for c in chunks if c.doc_id == d) for d in config.SOURCES},
        "pages_with_text": len({(c.doc_id, c.page) for c in chunks}),
        "words_per_chunk": {"mean": round(statistics.mean(lengths), 1), "min": min(lengths),
                            "max": max(lengths)},
        "chunk_words_setting": config.CHUNK_WORDS, "overlap_setting": config.CHUNK_OVERLAP,
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "kb_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
