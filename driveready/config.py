"""Central configuration for DriveReady PH.

Every tunable value lives here so that the app, the scripts, and the tests
all use the same settings. Values can be overridden with environment
variables (useful when Ollama runs on another machine or port).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # source PDFs (LTO Filipino Driver's Manual)
PROCESSED_DIR = DATA_DIR / "processed"     # chunks.jsonl + vector files
EVAL_DIR = DATA_DIR / "eval"               # evaluation questions with gold pages
QUIZ_DIR = DATA_DIR / "quiz"               # curated question bank
REPORTS_DIR = ROOT / "reports"             # JSON/PNG results written by scripts
DB_PATH = Path(os.getenv("DRIVEREADY_DB", DATA_DIR / "driveready.db"))

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
DENSE_FILE = PROCESSED_DIR / "dense_{backend}.npy"   # one matrix per embedding backend
FASTTEXT_MODEL = PROCESSED_DIR / "fasttext.model"
QUESTION_BANK = QUIZ_DIR / "question_bank.json"
EVAL_SET = EVAL_DIR / "qa_eval.json"

# Official source documents (public LTO publications). The download script
# fetches them; if that fails, save the PDFs into data/raw/ by hand.
SOURCES = {
    "FDM-Vol1": {
        "title": "Filipino Driver's Manual, Volume 1 (2nd Edition)",
        "url": "https://lto.gov.ph/wp-content/uploads/2023/10/FDM-Vol.-1-2nd-Edition.pdf",
        "file": "FDM-Vol.-1-2nd-Edition.pdf",
        # PDF pages 163-200 are Chapter VI "Reviewer" (official exam questions with answers).
        # They are held out of the search index and used instead as the quiz question bank
        # and the evaluation set, so the system is tested on questions it cannot simply look up.
        "exclude_pages": list(range(163, 201)),
    },
    "FDM-Vol2": {
        "title": "Filipino Driver's Manual, Volume 2 (2nd Edition)",
        "url": "https://lto.gov.ph/wp-content/uploads/2023/10/FDM-vol.-2-2nd-Edition.pdf",
        "file": "FDM-vol.-2-2nd-Edition.pdf",
    },
}

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_WORDS = 180        # target words per chunk (about 240 tokens)
CHUNK_OVERLAP = 40       # words repeated between neighbouring chunks
MIN_CHUNK_WORDS = 25     # drop tiny fragments such as page headers

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5                # passages given to the LLM
CANDIDATES = 20          # passages each retriever returns before fusion
RRF_K = 60               # Reciprocal Rank Fusion constant (Cormack et al., 2009)
# Dense backend: "ollama" (nomic-embed-text, best quality), "lsa" (TF-IDF + SVD,
# always available offline) or "fasttext" (optional, needs gensim).
DENSE_BACKEND = os.getenv("DRIVEREADY_DENSE", "auto")

# ---------------------------------------------------------------------------
# Local LLM (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("DRIVEREADY_LLM", "llama3.1:8b")
EMBED_MODEL = os.getenv("DRIVEREADY_EMBED", "nomic-embed-text")
LLM_TEMPERATURE = 0.1    # low temperature = consistent, factual answers
LLM_TIMEOUT = 120        # seconds; first call loads the model into RAM

# ---------------------------------------------------------------------------
# Adaptive quiz (reinforcement learning)
# ---------------------------------------------------------------------------
QUIZ_TOPICS = [
    "Road signs & traffic lights",
    "Pavement markings",
    "Right of way & intersections",
    "Speed, overtaking & lanes",
    "Parking & stopping",
    "Licensing & documents",
    "Traffic laws & penalties",
    "Safe driving & emergencies",
]
BANDIT_PRIOR = (1.0, 1.0)  # Beta(alpha, beta) prior on each topic's error rate
