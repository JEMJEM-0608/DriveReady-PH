"""Document ingestion: PDF -> clean page text -> overlapping chunks.

Each chunk keeps its source document and page number so that every answer
the system gives can cite exactly where the information came from.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config


@dataclass
class Chunk:
    """One searchable passage of the knowledge base."""

    chunk_id: str     # e.g. "FDM-Vol1-p023-c1"
    doc_id: str       # key in config.SOURCES
    doc_title: str
    page: int         # 1-based page number in the PDF
    section: str      # nearest heading found on the page (best effort)
    text: str

    def citation(self) -> str:
        return f"{self.doc_title}, p. {self.page}"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")      # "inter-\nsection" -> "intersection"
_MULTISPACE = re.compile(r"[ \t ]+")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d{1,3}\s*$", re.IGNORECASE)


def clean_page_text(text: str) -> str:
    """Normalise raw PDF text: join hyphenated words, drop page-number lines,
    collapse whitespace while keeping paragraph breaks."""
    text = text.replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = []
    for line in text.split("\n"):
        line = _MULTISPACE.sub(" ", line).strip()
        if not line or _PAGE_NUMBER_LINE.match(line):
            lines.append("")
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_section(text: str) -> str:
    """Return the first short line written in capitals or Title Case.

    Driver's-manual pages usually start with a heading such as
    "RIGHT OF WAY" or "Pavement Markings"; this is good enough for display.
    """
    for line in text.split("\n")[:8]:
        words = line.split()
        if 1 <= len(words) <= 9 and len(line) <= 70:
            letters = [c for c in line if c.isalpha()]
            if letters and (sum(c.isupper() for c in letters) / len(letters) > 0.7 or line.istitle()):
                return line.title() if line.isupper() else line
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of `size` words with `overlap` shared words."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    pieces = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        pieces.append(" ".join(piece))
        if start + size >= len(words):
            break
    return pieces


def chunk_pages(doc_id: str, pages: list[str]) -> list[Chunk]:
    """Turn the cleaned pages of one document into Chunk objects."""
    title = config.SOURCES.get(doc_id, {}).get("title", doc_id)
    chunks: list[Chunk] = []
    last_section = ""
    for page_no, raw in enumerate(pages, start=1):
        text = clean_page_text(raw)
        if not text:
            continue
        section = guess_section(text) or last_section
        last_section = section
        for i, piece in enumerate(split_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)):
            if len(piece.split()) < config.MIN_CHUNK_WORDS:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}-p{page_no:03d}-c{i + 1}",
                doc_id=doc_id, doc_title=title, page=page_no,
                section=section, text=piece,
            ))
    return chunks


def read_pdf_pages(path: Path) -> list[str]:
    """Extract the text of every page (empty string for image-only pages)."""
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def build_chunks(raw_dir: Path = config.RAW_DIR) -> list[Chunk]:
    """Read every known source PDF found in raw_dir and chunk it."""
    chunks: list[Chunk] = []
    for doc_id, meta in config.SOURCES.items():
        path = raw_dir / meta["file"]
        if not path.exists():
            print(f"[ingest] missing {path.name} - skipped (run scripts/download_docs.py)")
            continue
        pages = read_pdf_pages(path)
        for p in meta.get("exclude_pages", []):        # held-out pages (see config.py)
            if 1 <= p <= len(pages):
                pages[p - 1] = ""
        doc_chunks = chunk_pages(doc_id, pages)
        print(f"[ingest] {doc_id}: {len(pages)} pages -> {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    return chunks


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_chunks(chunks: list[Chunk], path: Path = config.CHUNKS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path = config.CHUNKS_FILE) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the knowledge base first: python scripts/build_index.py")
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
