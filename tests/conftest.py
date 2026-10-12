"""Shared test fixtures: a tiny fake manual so tests run without PDFs or Ollama."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driveready.ingest import Chunk  # noqa: E402
from driveready.retriever import DenseIndex, HybridRetriever  # noqa: E402

TEXTS = [
    ("Traffic Signals", "A flashing red light means the driver must come to a full stop and proceed only when it is safe."),
    ("Traffic Signals", "A flashing yellow light means slow down and proceed with caution through the intersection."),
    ("Speed Limits", "On crowded streets and school zones the maximum allowable speed is 20 kph for all vehicles."),
    ("Right of Way", "At an intersection the vehicle on the left must yield to the vehicle on the right."),
    ("Pavement Markings", "A solid yellow line means overtaking is not allowed; do not cross the line."),
    ("Licensing", "A student permit holder must be accompanied by a licensed driver when driving."),
]


class FakeClient:
    """Stands in for OllamaClient in tests."""

    def __init__(self, reply: str = "", available: bool = True, json_reply: dict | None = None):
        self.reply, self._available, self.json_reply = reply, available, json_reply
        self.model, self.embed_model, self.url = "fake-llm", "fake-embed", "http://fake"
        self.calls = []

    def available(self):
        return self._available

    def has_model(self, name):
        return self._available

    def chat(self, messages, temperature=0.1, json_mode=False):
        from driveready.llm import ChatResult
        self.calls.append(messages)
        return ChatResult(text=self.reply, seconds=0.01, model=self.model)

    def chat_json(self, messages, temperature=0.1):
        self.calls.append(messages)
        return self.json_reply


@pytest.fixture
def chunks():
    return [Chunk(chunk_id=f"T-p{i + 1:03d}-c1", doc_id="T", doc_title="Test Manual", page=i + 1,
                  section=s, text=t) for i, (s, t) in enumerate(TEXTS)]


@pytest.fixture
def retriever(chunks):
    # bag-of-words dense vectors: good enough to exercise the dense code path
    vocab = sorted({w for c in chunks for w in c.text.lower().split()})

    def enc(text):
        words = text.lower().split()
        return np.array([words.count(v) for v in vocab], dtype=np.float32) + 1e-3

    matrix = np.vstack([enc(c.text) for c in chunks])
    return HybridRetriever(chunks, DenseIndex(matrix, enc), dense_name="bow")
