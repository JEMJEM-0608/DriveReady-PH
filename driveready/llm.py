"""Thin client for a local Ollama server (open-source LLMs on your own PC).

Ollama exposes a small REST API on http://localhost:11434:
  * POST /api/chat   - chat completion with a local model (e.g. llama3.1:8b)
  * POST /api/embed  - embedding vectors (e.g. nomic-embed-text)
  * GET  /api/tags   - list the models that are already downloaded
No API key and no internet connection are needed once the models are pulled.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests

from . import config


class OllamaError(RuntimeError):
    """Raised when Ollama is not running or a model is missing."""


@dataclass
class ChatResult:
    text: str
    seconds: float
    model: str


class OllamaClient:
    def __init__(self, url: str = config.OLLAMA_URL, model: str = config.LLM_MODEL,
                 embed_model: str = config.EMBED_MODEL, timeout: int = config.LLM_TIMEOUT):
        self.url = url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout

    # -- health ---------------------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"Ollama is not reachable at {self.url} ({e.__class__.__name__}).") from e
        return [m["name"] for m in r.json().get("models", [])]

    def has_model(self, name: str) -> bool:
        """True if `name` (with or without the ':latest' tag) is downloaded."""
        try:
            models = self.list_models()
        except OllamaError:
            return False
        base = name.split(":")[0]
        return any(m == name or m == f"{name}:latest" or (":" not in name and m.split(":")[0] == base)
                   for m in models)

    def available(self) -> bool:
        return self.has_model(self.model)

    # -- generation -----------------------------------------------------------
    def chat(self, messages: list[dict], temperature: float = config.LLM_TEMPERATURE,
             json_mode: bool = False) -> ChatResult:
        """Send a chat request and return the full (non-streamed) reply."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096},
        }
        if json_mode:
            payload["format"] = "json"   # Ollama constrains the output to valid JSON
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise OllamaError(f"Could not reach Ollama: {e}") from e
        if r.status_code == 404:
            raise OllamaError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return ChatResult(text=text, seconds=time.perf_counter() - t0, model=self.model)

    def chat_json(self, messages: list[dict], temperature: float = config.LLM_TEMPERATURE) -> dict:
        """Chat in JSON mode and parse the reply; raises ValueError if it is not JSON."""
        res = self.chat(messages, temperature=temperature, json_mode=True)
        try:
            return json.loads(res.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {res.text[:200]}") from e

    # -- embeddings -----------------------------------------------------------
    def embed(self, texts: list[str], batch: int = 32) -> list[list[float]]:
        """Return one embedding vector per input text."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            try:
                r = requests.post(f"{self.url}/api/embed",
                                  json={"model": self.embed_model, "input": part},
                                  timeout=self.timeout)
            except requests.RequestException as e:
                raise OllamaError(f"Could not reach Ollama: {e}") from e
            if r.status_code == 404:
                raise OllamaError(f"Embedding model '{self.embed_model}' not found. "
                                  f"Run: ollama pull {self.embed_model}")
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
        return vectors
