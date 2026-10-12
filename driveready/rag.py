"""Retrieval-Augmented Generation: question -> passages -> grounded, cited answer.

Pipeline
  1. Retrieve the top-k passages from the driver's manual (hybrid search).
  2. Put them, numbered [1]..[k], into a prompt with strict rules.
  3. The local LLM answers ONLY from those passages and cites them.
  4. A validator keeps only citation numbers that really exist and flags
     answers that cite nothing.
If Ollama is not running, the system still answers with the best passages
(extractive mode), so the app never breaks during a demo.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import config
from .llm import OllamaClient, OllamaError
from .retriever import Hit, HybridRetriever

NOT_FOUND = "I could not find this in the LTO driver's manual."

SYSTEM_PROMPT = f"""You are DriveReady PH, a friendly reviewer for the Philippine LTO driver's license exam.
Rules:
1. Answer ONLY with facts from the numbered manual passages provided. Never use outside knowledge.
2. After each fact, cite its passage number in square brackets, e.g. [2].
3. If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
4. Keep the answer short: 2 to 5 sentences, in simple language.
   Reply in the language named in the "Answer language" line of the user message.
5. Do not invent fines, speeds, or law numbers that are not written in the passages."""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)   # 1-based passage numbers actually cited
    mode: str = "llm"                                 # "llm" or "extractive"
    grounded: bool = True                             # False if the LLM cited nothing
    timings: dict = field(default_factory=dict)       # retrieval_s, llm_s, total_s
    model: str = ""

    def sources(self) -> list[Hit]:
        """Passages to show under the answer (cited ones first)."""
        if self.cited:
            return [self.hits[i - 1] for i in self.cited]
        return self.hits


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        head = f"[{i}] ({c.doc_title}, page {c.page}{', ' + c.section if c.section else ''})"
        blocks.append(f"{head}\n{c.text}")
    return "\n\n".join(blocks)


_FILIPINO_WORDS = {"ang", "ng", "mga", "sa", "ano", "paano", "bakit", "kailan", "saan", "ba", "po", "ko",
                   "ako", "mo", "ka", "kung", "dapat", "pwede", "puwede", "hindi", "may", "nang", "lang"}


def detect_language(text: str) -> str:
    """Very small heuristic: 'Filipino' if the question uses common Tagalog function words.

    Llama 3.1 tends to switch to Filipino when it sees Philippine context, so the
    reply language is stated explicitly in every prompt (English by default)."""
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in _FILIPINO_WORDS for w in words)
    return "Filipino" if words and hits >= max(2, len(words) // 6) else "English"


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:          # short memory for follow-up questions
        msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content":
                 f"Manual passages:\n\n{format_context(hits)}\n\n"
                 f"Answer language: {detect_language(question)}\nQuestion: {question}"})
    return msgs


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(text: str, n_passages: int) -> list[int]:
    """Return the valid, unique passage numbers cited in `text`, in order."""
    found: list[int] = []
    for group in _CITE.findall(text):
        for num in group.split(","):
            k = int(num)
            if 1 <= k <= n_passages and k not in found:
                found.append(k)
    return found


def strip_invalid_citations(text: str, n_passages: int) -> str:
    """Remove citation numbers that do not match any passage (e.g. [9] with 5 passages)."""
    def fix(m: re.Match) -> str:
        keep = [n.strip() for n in m.group(1).split(",") if 1 <= int(n) <= n_passages]
        return f"[{', '.join(keep)}]" if keep else ""
    return _CITE.sub(fix, text)


def extractive_answer(question: str, hits: list[Hit]) -> str:
    """Fallback when no LLM is available: quote the best passage."""
    if not hits:
        return NOT_FOUND
    best = hits[0].chunk
    sentences = re.split(r"(?<=[.!?])\s+", best.text)
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    ranked = sorted(sentences, key=lambda s: -len(q_words & set(re.findall(r"[a-z0-9]+", s.lower()))))
    snippet = " ".join(ranked[:2]).strip()
    return f"(Offline mode - the LLM is not running.) Most relevant passage: \"{snippet}\" [1]"


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, client: OllamaClient | None = None,
                 top_k: int = config.TOP_K):
        self.retriever = retriever
        self.client = client or OllamaClient()
        self.top_k = top_k

    def ask(self, question: str, history: list[dict] | None = None,
            mode: str = "hybrid", use_llm: bool = True) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=self.top_k, mode=mode)
        t_ret = time.perf_counter() - t0
        if not use_llm or not self.client.available():
            text = extractive_answer(question, hits)
            return Answer(question, text, hits, cited=[1] if hits else [], mode="extractive",
                          timings={"retrieval_s": t_ret, "llm_s": 0.0,
                                   "total_s": time.perf_counter() - t0})
        try:
            res = self.client.chat(build_messages(question, hits, history))
        except OllamaError as e:
            text = extractive_answer(question, hits) + f"\n\n_LLM error: {e}_"
            return Answer(question, text, hits, cited=[1] if hits else [], mode="extractive",
                          timings={"retrieval_s": t_ret, "llm_s": 0.0,
                                   "total_s": time.perf_counter() - t0})
        text = strip_invalid_citations(res.text.strip(), len(hits))
        cited = extract_citations(text, len(hits))
        grounded = bool(cited) or NOT_FOUND.lower()[:30] in text.lower()
        return Answer(question, text, hits, cited=cited, mode="llm", grounded=grounded,
                      timings={"retrieval_s": t_ret, "llm_s": res.seconds,
                               "total_s": time.perf_counter() - t0},
                      model=res.model)
