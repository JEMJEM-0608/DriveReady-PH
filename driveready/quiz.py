"""Quiz questions: a curated question bank plus RAG-generated questions.

* Question bank (data/quiz/question_bank.json): hand-checked multiple-choice
  questions, each tagged with a topic and the manual page it comes from.
* Generated questions: the topic is turned into a retrieval query, the best
  manual passage is given to the LLM, and the LLM writes a new 4-choice
  question in JSON. The JSON is validated before it is shown; if anything is
  wrong (or Ollama is offline) the engine falls back to the question bank.
The topic itself is chosen by the reinforcement-learning bandit (bandit.py).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from . import config
from .llm import OllamaClient, OllamaError
from .retriever import HybridRetriever

# Retrieval queries used to find source passages for each topic.
TOPIC_QUERIES = {
    "Road signs & traffic lights": "regulatory warning informative road signs traffic light red yellow green arrow",
    "Pavement markings": "pavement markings solid broken yellow white lines yellow box",
    "Right of way & intersections": "right of way intersection yield pedestrians roundabout emergency vehicles",
    "Speed, overtaking & lanes": "speed limit kph overtaking passing lane changing expressway",
    "Parking & stopping": "parking prohibited stopping fire hydrant intersection uphill downhill curb",
    "Licensing & documents": "student permit driver's license requirements validity renewal registration",
    "Traffic laws & penalties": "violation penalty fine seatbelt helmet distracted driving drunk driving",
    "Safe driving & emergencies": "defensive driving following distance fatigue rain road crash early warning device",
}


@dataclass
class Question:
    qid: str
    topic: str
    question: str
    choices: list[str]
    answer: int              # index into choices (0-3)
    explanation: str
    source: str              # e.g. "Filipino Driver's Manual, Volume 1 (2nd Edition), p. 45"
    origin: str = "bank"     # "bank" or "generated"

    def is_correct(self, choice_index: int) -> bool:
        return int(choice_index) == int(self.answer)


def validate_question(d: dict) -> bool:
    """A usable question has text, exactly 4 distinct choices and a valid answer index."""
    try:
        choices = [str(c).strip() for c in d["choices"]]
        ans = int(d["answer"])
        return (len(str(d["question"]).strip()) > 10 and len(choices) == 4
                and len(set(c.lower() for c in choices)) == 4 and all(choices)
                and 0 <= ans < 4 and len(str(d.get("explanation", "")).strip()) > 0)
    except (KeyError, TypeError, ValueError):
        return False


def load_bank(path=config.QUESTION_BANK) -> list[Question]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Question(**q) for q in data if validate_question(q)]


GEN_PROMPT = """You write multiple-choice review questions for the Philippine LTO driver's license exam.
Use ONLY the manual passage below. Write ONE question about a fact stated in the passage.
Return JSON with keys: "question" (string), "choices" (list of exactly 4 short strings),
"answer" (index 0-3 of the correct choice), "explanation" (one sentence quoting the fact).
The three wrong choices must be plausible but clearly wrong according to the passage.

Topic: {topic}
Passage ({source}):
{text}"""


class QuizEngine:
    def __init__(self, retriever: HybridRetriever | None, client: OllamaClient | None = None,
                 bank: list[Question] | None = None, seed: int | None = None):
        self.retriever = retriever
        self.client = client or OllamaClient()
        self.bank = bank if bank is not None else load_bank()
        self.rng = random.Random(seed)

    def from_bank(self, topic: str, exclude: set[str] = frozenset()) -> Question | None:
        pool = [q for q in self.bank if q.topic == topic and q.qid not in exclude]
        if not pool:                                  # everything seen -> allow repeats
            pool = [q for q in self.bank if q.topic == topic]
        return self.rng.choice(pool) if pool else None

    def generate(self, topic: str) -> Question | None:
        """Ask the LLM for a new question grounded in a retrieved passage."""
        if self.retriever is None or not self.client.available():
            return None
        hits = self.retriever.search(TOPIC_QUERIES.get(topic, topic), k=5)
        if not hits:
            return None
        chunk = self.rng.choice(hits[:3]).chunk       # vary the source passage
        prompt = GEN_PROMPT.format(topic=topic, source=chunk.citation(), text=chunk.text)
        try:
            d = self.client.chat_json([{"role": "user", "content": prompt}], temperature=0.7)
        except (OllamaError, ValueError):
            return None
        if not validate_question(d):
            return None
        return Question(qid=f"gen-{chunk.chunk_id}-{self.rng.randint(0, 99999)}", topic=topic,
                        question=d["question"].strip(), choices=[str(c).strip() for c in d["choices"]],
                        answer=int(d["answer"]), explanation=str(d["explanation"]).strip(),
                        source=chunk.citation(), origin="generated")

    def next_question(self, topic: str, prefer_generated: bool = False,
                      exclude: set[str] = frozenset()) -> Question | None:
        q = self.generate(topic) if prefer_generated else None
        return q or self.from_bank(topic, exclude)


def question_to_dict(q: Question) -> dict:
    return asdict(q)
