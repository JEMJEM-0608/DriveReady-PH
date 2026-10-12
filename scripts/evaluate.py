"""Evaluate DriveReady PH and write JSON reports to reports/.

Parts
  1. Retrieval  - Recall@1/3/5 and MRR of BM25, dense and hybrid search
                  on data/eval/qa_eval.json (no LLM needed).
  2. Answers    - RAG answers vs. an LLM-only baseline (same model, no manual):
                  keyword accuracy, citation rate, citation correctness,
                  correct refusals on unanswerable questions, latency.
                  Needs Ollama; skipped automatically if it is not running.
  3. Quiz       - validity rate of LLM-generated quiz questions (needs Ollama).

Usage
  python scripts/evaluate.py              # everything that can run
  python scripts/evaluate.py --no-llm     # retrieval only
  python scripts/evaluate.py --quiz-n 3   # generated questions per topic
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from driveready import config  # noqa: E402
from driveready.ingest import load_chunks  # noqa: E402
from driveready.knowledge import build_dense, kb_fingerprint  # noqa: E402
from driveready.llm import OllamaClient  # noqa: E402
from driveready.quiz import QuizEngine, validate_question  # noqa: E402
from driveready.rag import NOT_FOUND, RAGPipeline  # noqa: E402
from driveready.retriever import HybridRetriever, gensim_available  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_eval() -> list[dict]:
    return json.loads(config.EVAL_SET.read_text(encoding="utf-8"))


def is_gold(chunk, item: dict) -> bool:
    return any(chunk.doc_id == g["doc"] and chunk.page == g["page"] for g in item["gold"])


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def keyword_correct(answer: str, must: list[str]) -> bool:
    """Every required fact must appear; 'a|b' means either spelling is fine."""
    low = answer.lower()
    return all(any(alt.strip().lower() in low for alt in group.split("|")) for group in must)


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(s in low for s in (NOT_FOUND.lower()[:30], "not in the manual", "cannot find",
                                  "could not find", "does not mention", "not mentioned", "do not know",
                                  "don't know", "not sure", "not familiar", "hindi ko makita"))


def save(name: str, data: dict) -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] reports/{name}")


# ---------------------------------------------------------------------------
# 1. retrieval
# ---------------------------------------------------------------------------
def eval_retrieval(items: list[dict], retrievers: dict[str, tuple[HybridRetriever, str]]) -> dict:
    answerable = [it for it in items if it.get("answerable", True)]
    out = {}
    for label, (ret, mode) in retrievers.items():
        ranks, times = [], []
        for it in answerable:
            t0 = time.perf_counter()
            hits = ret.search(it["question"], k=10, mode=mode)
            times.append((time.perf_counter() - t0) * 1000)
            rank = next((i for i, h in enumerate(hits, start=1) if is_gold(h.chunk, it)), None)
            ranks.append(rank)
        n = len(ranks)
        out[label] = {
            "recall@1": round(sum(r is not None and r <= 1 for r in ranks) / n, 3),
            "recall@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
            "recall@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
            "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
            "latency_ms_p50": round(pct(times, 50), 2),
            "latency_ms_p95": round(pct(times, 95), 2),
            "questions": n,
        }
        print(f"  {label:<28} R@1 {out[label]['recall@1']:.2f}  R@5 {out[label]['recall@5']:.2f}  "
              f"MRR {out[label]['mrr@10']:.2f}  {out[label]['latency_ms_p50']:.1f} ms")
    return out


# ---------------------------------------------------------------------------
# 2. answers
# ---------------------------------------------------------------------------
LLM_ONLY_SYSTEM = ("You are a reviewer for the Philippine LTO driver's license exam. Answer in 2 to 5 "
                   "short sentences. If you are not sure, say you do not know.")


def eval_answers(items: list[dict], rag: RAGPipeline, client: OllamaClient) -> dict:
    rows = []
    client.chat([{"role": "user", "content": "Say OK."}])          # warm-up: load the model into RAM
    for it in items:
        ans = rag.ask(it["question"])
        base = client.chat([{"role": "system", "content": LLM_ONLY_SYSTEM},
                            {"role": "user", "content": it["question"]}])
        answerable = it.get("answerable", True)
        cited_gold = any(is_gold(ans.hits[c - 1].chunk, it) for c in ans.cited) if answerable else None
        rows.append({
            "id": it["id"], "topic": it.get("topic", ""), "answerable": answerable,
            "rag_answer": ans.text, "llm_only_answer": base.text,
            "rag_correct": keyword_correct(ans.text, it["must_include"]) if answerable else is_refusal(ans.text),
            "llm_only_correct": (keyword_correct(base.text, it["must_include"]) if answerable
                                 else is_refusal(base.text)),
            "rag_cited": bool(ans.cited), "rag_cited_gold_page": cited_gold,
            "rag_seconds": round(ans.timings["total_s"], 3), "llm_only_seconds": round(base.seconds, 3),
        })
        print(f"  {it['id']:<6} RAG {'OK ' if rows[-1]['rag_correct'] else 'X  '} "
              f"LLM-only {'OK ' if rows[-1]['llm_only_correct'] else 'X  '} {ans.timings['total_s']:.1f}s", flush=True)
    ans_rows = [r for r in rows if r["answerable"]]
    un_rows = [r for r in rows if not r["answerable"]]
    rag_t = [r["rag_seconds"] for r in rows]
    summary = {
        "model": client.model,
        "questions": len(rows), "answerable": len(ans_rows), "unanswerable": len(un_rows),
        "rag_accuracy_answerable": round(statistics.mean(r["rag_correct"] for r in ans_rows), 3),
        "llm_only_accuracy_answerable": round(statistics.mean(r["llm_only_correct"] for r in ans_rows), 3),
        "rag_citation_rate": round(statistics.mean(r["rag_cited"] for r in ans_rows), 3),
        "rag_cited_gold_page_rate": round(statistics.mean(bool(r["rag_cited_gold_page"]) for r in ans_rows), 3),
        "rag_correct_refusals": (round(statistics.mean(r["rag_correct"] for r in un_rows), 3)
                                 if un_rows else None),
        "llm_only_correct_refusals": (round(statistics.mean(r["llm_only_correct"] for r in un_rows), 3)
                                      if un_rows else None),
        "rag_latency_s_p50": round(pct(rag_t, 50), 2), "rag_latency_s_p95": round(pct(rag_t, 95), 2),
        "llm_only_latency_s_p50": round(pct([r["llm_only_seconds"] for r in rows], 50), 2),
    }
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# 3. quiz generation
# ---------------------------------------------------------------------------
def eval_quiz(engine: QuizEngine, n_per_topic: int) -> dict:
    total, valid, times, samples = 0, 0, [], []
    for topic in config.QUIZ_TOPICS:
        for _ in range(n_per_topic):
            total += 1
            t0 = time.perf_counter()
            q = engine.generate(topic)
            times.append(time.perf_counter() - t0)
            if q is not None and validate_question(q.__dict__):
                valid += 1
                if len(samples) < 8:
                    samples.append(q.__dict__)
    return {"generated": total, "valid": valid, "validity_rate": round(valid / total, 3) if total else None,
            "seconds_p50": round(pct(times, 50), 2), "samples": samples}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="retrieval metrics only")
    ap.add_argument("--quiz-n", type=int, default=2, help="generated questions per topic")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N answerable questions (quick run)")
    args = ap.parse_args()

    items = load_eval()
    if args.limit:
        answerable = [it for it in items if it.get("answerable", True)][:args.limit]
        items = answerable + [it for it in items if not it.get("answerable", True)]
    chunks = load_chunks()
    client = OllamaClient()
    llm_ok = (not args.no_llm) and client.available()
    embed_ok = client.has_model(client.embed_model)

    meta = {"timestamp": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
            "python": platform.python_version(), "kb_fingerprint": kb_fingerprint(chunks),
            "chunks": len(chunks), "eval_questions": len(items),
            "llm_model": client.model if llm_ok else None,
            "embedding_model": client.embed_model if embed_ok else None}
    print(json.dumps(meta, indent=2))

    # --- retrieval: every available backend ---
    print("\n[1] Retrieval")
    lsa = HybridRetriever(chunks, build_dense(chunks, "lsa"), "lsa")
    retrievers = {"BM25 only": (lsa, "bm25"),
                  "LSA only": (lsa, "dense"),
                  "Hybrid BM25+LSA": (lsa, "hybrid")}
    if gensim_available():
        ft = HybridRetriever(chunks, build_dense(chunks, "fasttext"), "fasttext")
        retrievers["FastText only"] = (ft, "dense")
        retrievers["Hybrid BM25+FastText"] = (ft, "hybrid")
    if embed_ok:
        ol = HybridRetriever(chunks, build_dense(chunks, "ollama", client), "ollama")
        retrievers["nomic-embed only"] = (ol, "dense")
        retrievers["Hybrid BM25+nomic (app)"] = (ol, "hybrid")
    retrieval = eval_retrieval(items, retrievers)
    save("eval_retrieval.json", {"meta": meta, "results": retrieval})

    summary = {"meta": meta, "retrieval": retrieval}
    if llm_ok:
        best = retrievers.get("Hybrid BM25+nomic (app)", retrievers["Hybrid BM25+LSA"])[0]
        print("\n[2] Answers: RAG vs LLM-only")
        answers = eval_answers(items, RAGPipeline(best, client), client)
        save("eval_answers.json", {"meta": meta, **answers})
        summary["answers"] = answers["summary"]
        print("\n[3] Quiz generation")
        quiz = eval_quiz(QuizEngine(best, client, seed=42), args.quiz_n)
        save("eval_quiz.json", {"meta": meta, **quiz})
        summary["quiz_generation"] = {k: v for k, v in quiz.items() if k != "samples"}
    else:
        print("\n[2-3] skipped: Ollama/LLM not available (start Ollama and pull the model)")
    save("eval_summary.json", summary)


if __name__ == "__main__":
    main()
