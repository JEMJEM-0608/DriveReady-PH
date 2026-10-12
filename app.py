"""DriveReady PH - Streamlit web app.

Run with:   streamlit run app.py
Tabs
  Ask          chat with the LTO driver's manual (RAG + local LLM, with page citations)
  Quiz         adaptive practice quiz; the RL bandit picks the next topic
  My Progress  per-topic mastery estimated by the bandit + answer history
  About        system status (Ollama, models, knowledge base)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driveready import config
from driveready.knowledge import kb_fingerprint, load_retriever
from driveready.llm import OllamaClient
from driveready.quiz import QuizEngine
from driveready.rag import RAGPipeline
from driveready.storage import Store

st.set_page_config(page_title="DriveReady PH", page_icon="🚦", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources (loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the driver's manual index...")
def get_resources():
    client = OllamaClient()
    retriever = load_retriever(client=client)
    return client, retriever, RAGPipeline(retriever, client), QuizEngine(retriever, client), Store()


try:
    client, retriever, rag, quiz, store = get_resources()
except FileNotFoundError as e:
    st.error(f"Knowledge base not built yet. {e}")
    st.code("python scripts/download_docs.py\npython scripts/build_index.py")
    st.stop()

llm_on = client.available()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🚦 DriveReady PH")
    st.caption("LTO driver's exam reviewer")
    user = st.text_input("Your name", value=st.session_state.get("user", "student")).strip() or "student"
    st.session_state["user"] = user
    if llm_on:
        st.success(f"LLM online: {client.model}")
    else:
        st.warning("LLM offline - answers show the best manual passage. "
                   f"Start Ollama and run `ollama pull {client.model}`.")
    st.caption(f"Search: BM25 + {retriever.dense_name} · {len(retriever.chunks)} passages")

tab_ask, tab_quiz, tab_prog, tab_about = st.tabs(["💬 Ask", "📝 Quiz", "📈 My Progress", "ℹ️ About"])

# ---------------------------------------------------------------------------
# Ask tab - RAG chat
# ---------------------------------------------------------------------------
with tab_ask:
    st.subheader("Ask about traffic rules, road signs, and licensing")
    st.caption("Answers come only from the official LTO Filipino Driver's Manual and cite the page.")
    chat = st.session_state.setdefault("chat", [])
    examples = ["What should I do when I see a flashing red traffic light?",
                "What is the speed limit on a crowded street?",
                "What documents do I need for a student permit?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, width="stretch"):
            st.session_state["pending"] = ex

    for turn in chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            for src in turn.get("sources", []):
                st.caption(src)

    prompt = st.chat_input("Type your question (English or Filipino)...")
    prompt = prompt or st.session_state.pop("pending", None)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching the manual..."):
                history = [{"role": t["role"], "content": t["content"]} for t in chat]
                ans = rag.ask(prompt, history=history)
            st.markdown(ans.text)
            sources = [f"[{i}] {h.chunk.citation()}" + (f" — {h.chunk.section}" if h.chunk.section else "")
                       for i, h in zip(ans.cited or range(1, len(ans.hits) + 1), ans.sources())]
            for s in sources:
                st.caption(s)
            if not ans.grounded:
                st.info("This answer cited no passage - double-check it in the manual.")
            with st.expander("Show retrieved passages"):
                for i, h in enumerate(ans.hits, start=1):
                    st.markdown(f"**[{i}] {h.chunk.citation()}** · BM25 rank {h.bm25_rank or '-'}, "
                                f"dense rank {h.dense_rank or '-'}")
                    st.write(h.chunk.text)
            st.caption(f"{ans.mode} · retrieval {ans.timings['retrieval_s']*1000:.0f} ms · "
                       f"total {ans.timings['total_s']:.1f} s")
            qa_id = store.log_answer(user, ans)
            c1, c2, _ = st.columns([1, 1, 8])
            c1.button("👍", key=f"up{qa_id}", on_click=store.add_feedback, args=(qa_id, True))
            c2.button("👎", key=f"dn{qa_id}", on_click=store.add_feedback, args=(qa_id, False))
        chat += [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": ans.text, "sources": sources}]

    if chat and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Quiz tab - RL adaptive quiz
# ---------------------------------------------------------------------------
with tab_quiz:
    st.subheader("Adaptive practice quiz")
    st.caption("A reinforcement-learning bandit picks the next topic: it asks more about topics you miss, "
               "but still checks the others.")
    bandit = store.load_bandit(user)
    gen = st.toggle("Let the LLM write new questions from the manual", value=False, disabled=not llm_on)

    if "q" not in st.session_state or st.session_state.get("q_user") != user:
        st.session_state.update(q=None, q_user=user, answered=False)

    if st.button("Next question ▶", type="primary") or st.session_state["q"] is None:
        topic = bandit.select_topic()
        seen = {a["question_id"] for a in store.attempts(user)}
        with st.spinner(f"Preparing a '{topic}' question..."):
            st.session_state["q"] = quiz.next_question(topic, prefer_generated=gen, exclude=seen)
        st.session_state["answered"] = False

    q = st.session_state["q"]
    if q is None:
        st.warning("No questions available yet. Build the question bank (see README).")
    else:
        st.markdown(f"**Topic chosen by the bandit:** {q.topic}  ·  _{q.origin} question_")
        st.markdown(f"### {q.question}")
        choice = st.radio("Choose one:", range(4), format_func=lambda i: f"{'ABCD'[i]}. {q.choices[i]}",
                          index=None, key=f"radio-{q.qid}", disabled=st.session_state["answered"])
        if st.button("Submit answer", disabled=choice is None or st.session_state["answered"]):
            ok = q.is_correct(choice)
            bandit.update_topic(q.topic, ok)
            store.save_bandit(user, bandit)
            store.log_attempt(user, q.topic, q.qid, ok, q.origin)
            st.session_state["answered"] = True
            st.session_state["last_ok"] = ok
        if st.session_state["answered"]:
            if st.session_state["last_ok"]:
                st.success("Correct! ✅")
            else:
                st.error(f"Not quite. Correct answer: {'ABCD'[q.answer]}. {q.choices[q.answer]}")
            st.info(f"{q.explanation}\n\nSource: {q.source}")

# ---------------------------------------------------------------------------
# Progress tab
# ---------------------------------------------------------------------------
with tab_prog:
    st.subheader(f"Progress for {user}")
    bandit = store.load_bandit(user)
    attempts = store.attempts(user)
    m = bandit.mastery()
    df = pd.DataFrame({"Topic": list(m), "Estimated mastery": list(m.values()),
                       "Questions asked": bandit.pulls.tolist()})
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions answered", len(attempts))
    c2.metric("Accuracy", f"{(sum(a['correct'] for a in attempts) / len(attempts) * 100):.0f}%"
              if attempts else "—")
    weakest = min(m, key=m.get) if attempts else "—"
    c3.metric("Focus topic", weakest)
    st.bar_chart(df.set_index("Topic")["Estimated mastery"], horizontal=True)
    st.dataframe(df.style.format({"Estimated mastery": "{:.0%}"}), hide_index=True,
                 width="stretch")
    st.caption("Mastery = 1 − posterior mean error rate from the Thompson-sampling bandit "
               "(recent answers weigh more because of the discount factor).")
    if st.button("Reset my progress"):
        store.reset_user(user)
        st.rerun()

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("System status")
    st.json({
        "ollama_url": client.url,
        "llm_model": client.model,
        "llm_available": llm_on,
        "embedding_backend": retriever.dense_name,
        "passages_indexed": len(retriever.chunks),
        "knowledge_base_fingerprint": kb_fingerprint(retriever.chunks),
        "sources": [s["title"] for s in config.SOURCES.values()],
        "question_bank_size": len(quiz.bank),
    })
    st.markdown("""
**How it works**
1. Your question is matched against the manual with **BM25** (keywords) and **embeddings** (meaning).
2. The two result lists are merged with **Reciprocal Rank Fusion**; the top 5 passages are kept.
3. A **local open-source LLM** (Ollama) answers using only those passages and cites them.
4. In the quiz, a **Thompson-sampling bandit** (reinforcement learning) chooses topics where you make mistakes.

_DriveReady PH is a study aid. Always follow the official LTO materials and current laws._
""")
