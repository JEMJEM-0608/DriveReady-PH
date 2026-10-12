# DriveReady PH 🚦

**An LTO driver's-exam reviewer that answers questions from the official *Filipino Driver's Manual* using Retrieval-Augmented Generation (RAG), a local open-source LLM (Llama 3.1 8B via Ollama), and a reinforcement-learning adaptive quiz.**

- **Ask** – questions about traffic rules, signs, licensing and penalties are answered *only* from the manual, with page citations.
- **Quiz** – a Thompson-sampling multi-armed bandit (RL) picks the next topic, focusing on the topics you get wrong.
- **My Progress** – per-topic mastery estimated by the bandit.
- Runs **100% offline** on your own PC once the models are downloaded. No API keys.

> Project for *Emerging Technologies in Computer Engineering* – Mapúa University, School of EECE, Computer Engineering (AI specialization).

---

## 1. How it works

```
 Question ──► BM25 keyword search ─┐
                                   ├─► Reciprocal Rank Fusion ─► top-5 manual passages ─► Llama 3.1 8B (Ollama)
 Question ──► nomic-embed search ──┘                                                       │
                                                                                            ▼
                                                              answer + [1]..[5] citations ─► citation validator ─► app
 Quiz:  Thompson-sampling bandit ─► topic ─► question (curated bank or LLM-generated from a retrieved passage)
        answer right/wrong ─► reward ─► bandit update (discounted Beta posterior)
```

| Component | File | What it does |
|---|---|---|
| Ingestion | `driveready/ingest.py` | PDF → cleaned page text → 180-word overlapping chunks with page numbers |
| Retrieval | `driveready/retriever.py` | BM25 + dense vectors (nomic-embed-text, LSA, or FastText) fused with RRF |
| Knowledge base | `driveready/knowledge.py` | Builds/caches vectors; picks the best available embedding backend |
| LLM client | `driveready/llm.py` | Talks to the local Ollama server (`/api/chat`, `/api/embed`) |
| RAG pipeline | `driveready/rag.py` | Prompt with numbered passages, citation validation, offline fallback |
| RL quiz | `driveready/bandit.py`, `driveready/quiz.py` | Thompson-sampling topic selection; question bank + LLM question generation |
| Storage | `driveready/storage.py` | SQLite: chat log, quiz attempts, bandit state, feedback |
| Web app | `app.py` | Streamlit UI with Ask / Quiz / My Progress / About tabs |

The manual's **Chapter VI "Reviewer"** (Vol. 1, PDF pages 163–200, official exam questions) is **held out** of the search index. It is used as the quiz question bank and as the evaluation set, so the system is tested on questions it cannot simply look up.

---

## 2. Requirements

- Windows 10/11, macOS or Linux; **16 GB RAM or more** (32 GB recommended)
- **Python 3.10 – 3.14** – <https://www.python.org/downloads/> (tick *Add python.exe to PATH*)
- **Ollama** – <https://ollama.com/download>
- About 6 GB of free disk space for the models

---

## 3. Setup (Windows, VS Code terminal / PowerShell)

**Quickest way (Windows):** double-click or run `setup_windows.bat`, then `run_app.bat`. To reproduce the results, run `run_evaluation.bat`. These batch files call `.venv\Scripts\python.exe` directly, so PowerShell's script-execution policy does not get in the way.

Manual steps:

```powershell
# 1) get the code
git clone https://github.com/JEMJEM-0608/DriveReady-PH.git
cd DriveReady-PH

# 2) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#   (if PowerShell blocks it: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# 3) install the Python packages
pip install -r requirements.txt

# 4) download the two open-source models (one time, ~5 GB)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 5) get the official LTO manuals into data/raw/
python scripts/download_docs.py
#   If the download fails, save these two PDFs into data\raw\ by hand:
#   https://lto.gov.ph/wp-content/uploads/2023/10/FDM-Vol.-1-2nd-Edition.pdf
#   https://lto.gov.ph/wp-content/uploads/2023/10/FDM-vol.-2-2nd-Edition.pdf

# 6) build the search index (chunks + embeddings)
python scripts/build_index.py
```

macOS/Linux: use `source .venv/bin/activate` in step 2.

> `data/processed/chunks.jsonl` is already in the repository, so the app also starts without step 5–6; the embeddings are then built automatically on the first run.

---

## 4. Run the app

Make sure Ollama is running (it starts with Windows; otherwise run `ollama serve`), then:

```powershell
streamlit run app.py
```

Open <http://localhost:8501>. The sidebar shows **LLM online: llama3.1:8b** when everything is connected. If Ollama is off, the app still works in *offline mode* and shows the most relevant manual passage instead of a generated answer.

---

## 5. Evaluate (reproduces the numbers in the report)

```powershell
python -m pytest -q                      # 24 unit tests (no Ollama needed)
python scripts/simulate_bandit.py        # RL experiment: bandit vs. random / round-robin / epsilon-greedy
python scripts/evaluate.py --no-llm      # retrieval only: Recall@k, MRR (seconds)
python scripts/evaluate.py               # full: retrieval + RAG vs LLM-only answers + quiz generation
python scripts/evaluate.py --limit 20    # quicker run on the first 20 questions
```

All results are written to `reports/` as JSON (and PNG for the RL chart):

| File | Contents |
|---|---|
| `reports/kb_stats.json` | number of chunks, pages, embedding backends, knowledge-base fingerprint |
| `reports/eval_retrieval.json` | Recall@1/3/5, MRR@10, latency for BM25, LSA, nomic and hybrid search (`eval_retrieval_dev.json` adds FastText) |
| `reports/eval_answers.json` | per-question RAG and LLM-only answers, accuracy, citations, latency |
| `reports/eval_quiz.json` | validity rate of LLM-generated quiz questions + samples |
| `reports/bandit_simulation.json/.png` | RL simulation results |

---

## 6. Project structure

```
DriveReady-PH/
├── app.py                     Streamlit web app
├── driveready/                main package (see table in section 1)
├── scripts/
│   ├── download_docs.py       fetch the LTO manuals
│   ├── build_index.py         build chunks + embeddings
│   ├── evaluate.py            retrieval / answer / quiz evaluation
│   └── simulate_bandit.py     reinforcement-learning experiment
├── data/
│   ├── raw/                   source PDFs (not committed)
│   ├── processed/chunks.jsonl searchable passages with page numbers
│   ├── eval/qa_eval.json      53 evaluation questions (48 answerable + 5 out-of-scope)
│   └── quiz/question_bank.json 66 multiple-choice questions from the official reviewer
├── reports/                   JSON/PNG results
├── tests/                     pytest unit tests
└── docs/                      project documentation
```

---

## 7. Configuration

All settings live in `driveready/config.py` and can be overridden with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `DRIVEREADY_LLM` | `llama3.1:8b` | any chat model pulled in Ollama (e.g. `qwen2.5:7b`, `llama3.2:3b` for 8 GB PCs) |
| `DRIVEREADY_EMBED` | `nomic-embed-text` | embedding model in Ollama |
| `DRIVEREADY_DENSE` | `auto` | `ollama`, `lsa`, `fasttext` (needs gensim), or `auto` (Ollama if available, else LSA) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |

PowerShell example: `$env:DRIVEREADY_LLM="llama3.2:3b"; streamlit run app.py`

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar says *LLM offline* | Start Ollama (`ollama serve`) and check `ollama list` shows `llama3.1:8b` |
| `Knowledge base not built yet` | Run `python scripts/build_index.py` |
| First answer takes 30–60 s | Normal: Ollama loads the 4.9 GB model into memory on the first request |
| Very slow answers | No GPU – try the smaller model: `ollama pull llama3.2:3b` and set `DRIVEREADY_LLM` |
| `Activate.ps1 cannot be loaded` | Use the `.bat` files, call `.venv\Scripts\python.exe` directly, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `pip` fails building gensim | gensim is optional; it has no wheels for Python 3.13+. Skip it (LSA is used instead) |

---

## 9. Data, licences and disclaimer

- Source documents: *Filipino Driver's Manual*, Volumes 1 and 2 (2nd Edition), © Land Transportation Office (LTO), Republic of the Philippines. They are downloaded from the LTO website for educational use and are **not** redistributed in this repository.
- Models: Llama 3.1 8B (Meta Llama 3.1 Community License), nomic-embed-text (Apache 2.0), served by Ollama (MIT).
- Code: MIT License.
- DriveReady PH is a **study aid**, not legal advice. Always follow the official LTO materials and current laws.

## 10. Team

- Jezreel Mari Manjares Hembrador
- Andrea Bague
- Miguel Jose Malvar
