"""SQLite persistence (Python standard library, no server needed).

Tables
  qa_log         every question asked in the Ask tab, with mode and latency
  quiz_attempts  every quiz answer (who, topic, question, correct?)
  bandit_state   the learner's Thompson-sampling state, one row per user
  feedback       thumbs up / down on answers (used in the evaluation)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config
from .bandit import TopicBandit

SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, user TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
    mode TEXT NOT NULL, cited TEXT NOT NULL, retrieval_s REAL, llm_s REAL, total_s REAL
);
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, user TEXT NOT NULL, topic TEXT NOT NULL, question_id TEXT NOT NULL,
    correct INTEGER NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bandit_state (
    user TEXT PRIMARY KEY, state TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, qa_id INTEGER NOT NULL, helpful INTEGER NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str = config.DB_PATH):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- Ask tab ----------------------------------------------------------------
    def log_answer(self, user: str, ans) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO qa_log (ts,user,question,answer,mode,cited,retrieval_s,llm_s,total_s)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (_now(), user, ans.question, ans.text, ans.mode, json.dumps(ans.cited),
                 ans.timings.get("retrieval_s"), ans.timings.get("llm_s"), ans.timings.get("total_s")))
            return int(cur.lastrowid)

    def add_feedback(self, qa_id: int, helpful: bool) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO feedback (ts,qa_id,helpful) VALUES (?,?,?)",
                      (_now(), qa_id, int(helpful)))

    def recent_questions(self, user: str, limit: int = 10) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT ts,question,mode,total_s FROM qa_log WHERE user=? "
                             "ORDER BY id DESC LIMIT ?", (user, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- Quiz tab -----------------------------------------------------------------
    def log_attempt(self, user: str, topic: str, question_id: str, correct: bool, source: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO quiz_attempts (ts,user,topic,question_id,correct,source)"
                      " VALUES (?,?,?,?,?,?)", (_now(), user, topic, question_id, int(correct), source))

    def attempts(self, user: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT ts,topic,question_id,correct,source FROM quiz_attempts "
                             "WHERE user=? ORDER BY id", (user,)).fetchall()
        return [dict(r) for r in rows]

    def load_bandit(self, user: str) -> TopicBandit:
        with self._conn() as c:
            row = c.execute("SELECT state FROM bandit_state WHERE user=?", (user,)).fetchone()
        if row is None:
            return TopicBandit()
        state = json.loads(row["state"])
        if state.get("topics") != list(config.QUIZ_TOPICS):   # topic list changed -> reset
            return TopicBandit()
        return TopicBandit.from_dict(state)

    def save_bandit(self, user: str, bandit: TopicBandit) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO bandit_state (user,state,updated) VALUES (?,?,?) "
                      "ON CONFLICT(user) DO UPDATE SET state=excluded.state, updated=excluded.updated",
                      (user, json.dumps(bandit.to_dict()), _now()))

    def reset_user(self, user: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM bandit_state WHERE user=?", (user,))
            c.execute("DELETE FROM quiz_attempts WHERE user=?", (user,))
