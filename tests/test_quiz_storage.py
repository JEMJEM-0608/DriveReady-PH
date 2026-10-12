from conftest import FakeClient

from driveready.quiz import Question, QuizEngine, validate_question
from driveready.rag import RAGPipeline
from driveready.storage import Store

GOOD = {"question": "What does a solid yellow line mean?",
        "choices": ["No overtaking", "Parking allowed", "Bus lane", "School zone"],
        "answer": 0, "explanation": "A solid yellow line means overtaking is not allowed."}


def test_validate_question_rules():
    assert validate_question(GOOD)
    assert not validate_question({**GOOD, "answer": 4})
    assert not validate_question({**GOOD, "choices": GOOD["choices"][:3]})
    assert not validate_question({**GOOD, "choices": ["A", "A", "B", "C"]})


def test_generated_question_is_grounded(retriever):
    engine = QuizEngine(retriever, FakeClient(json_reply=GOOD), bank=[], seed=0)
    q = engine.next_question("Pavement markings & signals", prefer_generated=True)
    assert q is not None and q.origin == "generated" and "Test Manual" in q.source


def test_falls_back_to_bank_when_generation_invalid(retriever):
    bank = [Question("b1", "Road signs", "What shape is a STOP sign?",
                     ["Octagon", "Circle", "Triangle", "Square"], 0, "It is an octagon.", "p. 1")]
    engine = QuizEngine(retriever, FakeClient(json_reply={"bad": True}), bank=bank, seed=0)
    q = engine.next_question("Road signs", prefer_generated=True)
    assert q.qid == "b1" and q.is_correct(0) and not q.is_correct(1)


def test_store_round_trip(tmp_path, retriever):
    store = Store(tmp_path / "t.db")
    b = store.load_bandit("ana")
    b.update_topic("Road signs & traffic lights", correct=False)
    store.save_bandit("ana", b)
    assert store.load_bandit("ana").pulls[0] == 1
    store.log_attempt("ana", "Road signs", "b1", False, "bank")
    assert store.attempts("ana")[0]["correct"] == 0
    ans = RAGPipeline(retriever, FakeClient(available=False)).ask("flashing red light")
    qa_id = store.log_answer("ana", ans)
    store.add_feedback(qa_id, True)
    assert store.recent_questions("ana")[0]["mode"] == "extractive"
    store.reset_user("ana")
    assert store.attempts("ana") == []
