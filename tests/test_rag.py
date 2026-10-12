from conftest import FakeClient

from driveready.rag import RAGPipeline, build_messages, detect_language, extract_citations, strip_invalid_citations


def test_extract_citations_valid_unique_in_order():
    assert extract_citations("Stop fully [2]. Then go [1, 2]. Ignore [9].", 5) == [2, 1]


def test_strip_invalid_citations():
    assert strip_invalid_citations("Fact [7]. Other [1, 8].", 5) == "Fact . Other [1]."


def test_prompt_contains_numbered_passages_and_question(retriever):
    hits = retriever.search("flashing red light", k=2)
    msgs = build_messages("What does a flashing red light mean?", hits)
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    assert "[1] (Test Manual, page" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("What does a flashing red light mean?")
    assert "Answer language: English" in msgs[-1]["content"]


def test_detect_language():
    assert detect_language("What does a flashing red light mean?") == "English"
    assert detect_language("Ano ang ibig sabihin ng flashing red light?") == "Filipino"
    assert detect_language("Pwede ba mag-park sa tabi ng fire hydrant?") == "Filipino"


def test_llm_answer_with_citations(retriever):
    client = FakeClient(reply="Come to a full stop, then proceed when safe [1].")
    ans = RAGPipeline(retriever, client).ask("What does a flashing red light mean?")
    assert ans.mode == "llm" and ans.cited == [1] and ans.grounded
    assert ans.sources()[0].chunk.section == "Traffic Signals"


def test_uncited_answer_is_flagged(retriever):
    client = FakeClient(reply="You should stop.")
    ans = RAGPipeline(retriever, client).ask("flashing red light?")
    assert not ans.grounded


def test_offline_fallback_quotes_best_passage(retriever):
    ans = RAGPipeline(retriever, FakeClient(available=False)).ask("What is the speed limit in a school zone?")
    assert ans.mode == "extractive"
    assert "20 kph" in ans.text and "[1]" in ans.text
