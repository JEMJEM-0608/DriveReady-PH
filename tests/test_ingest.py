from driveready import config
from driveready.ingest import chunk_pages, clean_page_text, guess_section, split_words


def test_clean_joins_hyphenated_words_and_drops_page_numbers():
    raw = "RIGHT OF WAY\ninter-\nsection rules\n  12  \nyield   to the right"
    out = clean_page_text(raw)
    assert "intersection" in out
    assert "12" not in out.split()
    assert "yield to the right" in out


def test_split_words_overlap():
    text = " ".join(str(i) for i in range(100))
    pieces = split_words(text, size=40, overlap=10)
    assert pieces[0].split()[-10:] == pieces[1].split()[:10]      # 10 shared words
    assert pieces[-1].split()[-1] == "99"                          # nothing lost at the end


def test_guess_section_finds_heading():
    assert guess_section("TRAFFIC SIGNALS\nA red light means stop.") == "Traffic Signals"


def test_chunk_pages_keeps_page_numbers():
    long_page = "PARKING\n" + " ".join(["word"] * (config.CHUNK_WORDS + 50))
    chunks = chunk_pages("FDM-Vol1", ["", long_page, "short"])
    assert chunks, "expected chunks"
    assert all(c.page == 2 for c in chunks)                      # page 1 empty, page 3 too short
    assert chunks[0].chunk_id == "FDM-Vol1-p002-c1"
    assert "p. 2" in chunks[0].citation()
