from driveready.retriever import tokenize


def test_tokenize_keeps_law_numbers_and_drops_stopwords():
    toks = tokenize("What is the penalty under RA-10913 at 60kph?")
    assert "ra-10913" in toks and "60kph" in toks
    assert "the" not in toks and "is" not in toks


def test_bm25_finds_exact_terms(retriever):
    hits = retriever.search("flashing red light", k=3, mode="bm25")
    assert hits[0].chunk.page == 1


def test_hybrid_returns_k_unique_results_with_ranks(retriever):
    hits = retriever.search("speed limit in school zones", k=3)
    assert len(hits) == 3
    assert len({h.chunk.chunk_id for h in hits}) == 3
    assert hits[0].chunk.section == "Speed Limits"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_rrf_rewards_agreement(retriever):
    # a passage ranked well by BOTH retrievers should beat one found by only one
    hits = retriever.search("yield to the vehicle on the right at an intersection", k=6)
    top = hits[0]
    assert top.bm25_rank is not None and top.dense_rank is not None
