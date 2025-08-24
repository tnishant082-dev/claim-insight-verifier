from pathlib import Path

from app.services.retriever import CorpusRetriever

ROOT = Path(__file__).resolve().parent.parent


def test_corpus_loads():
    r = CorpusRetriever(ROOT / "data" / "corpus")
    assert r.size >= 20
    hits = r.search("measles vaccine protection", top_k=3)
    assert hits
    assert hits[0].score > 0
    assert "measles" in hits[0].snippet.lower() or "vaccine" in hits[0].snippet.lower()
