from app.services.features import compute_features, heuristic_verdict
from app.services.retriever import CorpusRetriever, EvidenceDoc, RetrievedHit


def test_heuristic_insufficient_when_empty():
    feats = compute_features("obscure unrelated claim xyzzy", [])
    verdict, conf = heuristic_verdict(feats)
    assert verdict == "INSUFFICIENT"
    assert 0 <= conf <= 1


def test_refute_markers_boost():
    doc = EvidenceDoc("x", "NASA", "Earth is an oblate spheroid. Flat-Earth claims contradict evidence.", "")
    hit = RetrievedHit(doc=doc, score=0.6, snippet=doc.text)
    feats = compute_features("The Earth is flat", [hit])
    verdict, _ = heuristic_verdict(feats)
    assert verdict in {"REFUTE", "MIXED"}
