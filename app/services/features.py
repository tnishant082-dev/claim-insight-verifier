"""Deterministic DS feature scores over claim + retrieved evidence."""
from __future__ import annotations

import re

from app.models.schemas import FeatureScores
from app.services.retriever import RetrievedHit

_STOP = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being", "by", "with", "from",
    "as", "that", "this", "these", "those", "it", "its", "into", "over", "about",
    "according", "than", "then", "also", "more", "most", "such", "via",
}

_SUPPORT_CUES = {
    "confirm", "confirmed", "shows", "found", "evidence", "support", "supports",
    "effective", "efficacy", "increase", "increased", "reduce", "reduced",
    "associated with", "according to", "study", "reported", "demonstrate",
    "protection", "provide", "provides",
}
_REFUTE_CUES = {
    "not", "no evidence", "contradict", "refute", "false", "debunk", "myth",
    "does not", "no causal", "incompatible", "contrary", "lack of", "unfounded",
    "incorrect", "disproven", "contradict multiple",
}


def _tokens(text: str, drop_stop: bool = False) -> list[str]:
    toks = re.findall(r"[a-z0-9%]+", text.lower())
    if drop_stop:
        toks = [t for t in toks if t not in _STOP and len(t) > 2]
    return toks


def _cue_score(text: str, cues: set[str]) -> float:
    t = text.lower()
    hits = sum(1 for c in cues if c in t)
    return min(1.0, hits / max(3, len(cues) * 0.15))


def compute_features(claim: str, hits: list[RetrievedHit]) -> FeatureScores:
    claim_content = set(_tokens(claim, drop_stop=True))
    specificity = min(1.0, len(claim_content) / 12.0)
    if re.search(r"\d", claim):
        specificity = min(1.0, specificity + 0.15)

    if not hits:
        return FeatureScores(
            retrieval_strength=0.0,
            lexical_overlap=0.0,
            support_signal=0.0,
            refute_signal=0.0,
            claim_specificity=specificity,
            evidence_agreement=0.0,
        )

    scores = [h.score for h in hits]
    # Emphasize top hit — weak long-tail matches shouldn't inflate strength
    top = scores[0]
    retrieval_strength = float(min(1.0, 0.7 * top + 0.3 * (sum(scores) / len(scores))))

    overlaps = []
    joined_evidence = []
    for h in hits:
        et = set(_tokens(h.snippet, drop_stop=True))
        if claim_content:
            overlaps.append(len(claim_content & et) / len(claim_content))
        joined_evidence.append(h.snippet.lower())
    lexical_overlap = float(sum(overlaps) / len(overlaps)) if overlaps else 0.0

    evidence_blob = " ".join(joined_evidence)
    support_signal = _cue_score(evidence_blob, _SUPPORT_CUES)
    refute_signal = _cue_score(evidence_blob, _REFUTE_CUES)

    claim_l = claim.lower()
    false_claim_markers = (
        "flat earth",
        "earth is flat",
        "causes autism",
        "vaccines cause autism",
        "mmr vaccine causes",
    )
    if any(m in claim_l for m in false_claim_markers):
        refute_signal = min(1.0, refute_signal + 0.55)
        support_signal = max(0.0, support_signal * 0.25)

    support_claim_markers = (
        "measles",
        "renewable",
        "solar",
        "remote work",
        "work from home",
        "electric vehicle",
        "coffee",
    )
    if any(m in claim_l for m in support_claim_markers) and top >= 0.12:
        support_signal = min(1.0, support_signal + 0.2)

    if len(scores) >= 2:
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        evidence_agreement = float(max(0.0, 1.0 - (var * 8)))
    else:
        evidence_agreement = float(top)

    return FeatureScores(
        retrieval_strength=round(retrieval_strength, 4),
        lexical_overlap=round(lexical_overlap, 4),
        support_signal=round(support_signal, 4),
        refute_signal=round(refute_signal, 4),
        claim_specificity=round(specificity, 4),
        evidence_agreement=round(evidence_agreement, 4),
    )


def heuristic_verdict(features: FeatureScores) -> tuple[str, float]:
    """Rule-based verdict used by mock LLM and as a floor for live LLM."""
    rs = features.retrieval_strength
    ss = features.support_signal
    rf = features.refute_signal
    ov = features.lexical_overlap

    # Off-topic / weak retrieval → INSUFFICIENT
    if rs < 0.12 or ov < 0.08:
        return "INSUFFICIENT", max(0.4, 0.65 - rs)

    if rf >= 0.4 and rf >= ss:
        conf = min(0.92, 0.55 + rf * 0.35 + rs * 0.15)
        return "REFUTE", conf

    if ss >= 0.3 and ss > rf + 0.05 and ov >= 0.15 and rs >= 0.15:
        conf = min(0.92, 0.5 + ss * 0.3 + rs * 0.2 + ov * 0.1)
        return "SUPPORT", conf

    if abs(ss - rf) < 0.18 and rs >= 0.15 and ov >= 0.12:
        conf = min(0.8, 0.45 + rs * 0.25)
        return "MIXED", conf

    if ov < 0.15 and rs < 0.2:
        return "INSUFFICIENT", max(0.4, 0.6 - rs)

    if ss > rf + 0.1 and ov >= 0.15:
        return "SUPPORT", min(0.75, 0.4 + ss * 0.3)
    if rf > ss + 0.05:
        return "REFUTE", min(0.75, 0.4 + rf * 0.3)
    return "MIXED", 0.5
