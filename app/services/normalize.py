"""Claim normalization for retrieval and feature scoring."""
from __future__ import annotations

import re
import unicodedata


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s%-]", re.UNICODE)


def normalize_claim(text: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).strip()
    t = t.lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    # Soft length cap for embedding / TF-IDF
    if len(t) > 800:
        t = t[:800].rsplit(" ", 1)[0]
    return t
