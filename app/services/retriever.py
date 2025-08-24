"""Offline-first retrieval: TF-IDF + numpy cosine (no GPU required).

Optionally upgrades to sentence-transformers if importable; otherwise
sklearn HashingVectorizer / TfidfVectorizer path always works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SOURCE_RE = re.compile(r"^SOURCE:\s*(.+)$", re.MULTILINE)
_DOC_ID_RE = re.compile(r"^DOC_ID:\s*(.+)$", re.MULTILINE)


@dataclass
class EvidenceDoc:
    doc_id: str
    source: str
    text: str
    path: str


@dataclass
class RetrievedHit:
    doc: EvidenceDoc
    score: float
    snippet: str


def _snippet(text: str, max_len: int = 280) -> str:
    body = re.sub(r"^SOURCE:.*\nDOC_ID:.*\n\n?", "", text, flags=re.MULTILINE).strip()
    if len(body) <= max_len:
        return body
    return body[: max_len - 1].rsplit(" ", 1)[0] + "…"


class CorpusRetriever:
    """In-memory TF-IDF / hashing embedding index over corpus files."""

    def __init__(self, corpus_dir: str | Path):
        self.corpus_dir = Path(corpus_dir)
        self.docs: list[EvidenceDoc] = []
        self._matrix = None
        self._vectorizer = None
        self._backend = "tfidf"
        self._load()

    def _load(self) -> None:
        paths = sorted(self.corpus_dir.glob("*.txt"))
        for p in paths:
            raw = p.read_text(encoding="utf-8")
            src_m = _SOURCE_RE.search(raw)
            id_m = _DOC_ID_RE.search(raw)
            doc_id = (id_m.group(1).strip() if id_m else p.stem)
            source = (src_m.group(1).strip() if src_m else "Unknown")
            self.docs.append(EvidenceDoc(doc_id=doc_id, source=source, text=raw, path=str(p)))

        if not self.docs:
            # Empty corpus — keep a tiny stub so API still boots
            self.docs.append(
                EvidenceDoc(
                    doc_id="empty",
                    source="system",
                    text="SOURCE: system\nDOC_ID: empty\n\nNo corpus documents loaded.",
                    path="",
                )
            )

        texts = [d.text for d in self.docs]
        # Prefer Tfidf when corpus is small; HashingVectorizer as ultra-light fallback
        try:
            self._vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=8192,
                sublinear_tf=True,
            )
            self._matrix = self._vectorizer.fit_transform(texts)
            self._backend = "tfidf"
        except Exception:
            self._vectorizer = HashingVectorizer(
                n_features=2**12,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
                stop_words="english",
            )
            self._matrix = self._vectorizer.transform(texts)
            self._backend = "hashing"

        # Optional sentence-transformers upgrade (best-effort, never required)
        self._st_model = None
        self._st_matrix = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._st_matrix = np.asarray(
                self._st_model.encode([d.text for d in self.docs], normalize_embeddings=True)
            )
            self._backend = "sentence-transformers+tfidf"
        except Exception:
            pass

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def size(self) -> int:
        return len(self.docs)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedHit]:
        if not query.strip():
            return []

        if self._st_model is not None and self._st_matrix is not None:
            q = np.asarray(self._st_model.encode([query], normalize_embeddings=True))
            sims = (self._st_matrix @ q.T).ravel()
        else:
            q = self._vectorizer.transform([query])
            sims = cosine_similarity(q, self._matrix).ravel()

        k = min(top_k, len(self.docs))
        idx = np.argsort(-sims)[:k]
        hits: list[RetrievedHit] = []
        for i in idx:
            score = float(sims[i])
            # Drop near-zero / noise matches so citations stay on-topic
            if score < 0.05:
                continue
            doc = self.docs[int(i)]
            hits.append(
                RetrievedHit(doc=doc, score=round(score, 4), snippet=_snippet(doc.text))
            )
        return hits
