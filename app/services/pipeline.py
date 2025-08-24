"""End-to-end verify pipeline: normalize → retrieve → features → LLM → persist."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models.db import CheckRecord
from app.models.schemas import Citation, FeatureScores, VerifyResponse
from app.services.features import compute_features
from app.services.llm import LLMClient
from app.services.normalize import normalize_claim
from app.services.retriever import CorpusRetriever


class VerifyPipeline:
    def __init__(self, settings: Settings | None = None, retriever: CorpusRetriever | None = None):
        self.settings = settings or get_settings()
        self.retriever = retriever or CorpusRetriever(self.settings.corpus_dir)
        self.llm = LLMClient(self.settings)

    def verify(self, claim: str, db: Session) -> VerifyResponse:
        normalized = normalize_claim(claim)
        hits = self.retriever.search(normalized or claim, top_k=self.settings.top_k)
        features = compute_features(normalized or claim, hits)
        judgment = self.llm.judge(claim.strip(), features, hits)

        citations = [
            Citation(
                doc_id=h.doc.doc_id,
                source=h.doc.source,
                snippet=h.snippet,
                score=h.score,
            )
            for h in hits
        ]

        record = CheckRecord(
            claim=claim.strip(),
            normalized_claim=normalized,
            verdict=judgment["verdict"].value,
            confidence=judgment["confidence"],
            rationale=judgment["rationale"],
            features_json=features.model_dump_json(),
            citations_json=json.dumps([c.model_dump() for c in citations]),
            llm_backend=judgment["backend"],
            created_at=datetime.now(timezone.utc),
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return VerifyResponse(
            id=record.id,
            claim=record.claim,
            normalized_claim=record.normalized_claim,
            verdict=judgment["verdict"],
            confidence=judgment["confidence"],
            rationale=judgment["rationale"],
            features=features,
            citations=citations,
            llm_backend=judgment["backend"],
            created_at=record.created_at,
        )

    def get_check(self, check_id: int, db: Session) -> VerifyResponse | None:
        record = db.get(CheckRecord, check_id)
        if not record:
            return None
        return self._to_response(record)

    def list_checks(self, db: Session, limit: int = 50, offset: int = 0):
        q = db.query(CheckRecord).order_by(CheckRecord.id.desc())
        total = q.count()
        rows = q.offset(offset).limit(limit).all()
        return total, rows

    @staticmethod
    def _to_response(record: CheckRecord) -> VerifyResponse:
        features = FeatureScores.model_validate_json(record.features_json)
        citations = [Citation.model_validate(c) for c in json.loads(record.citations_json)]
        from app.models.schemas import Verdict

        return VerifyResponse(
            id=record.id,
            claim=record.claim,
            normalized_claim=record.normalized_claim,
            verdict=Verdict(record.verdict),
            confidence=record.confidence,
            rationale=record.rationale,
            features=features,
            citations=citations,
            llm_backend=record.llm_backend,
            created_at=record.created_at,
        )
