"""Pydantic request/response models."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SUPPORT = "SUPPORT"
    MIXED = "MIXED"
    REFUTE = "REFUTE"
    INSUFFICIENT = "INSUFFICIENT"


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=5, max_length=2000, examples=[
        "Measles vaccines provide strong protection against measles."
    ])


class Citation(BaseModel):
    doc_id: str
    source: str
    snippet: str
    score: float


class FeatureScores(BaseModel):
    retrieval_strength: float = Field(..., ge=0, le=1)
    lexical_overlap: float = Field(..., ge=0, le=1)
    support_signal: float = Field(..., ge=0, le=1)
    refute_signal: float = Field(..., ge=0, le=1)
    claim_specificity: float = Field(..., ge=0, le=1)
    evidence_agreement: float = Field(..., ge=0, le=1)


class VerifyResponse(BaseModel):
    id: int
    claim: str
    normalized_claim: str
    verdict: Verdict
    confidence: float = Field(..., ge=0, le=1)
    rationale: str
    features: FeatureScores
    citations: list[Citation]
    llm_backend: str
    created_at: datetime


class CheckSummary(BaseModel):
    id: int
    claim: str
    verdict: Verdict
    confidence: float
    llm_backend: str
    created_at: datetime


class CheckListResponse(BaseModel):
    total: int
    items: list[CheckSummary]


class HealthResponse(BaseModel):
    status: str
    corpus_docs: int
    llm_backend: str
    details: dict[str, Any] = Field(default_factory=dict)
