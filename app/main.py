"""FastAPI entrypoint for Claim Insight Verifier."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.db import get_db, init_db
from app.models.schemas import (
    CheckListResponse,
    CheckSummary,
    HealthResponse,
    VerifyRequest,
    VerifyResponse,
    Verdict,
)
from app.services.pipeline import VerifyPipeline

pipeline: VerifyPipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline
    init_db()
    pipeline = VerifyPipeline(get_settings())
    yield


app = FastAPI(
    title="Claim Insight Verifier",
    version="0.1.0",
    description="Paste a claim → RAG evidence → DS scores → SUPPORT/MIXED/REFUTE/INSUFFICIENT",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pipe() -> VerifyPipeline:
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    return pipeline


@app.get("/health", response_model=HealthResponse)
def health():
    p = _pipe()
    settings = get_settings()
    return HealthResponse(
        status="ok",
        corpus_docs=p.retriever.size,
        llm_backend=p.llm.backend,
        details={"retriever": p.retriever.backend, "top_k": settings.top_k},
    )


@app.post("/api/v1/verify", response_model=VerifyResponse)
def verify(body: VerifyRequest, db: Session = Depends(get_db)):
    return _pipe().verify(body.claim, db)


@app.get("/api/v1/checks", response_model=CheckListResponse)
def list_checks(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    total, rows = _pipe().list_checks(db, limit=limit, offset=offset)
    items = [
        CheckSummary(
            id=r.id,
            claim=r.claim,
            verdict=Verdict(r.verdict),
            confidence=r.confidence,
            llm_backend=r.llm_backend,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return CheckListResponse(total=total, items=items)


@app.get("/api/v1/checks/{check_id}", response_model=VerifyResponse)
def get_check(check_id: int, db: Session = Depends(get_db)):
    result = _pipe().get_check(check_id, db)
    if result is None:
        raise HTTPException(404, f"Check {check_id} not found")
    return result
