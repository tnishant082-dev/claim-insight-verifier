"""Shared pytest fixtures — isolated SQLite + real corpus."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_checks.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("CORPUS_DIR", str(CORPUS))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from app.config import get_settings

    get_settings.cache_clear()

    import app.models.db as dbmod
    from app.models.db import Base, make_engine

    engine = make_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    dbmod.engine = engine
    dbmod.SessionLocal = TestSession

    # Ensure init_db in lifespan also uses our engine
    def _init_db():
        Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(dbmod, "init_db", _init_db)

    import app.main as mainmod

    def _override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    mainmod.app.dependency_overrides[dbmod.get_db] = _override_db

    with TestClient(mainmod.app) as c:
        yield c

    mainmod.app.dependency_overrides.clear()
    get_settings.cache_clear()
