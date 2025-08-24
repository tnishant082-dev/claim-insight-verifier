"""Application settings — offline-first defaults, optional OpenAI key."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Claim Insight Verifier"
    api_prefix: str = "/api/v1"
    database_url: str = f"sqlite:///{ROOT / 'data' / 'checks.db'}"
    corpus_dir: str = str(ROOT / "data" / "corpus")
    top_k: int = 5

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
