"""Pluggable LLM client: OpenAI-compatible when keyed, else deterministic mock."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.config import Settings
from app.models.schemas import FeatureScores, Verdict
from app.services.features import heuristic_verdict
from app.services.retriever import RetrievedHit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an evidence-based claim verifier.
Given a claim, DS feature scores, and retrieved evidence snippets, return ONLY valid JSON:
{
  "verdict": "SUPPORT" | "MIXED" | "REFUTE" | "INSUFFICIENT",
  "confidence": 0.0-1.0,
  "rationale": "2-4 sentences citing evidence"
}
Rules: base the verdict on the provided evidence only; do not invent sources;
prefer INSUFFICIENT when evidence is weak or off-topic.
"""


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._key = (settings.openai_api_key or "").strip() or None

    @property
    def backend(self) -> str:
        return f"openai:{self.settings.openai_model}" if self._key else "mock"

    def judge(
        self,
        claim: str,
        features: FeatureScores,
        hits: list[RetrievedHit],
    ) -> dict[str, Any]:
        if self._key:
            try:
                return self._openai_judge(claim, features, hits)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI call failed (%s); falling back to mock", exc)
        return self._mock_judge(claim, features, hits)

    def _mock_judge(
        self,
        claim: str,
        features: FeatureScores,
        hits: list[RetrievedHit],
    ) -> dict[str, Any]:
        verdict, confidence = heuristic_verdict(features)
        cites = []
        for h in hits[:3]:
            cites.append(f"[{h.doc.source}] {h.snippet[:120]}")
        cite_txt = "; ".join(cites) if cites else "no strong corpus matches"

        rationale_map = {
            "SUPPORT": (
                f"Retrieved evidence aligns with the claim. Key sources: {cite_txt}. "
                f"Feature scores show support_signal={features.support_signal:.2f} "
                f"vs refute_signal={features.refute_signal:.2f} with retrieval_strength="
                f"{features.retrieval_strength:.2f}."
            ),
            "REFUTE": (
                f"Evidence contradicts or undermines the claim. Key sources: {cite_txt}. "
                f"refute_signal={features.refute_signal:.2f} exceeds support_signal="
                f"{features.support_signal:.2f}."
            ),
            "MIXED": (
                f"Evidence pulls in more than one direction. Sources: {cite_txt}. "
                f"support={features.support_signal:.2f}, refute={features.refute_signal:.2f}."
            ),
            "INSUFFICIENT": (
                f"Corpus retrieval did not yield enough on-topic evidence to judge "
                f"the claim confidently (retrieval_strength={features.retrieval_strength:.2f}). "
                f"Matched snippets: {cite_txt}."
            ),
        }
        return {
            "verdict": Verdict(verdict),
            "confidence": round(float(confidence), 4),
            "rationale": rationale_map[verdict],
            "backend": "mock",
        }

    def _openai_judge(
        self,
        claim: str,
        features: FeatureScores,
        hits: list[RetrievedHit],
    ) -> dict[str, Any]:
        evidence = [
            {
                "source": h.doc.source,
                "doc_id": h.doc.doc_id,
                "score": h.score,
                "snippet": h.snippet,
            }
            for h in hits
        ]
        user = {
            "claim": claim,
            "features": features.model_dump(),
            "evidence": evidence,
        }
        payload = {
            "model": self.settings.openai_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        verdict = str(parsed.get("verdict", "INSUFFICIENT")).upper()
        if verdict not in {v.value for v in Verdict}:
            verdict = "INSUFFICIENT"
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(parsed.get("rationale", "No rationale returned.")).strip()
        return {
            "verdict": Verdict(verdict),
            "confidence": round(confidence, 4),
            "rationale": rationale,
            "backend": f"openai:{self.settings.openai_model}",
        }
