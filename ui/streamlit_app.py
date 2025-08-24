"""Streamlit dashboard for Claim Insight Verifier."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Claim Insight Verifier",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Claim Insight Verifier")
st.caption("Normalize → RAG over seeded corpus → DS scores → verdict with citations (offline-first)")

SAMPLES = {
    "SUPPORT — measles vaccine efficacy":
        "Two doses of measles vaccine provide strong protection against measles.",
    "REFUTE — vaccines cause autism":
        "The MMR vaccine causes autism in children.",
    "REFUTE — flat Earth":
        "The Earth is flat according to modern satellite evidence.",
    "SUPPORT — renewable capacity growth":
        "Global renewable electricity capacity additions hit a record in 2023 driven by solar.",
    "MIXED / nuance — intermittent fasting":
        "Intermittent fasting always produces better metabolic outcomes than calorie restriction.",
    "INSUFFICIENT — off-corpus":
        "The lost city of Atlantis was discovered under Antarctica in 2022.",
}


@st.cache_data(ttl=10)
def fetch_health():
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


col_a, col_b = st.columns([2, 1])
with col_b:
    health = fetch_health()
    if health.get("status") == "ok":
        st.success(
            f"API OK · {health.get('corpus_docs', '?')} docs · LLM: `{health.get('llm_backend')}`"
        )
    else:
        st.error(f"API unreachable at `{API_URL}` — start FastAPI first.\n\n{health.get('error', '')}")

with col_a:
    sample_label = st.selectbox("Sample claims", list(SAMPLES.keys()))
    claim = st.text_area("Claim", value=SAMPLES[sample_label], height=100)

run = st.button("Verify claim", type="primary", use_container_width=False)

if run and claim.strip():
    with st.spinner("Running verify pipeline…"):
        try:
            resp = httpx.post(
                f"{API_URL}/api/v1/verify",
                json={"claim": claim.strip()},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Verify failed: {exc}")
            data = None

    if data:
        verdict = data["verdict"]
        color = {
            "SUPPORT": "green",
            "MIXED": "orange",
            "REFUTE": "red",
            "INSUFFICIENT": "gray",
        }.get(verdict, "blue")
        st.markdown(f"### Verdict: :{color}[**{verdict}**]  ·  confidence `{data['confidence']:.2f}`")
        st.write(data["rationale"])
        st.caption(
            f"Normalized: `{data['normalized_claim']}` · backend `{data['llm_backend']}` · id `{data['id']}`"
        )

        fcols = st.columns(3)
        feats = data["features"]
        keys = list(feats.keys())
        for i, k in enumerate(keys):
            fcols[i % 3].metric(k.replace("_", " ").title(), f"{feats[k]:.3f}")

        st.subheader("Citations")
        if data["citations"]:
            cdf = pd.DataFrame(data["citations"])
            st.dataframe(cdf, use_container_width=True, hide_index=True)
        else:
            st.info("No citations retrieved.")

st.divider()
st.subheader("Check history")
if st.button("Refresh history"):
    st.cache_data.clear()

try:
    hist = httpx.get(f"{API_URL}/api/v1/checks", params={"limit": 25}, timeout=10.0)
    hist.raise_for_status()
    payload = hist.json()
    if payload["items"]:
        hdf = pd.DataFrame(payload["items"])
        st.dataframe(
            hdf[["id", "verdict", "confidence", "claim", "llm_backend", "created_at"]],
            use_container_width=True,
            hide_index=True,
        )
        pick = st.number_input("Load check id", min_value=1, step=1, value=int(hdf.iloc[0]["id"]))
        if st.button("Load detail"):
            detail = httpx.get(f"{API_URL}/api/v1/checks/{int(pick)}", timeout=10.0)
            if detail.status_code == 200:
                st.json(detail.json())
            else:
                st.warning(detail.text)
    else:
        st.info("No checks yet — verify a claim above.")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not load history: {exc}")
