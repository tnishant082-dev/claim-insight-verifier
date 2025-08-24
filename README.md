# Claim Insight Verifier

**Paste a claim → normalize → retrieve evidence (RAG) → score with DS features → return SUPPORT / MIXED / REFUTE / INSUFFICIENT with citations.**

Offline-first MVP for recruiters and demo walks. No paid API key required — a deterministic mock LLM produces structured verdicts from retrieved snippets. Optionally plug in an OpenAI-compatible key for GenAI rationale.

| | |
|---|---|
| **Stack** | FastAPI · Pydantic · SQLAlchemy/SQLite · scikit-learn TF-IDF · Streamlit |
| **APIs** | `POST /api/v1/verify` · `GET /api/v1/checks` · `GET /api/v1/checks/{id}` |
| **Cost** | ₹0 to run locally (mock LLM + local corpus) |
| **Status** | Working MVP — not a production fact-checker |

---

## Why this exists

Claim verification is a clean fullstack + DS + GenAI slice:

1. **Normalize** noisy user text  
2. **Retrieve** short evidence docs from a seeded corpus (TF-IDF cosine; sentence-transformers if installed)  
3. **Score** retrieval strength, lexical overlap, support/refute cues, specificity, agreement  
4. **Judge** via pluggable LLM (OpenAI-compatible **or** offline mock)  
5. **Persist** check history in SQLite and show it in Streamlit  

Honest scope: synthetic news-like corpus (~35 docs), not live web search. Verdicts are corpus-relative, not ground truth about the world.

---

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — API
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
API_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py
```

- API docs: http://127.0.0.1:8000/docs  
- UI: http://127.0.0.1:8501  
- Health: `curl -s http://127.0.0.1:8000/health | jq`

### Sample verify

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{"claim":"Two doses of measles vaccine provide strong protection against measles."}' | jq
```

Expected shape (mock backend):

```json
{
  "id": 1,
  "claim": "Two doses of measles vaccine...",
  "normalized_claim": "two doses of measles vaccine...",
  "verdict": "SUPPORT",
  "confidence": 0.7,
  "rationale": "...",
  "features": {
    "retrieval_strength": 0.3,
    "lexical_overlap": 0.4,
    "support_signal": 0.5,
    "refute_signal": 0.1,
    "claim_specificity": 0.6,
    "evidence_agreement": 0.8
  },
  "citations": [{"doc_id": "...", "source": "WHO Fact Sheet", "snippet": "...", "score": 0.4}],
  "llm_backend": "mock",
  "created_at": "..."
}
```

### Demo fixtures (seeded corpus)

| Claim | Expected direction |
|---|---|
| Measles vaccine provides strong protection | **SUPPORT** |
| MMR vaccine causes autism | **REFUTE** (or MIXED) |
| Earth is flat / satellite evidence | **REFUTE** |
| Atlantis under Antarctica in 2022 | **INSUFFICIENT** (off-corpus) |

---

## Docker Compose (optional)

```bash
docker compose up --build
# API :8000  ·  UI :8501
```

Optional GenAI (never commit secrets):

```bash
export OPENAI_API_KEY=sk-...
docker compose up --build
```

Compatible proxies work via `OPENAI_BASE_URL` + `OPENAI_MODEL`.

---

## Tests

```bash
pytest -q
```

---

## Layout

```
app/                  # FastAPI + services (normalize, retrieve, features, LLM, pipeline)
ui/streamlit_app.py   # Dashboard + history
data/corpus/          # ~35 synthetic evidence paragraphs with SOURCE labels
tests/                # pytest (API + unit)
Dockerfile
docker-compose.yml
requirements.txt
```

---

## Design notes

| Choice | Why |
|---|---|
| **TF-IDF / hashing embeddings** | Installs cleanly on Linux, no GPU, works offline. Optional `sentence-transformers` upgrade if present. |
| **Mock LLM default** | Demos and CI without keys; still returns valid structured JSON using features + snippets. |
| **SQLite** | Zero-ops check history for MVP. |
| **Synthetic corpus** | Reproducible offline demos; not a substitute for live retrieval. |

### Verdict labels

- **SUPPORT** — evidence aligns with the claim  
- **MIXED** — evidence pulls both ways  
- **REFUTE** — evidence contradicts the claim  
- **INSUFFICIENT** — weak / off-topic retrieval  

---

## What this is not

- Not a live web fact-checker  
- Not calibrated production accuracy metrics (none claimed)  
- Not legal / medical advice — corpus is synthetic for demo purposes  

---

## License

MIT (or as declared by the repository owner).
