.PHONY: install test api ui compose

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest -q

api:
	. .venv/bin/activate && uvicorn app.main:app --reload --port 8000

ui:
	. .venv/bin/activate && API_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py

compose:
	docker compose up --build
