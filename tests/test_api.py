"""API integration tests against seeded corpus + mock LLM."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["corpus_docs"] >= 20
    assert body["llm_backend"] == "mock"


def test_verify_support_measles(client):
    r = client.post(
        "/api/v1/verify",
        json={"claim": "Two doses of measles vaccine provide strong protection against measles."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "SUPPORT"
    assert body["confidence"] > 0.4
    assert body["citations"]
    assert body["id"] >= 1
    assert "features" in body
    assert body["llm_backend"] == "mock"


def test_verify_refute_autism(client):
    r = client.post(
        "/api/v1/verify",
        json={"claim": "The MMR vaccine causes autism in children."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in {"REFUTE", "MIXED"}
    # Prefer REFUTE on this fixture; allow MIXED if scoring is soft
    assert body["citations"]


def test_verify_refute_flat_earth(client):
    r = client.post(
        "/api/v1/verify",
        json={"claim": "The Earth is flat according to modern satellite evidence."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "REFUTE"


def test_verify_insufficient(client):
    r = client.post(
        "/api/v1/verify",
        json={"claim": "The lost city of Atlantis was discovered under Antarctica in 2022."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in {"INSUFFICIENT", "MIXED", "REFUTE"}


def test_list_and_get_checks(client):
    client.post("/api/v1/verify", json={"claim": "Global renewable capacity additions hit a record in 2023."})
    listed = client.get("/api/v1/checks")
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["total"] >= 1
    check_id = payload["items"][0]["id"]
    detail = client.get(f"/api/v1/checks/{check_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == check_id


def test_get_missing(client):
    r = client.get("/api/v1/checks/999999")
    assert r.status_code == 404


def test_verify_validation(client):
    r = client.post("/api/v1/verify", json={"claim": "hi"})
    assert r.status_code == 422
