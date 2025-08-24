from app.services.normalize import normalize_claim


def test_normalize_basic():
    assert normalize_claim("  Hello, WORLD!!  ") == "hello world"


def test_normalize_keeps_percent():
    out = normalize_claim("Efficacy is ~97% according to WHO.")
    assert "97%" in out
    assert "who" in out
