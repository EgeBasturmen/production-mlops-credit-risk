import pytest
from fastapi.testclient import TestClient

from credit_risk.config import PROJECT_ROOT


def _artifacts_ready() -> bool:
    art = PROJECT_ROOT / "artifacts" / "model_v2"
    return (art / "model.joblib").exists() and (art / "thresholds.json").exists()


@pytest.mark.skipif(
    not _artifacts_ready(), reason="Model artifacts not found. Run train.py first."
)
def test_health():
    from credit_risk.app import app

    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.skipif(
    not _artifacts_ready(), reason="Model artifacts not found. Run train.py first."
)
def test_model_info_has_thresholds():
    from credit_risk.app import app

    client = TestClient(app)

    r = client.get("/model/info")
    assert r.status_code == 200

    data = r.json()
    assert "thresholds" in data
    assert "recall80" in data["thresholds"]
    assert "maxf1" in data["thresholds"]


@pytest.mark.skipif(
    not _artifacts_ready(), reason="Model artifacts not found. Run train.py first."
)
def test_predict_valid_payload_returns_200():
    from credit_risk.app import app

    client = TestClient(app)

    payload = {
        "X1": 200000,
        "X2": 2,
        "X3": 2,
        "X4": 1,
        "X5": 30,
        "X6": 0,
        "X7": 0,
        "X8": 0,
        "X9": 0,
        "X10": 0,
        "X11": 0,
        "X12": 0,
        "X13": 0,
        "X14": 0,
        "X15": 0,
        "X16": 0,
        "X17": 0,
        "X18": 0,
        "X19": 0,
        "X20": 0,
        "X21": 0,
        "X22": 0,
        "X23": 0,
    }

    r = client.post("/predict?mode=fixed_0_5", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "proba_default" in data
    assert "threshold" in data
    assert "mode" in data
    assert "prediction" in data
    assert data["mode"] == "fixed_0_5"


@pytest.mark.skipif(
    not _artifacts_ready(), reason="Model artifacts not found. Run train.py first."
)
def test_predict_invalid_payload_returns_422():
    from credit_risk.app import app

    client = TestClient(app)

    bad_payload = {"X1": 123}  # eksik alanlar var -> 422 bekleriz
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422
