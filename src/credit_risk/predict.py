from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd

ARTIFACT_DIR = Path("artifacts/model_v1")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    data = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))

    t = data.get("recall80")
    if t is None:
        raise KeyError(
            f"'recall80' not found in {THRESHOLD_PATH}. Keys: {list(data.keys())}"
        )
    return float(t)


def predict_one(row: Dict[str, Any]) -> Dict[str, Any]:
    model = load_model()
    threshold = load_threshold()

    X = pd.DataFrame([row])
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= threshold)

    return {"prob_default": prob, "threshold": threshold, "pred_default": pred}


def main():
    sample = {
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

    out = predict_one(sample)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
