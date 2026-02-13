from __future__ import annotations
from fastapi import Query
import json
from pathlib import Path
from typing import Literal, Dict, Any
from datetime import datetime
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from credit_risk.drift import load_prod_jsonl, compute_drift_summary
from credit_risk.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from credit_risk.schema import TARGET_COLUMN
from credit_risk.data_load import load_raw_data  # sadece debug endpointi için
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from fastapi import Request
from fastapi.responses import JSONResponse
from credit_risk.config import PROD_PRED_PATH
from credit_risk.config import PROJECT_ROOT

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "model_v2"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
THRESH_PATH = ARTIFACT_DIR / "thresholds.json"
BASELINE_STATS_PATH = ARTIFACT_DIR / "baseline_feature_stats.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found: {MODEL_PATH.resolve()}")
if not THRESH_PATH.exists():
    raise RuntimeError(f"Thresholds not found: {THRESH_PATH.resolve()}")

MODEL = joblib.load(MODEL_PATH)
THRESHOLDS: Dict[str, float] = json.loads(THRESH_PATH.read_text(encoding="utf-8"))

DEFAULT_MODE = "recall80"


app = FastAPI(title="Credit Risk API", version="0.0.1")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("credit_risk_api")
logger.setLevel(logging.INFO)
PROD_LOG_DIR = PROJECT_ROOT / "data" / "prod"
PROD_LOG_DIR.mkdir(parents=True, exist_ok=True)
PROD_LOG_PATH = PROD_LOG_DIR / "predictions.jsonl"



handler = RotatingFileHandler(
    LOG_DIR / "app.log",
    maxBytes=2_000_000,
    backupCount=5,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


class PredictRequest(BaseModel):
    # X1..X23
    X1: float
    X2: int
    X3: int
    X4: int
    X5: float
    X6: int
    X7: int
    X8: int
    X9: int
    X10: int
    X11: int
    X12: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float
    X18: float
    X19: float
    X20: float
    X21: float
    X22: float
    X23: float


class PredictResponse(BaseModel):
    proba_default: float
    threshold: float
    mode: str
    prediction: int

@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception(f"request_id={request_id} path={request.url.path} status=500 latency_ms={latency_ms:.2f}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "request_id": request_id},
        )

    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-latency-ms"] = f"{latency_ms:.2f}"

    logger.info(f"request_id={request_id} path={request.url.path} status={response.status_code} latency_ms={latency_ms:.2f}")
    return response

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    return {
        "artifact_dir": str(ARTIFACT_DIR),
        "model_path": str(MODEL_PATH),
        "thresholds": THRESHOLDS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
        "default_mode": DEFAULT_MODE,
    }

@app.get("/drift/summary")
def drift_summary(limit: int = Query(500, ge=1, le=5000)):
    if not BASELINE_STATS_PATH.exists():
        return {"status": "NO_BASELINE", "detail": f"Missing {str(BASELINE_STATS_PATH)}"}


    baseline_stats = json.loads(BASELINE_STATS_PATH.read_text(encoding="utf-8"))
    prod_df = load_prod_jsonl(PROD_PRED_PATH, max_rows=limit)

    summary = compute_drift_summary(
        baseline_stats=baseline_stats,
        prod_df=prod_df,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    return summary


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    request: Request,
    mode: Literal["recall80", "maxf1", "fixed_0_5"] = DEFAULT_MODE,

):
    if mode not in THRESHOLDS:
        raise HTTPException(status_code=400, detail=f"Unknown mode={mode}. Available={list(THRESHOLDS.keys())}")

    request_id = request.headers.get("x-request-id", "-")

    # request -> dataframe
    row = payload.model_dump()
    X = pd.DataFrame([row])

    # proba
    proba = float(MODEL.predict_proba(X)[:, 1][0])

    # decision
    thr = float(THRESHOLDS[mode])
    pred = int(proba >= thr)

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id,
        "mode": mode,
        "threshold": thr,
        "proba_default": proba,
        "prediction": pred,
        "features": row,  # payload'tan gelen X1..X23
    }

    with open(PROD_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        f"request_id={request_id} mode={mode} proba={proba:.6f} "
        f"threshold={thr:.6f} pred={pred}"
    )

    return PredictResponse(
        proba_default=proba,
        threshold=thr,
        mode=mode,
        prediction=pred,
    )


@app.get("/debug/data")
def debug_data():
    df = load_raw_data()
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "target_counts": df["default"].value_counts().to_dict(),
    }
