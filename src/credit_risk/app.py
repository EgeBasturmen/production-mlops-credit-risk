from __future__ import annotations
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import json
import logging
import time
import uuid
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Literal
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse


from credit_risk.config import (
    CHAMPION_PATH,
    APP_ENV,
    PROJECT_ROOT,
    LATEST_PATH,
    MODEL_SELECT_MODE,
    DEFAULT_MODE,
    MODEL_VERSION,
    LOG_DIR,
    PROD_PRED_PATH,
    LOG_LEVEL,
)
from credit_risk.data_load import load_raw_data
from credit_risk.drift import compute_drift_summary, load_prod_jsonl
from credit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.schema import TARGET_COLUMN
from credit_risk.logging_utils import JsonFormatter


if MODEL_SELECT_MODE not in {"latest", "manual", "champion"}:
    raise RuntimeError(
        f"Invalid MODEL_SELECT_MODE={MODEL_SELECT_MODE}. "
        f"Allowed: latest, manual, champion"
    )

LATEST_META = {}
CHAMPION_META = {}

if LATEST_PATH.exists():
    LATEST_META = json.loads(LATEST_PATH.read_text(encoding="utf-8"))

if CHAMPION_PATH.exists():
    CHAMPION_META = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))

if MODEL_SELECT_MODE == "latest":
    if not LATEST_PATH.exists():
        raise RuntimeError(f"latest.json not found: {LATEST_PATH.resolve()}")
    if "artifact_subdir" not in LATEST_META:
        raise RuntimeError("latest.json missing required key: artifact_subdir")
    artifact_subdir = LATEST_META["artifact_subdir"]

elif MODEL_SELECT_MODE == "champion":
    if not CHAMPION_PATH.exists():
        raise RuntimeError(f"champion.json not found: {CHAMPION_PATH.resolve()}")
    if "artifact_subdir" not in CHAMPION_META:
        raise RuntimeError("champion.json missing required key: artifact_subdir")
    artifact_subdir = CHAMPION_META["artifact_subdir"]

else:  # manual
    artifact_subdir = MODEL_VERSION

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / artifact_subdir

USE_CALIBRATED_MODEL = os.getenv("USE_CALIBRATED_MODEL", "false").lower() == "true"

MODEL_FILENAME = "model_calibrated.joblib" if USE_CALIBRATED_MODEL else "model.joblib"
THRESH_FILENAME = (
    "thresholds_calibrated.json" if USE_CALIBRATED_MODEL else "thresholds.json"
)

MODEL_PATH = ARTIFACTS_DIR / MODEL_FILENAME
THRESH_PATH = ARTIFACTS_DIR / THRESH_FILENAME
BASELINE_STATS_PATH = ARTIFACTS_DIR / "baseline_feature_stats.json"
RUN_META_PATH = ARTIFACTS_DIR / "run_meta.json"

RUN_META = json.loads(RUN_META_PATH.read_text(encoding="utf-8"))

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found: {MODEL_PATH.resolve()}")
if not THRESH_PATH.exists():
    raise RuntimeError(f"Thresholds not found: {THRESH_PATH.resolve()}")

MODEL = joblib.load(MODEL_PATH)
THRESHOLDS: Dict[str, float] = json.loads(THRESH_PATH.read_text(encoding="utf-8"))

app = FastAPI(title="Credit Risk API", version="0.0.1")

REQUIRED_THRESHOLD_KEYS = {"recall80", "maxf1", "fixed_0_5"}


@app.on_event("startup")
def startup_checks():
    if DEFAULT_MODE not in {"recall80", "maxf1", "fixed_0_5"}:
        raise RuntimeError(
            f"Invalid DEFAULT_MODE={DEFAULT_MODE}. Allowed: recall80, maxf1, fixed_0_5"
        )
    allowed_select_modes = {"latest", "manual", "champion"}
    if MODEL_SELECT_MODE not in allowed_select_modes:
        raise RuntimeError(
            f"Invalid MODEL_SELECT_MODE={MODEL_SELECT_MODE}."
            f"Allowed:{sorted(allowed_select_modes)}"
        )

    allowed_modes = {"recall80", "maxf1", "fixed_0_5"}
    if DEFAULT_MODE not in allowed_modes:
        raise RuntimeError(
            f"Invalid DEFAULT_MODE={DEFAULT_MODE}. Allowed: {sorted(allowed_modes)}"
        )

    allowed_envs = {"dev", "prod", "test"}
    if APP_ENV not in allowed_envs:
        raise RuntimeError(
            f"Invalid APP_ENV={APP_ENV}. Allowed: {sorted(allowed_envs)}"
        )
    if "artifact_subdir" not in LATEST_META or "model_version" not in LATEST_META:
        raise RuntimeError(
            "latest.json missing required keys: artifact_subdir/model_version"
        )

    missing = []
    required_files = [
        MODEL_PATH,
        THRESH_PATH,
        LATEST_PATH,
        BASELINE_STATS_PATH,
        RUN_META_PATH,
    ]
    for p in required_files:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise RuntimeError(f"Missing required artifacts:{missing}")
    # thresholds.json
    thresholds = json.loads(THRESH_PATH.read_text(encoding="utf-8"))
    missing_keys = REQUIRED_THRESHOLD_KEYS - set(thresholds.keys())
    if missing_keys:
        raise RuntimeError(f"thresholds.json missing keys: {sorted(missing_keys)}")

    # baseline json
    baseline = json.loads(BASELINE_STATS_PATH.read_text(encoding="utf-8"))
    if "numeric" not in baseline or "categorical" not in baseline:
        raise RuntimeError(
            "baseline_feature_stats.json missing 'numeric' or 'categorical'"
        )

    # model validasyonu
    if not hasattr(MODEL, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba")

    logger.info(
        "startup_validation_ok",
        extra={
            "event": "startup_validation_ok",
            "artifact_dir": str(ARTIFACTS_DIR),
            "model_path": str(MODEL_PATH),
            "model_file": MODEL_FILENAME,
            "threshold_file": THRESH_FILENAME,
            "using_calibrated_model": USE_CALIBRATED_MODEL,
            "threshold_keys": sorted(list(thresholds.keys())),
        },
    )


PRED_REQUESTS_TOTAL = Counter(
    "pred_requests_total",
    "Total number of/predict requests",
)

PRED_RISKY_TOTAL = Counter(
    "pred_risky_total",
    "Total number of risky predictions(prediction=1)",
)

PRED_LATENCY_MS = Histogram(
    "pred_latency_ms",
    "Latency of /predict in milliseconds",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2000),
)

PRED_RESPONSES_TOTAL = Counter(
    "pred_responses_total",
    "Total /predict responses by HTTP status",
    labelnames=("status_code",),
)

PRED_ERRORS_TOTAL = Counter(
    "pred_errors_total",
    "Total /predict errors by type",
    labelnames=("error_type",),
)

PRED_MODE_TOTAL = Counter(
    "pred_mode_total",
    "Total /predict requests by mode",
    labelnames=("mode",),
)


LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("credit_risk_api")

logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

PROD_LOG_DIR = PROJECT_ROOT / "data" / "prod"
PROD_LOG_DIR.mkdir(parents=True, exist_ok=True)
PROD_LOG_PATH = PROD_LOG_DIR / "predictions.jsonl"


handler = RotatingFileHandler(
    LOG_DIR / "app.log",
    maxBytes=2_000_000,
    backupCount=5,
    encoding="utf-8",
)

handler.setFormatter(JsonFormatter())

if not logger.handlers:
    logger.addHandler(handler)


class PredictRequest(BaseModel):
    # Credit amount / demographic
    X1: float = Field(..., ge=0, le=1_000_000, description="LIMIT_BAL")
    X2: int = Field(..., description="SEX")
    X3: int = Field(..., description="EDUCATION")
    X4: int = Field(..., description="MARRIAGE")
    X5: float = Field(..., ge=18, le=100, description="AGE")

    # Repayment status
    X6: int = Field(..., ge=-2, le=10, description="PAY_0")
    X7: int = Field(..., ge=-2, le=10, description="PAY_2")
    X8: int = Field(..., ge=-2, le=10, description="PAY_3")
    X9: int = Field(..., ge=-2, le=10, description="PAY_4")
    X10: int = Field(..., ge=-2, le=10, description="PAY_5")
    X11: int = Field(..., ge=-2, le=10, description="PAY_6")

    # Bill amounts
    X12: float = Field(..., ge=0, le=2_000_000, description="BILL_AMT1")
    X13: float = Field(..., ge=0, le=2_000_000, description="BILL_AMT2")
    X14: float = Field(..., ge=0, le=2_000_000, description="BILL_AMT3")
    X15: float = Field(..., ge=0, le=2_000_000, description="BILL_AMT4")
    X16: float = Field(..., ge=0, le=2_000_000, description="BILL_AMT5")
    X17: float = Field(..., ge=0, le=2_000_000, description="BILL_AMT6")

    # Payment amounts
    X18: float = Field(..., ge=0, le=2_000_000, description="PAY_AMT1")
    X19: float = Field(..., ge=0, le=2_000_000, description="PAY_AMT2")
    X20: float = Field(..., ge=0, le=2_000_000, description="PAY_AMT3")
    X21: float = Field(..., ge=0, le=2_000_000, description="PAY_AMT4")
    X22: float = Field(..., ge=0, le=2_000_000, description="PAY_AMT5")
    X23: float = Field(..., ge=0, le=2_000_000, description="PAY_AMT6")

    @field_validator("X2")
    @classmethod
    def validate_x2(cls, v: int) -> int:
        allowed = {1, 2}
        if v not in allowed:
            raise ValueError(f"X2 (SEX) must be one of {sorted(allowed)}")
        return v

    @field_validator("X3")
    @classmethod
    def validate_x3(cls, v: int) -> int:
        allowed = {0, 1, 2, 3, 4, 5, 6}
        if v not in allowed:
            raise ValueError(f"X3 (EDUCATION) must be one of {sorted(allowed)}")
        return v

    @field_validator("X4")
    @classmethod
    def validate_x4(cls, v: int) -> int:
        allowed = {0, 1, 2, 3}
        if v not in allowed:
            raise ValueError(f"X4 (MARRIAGE) must be one of {sorted(allowed)}")
        return v


class PredictResponse(BaseModel):
    proba_default: float
    threshold: float
    mode: str
    prediction: int


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid

        response: Response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response


app.add_middleware(RequestIdMiddleware)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = getattr(request.state, "request_id", "-")
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "unhandled_exception",
            extra={
                "event": "http_error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": round(latency_ms, 3),
            },
        )

        if request.url.path == "/predict":
            PRED_RESPONSES_TOTAL.labels(status_code="500").inc()
            PRED_ERRORS_TOTAL.labels(error_type="internal_error").inc()
            PRED_LATENCY_MS.observe(latency_ms)

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "request_id": request_id},
        )

    latency_ms = (time.perf_counter() - start) * 1000

    response.headers["x-latency-ms"] = f"{latency_ms:.2f}"

    logger.info(
        "request",
        extra={
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": int(response.status_code),
            "latency_ms": round(latency_ms, 3),
        },
    )

    if request.url.path == "/predict":
        PRED_RESPONSES_TOTAL.labels(status_code=str(response.status_code)).inc()
        PRED_LATENCY_MS.observe(latency_ms)

        if response.status_code >= 500:
            PRED_ERRORS_TOTAL.labels(error_type="server_error").inc()
        elif response.status_code >= 400:
            PRED_ERRORS_TOTAL.labels(error_type="client_error").inc()

    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": APP_ENV,
        "default_mode": DEFAULT_MODE,
        "active_model_version": artifact_subdir,
        "mlflow_run_id": LATEST_META.get("mlflow_run_id"),
        "model_path": str(MODEL_PATH),
        "model_file": MODEL_FILENAME,
        "using_calibrated_model": USE_CALIBRATED_MODEL,
        "model_select_mode": MODEL_SELECT_MODE,
        "configured_model_version": MODEL_VERSION,
        "champion_model_version": CHAMPION_META.get("model_version"),
        "latest_model_version": LATEST_META.get("model_version"),
        "artifact_dir": str(ARTIFACTS_DIR),
        "threshold_file": THRESH_FILENAME,
        "threshold_modes": list(THRESHOLDS.keys()),
        "thresholds": THRESHOLDS,
        "model_type": type(MODEL).__name__,
        "feature_count": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "run_meta": {
            "created_at_utc": RUN_META.get("created_at_utc"),
            "model_version": RUN_META.get("model_version"),
            "sklearn_version": RUN_META.get("sklearn_version"),
        },
    }


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    return {
        "artifact_dir": str(ARTIFACTS_DIR),
        "model_path": str(MODEL_PATH),
        "model_file": MODEL_FILENAME,
        "using_calibrated_model": USE_CALIBRATED_MODEL,
        "threshold_file": THRESH_FILENAME,
        "thresholds": THRESHOLDS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_select_mode": MODEL_SELECT_MODE,
        "configured_model_version": MODEL_VERSION,
        "target": TARGET_COLUMN,
        "default_mode": DEFAULT_MODE,
        "active_model_version": artifact_subdir,
        "champion_model_version": CHAMPION_META.get("model_version"),
        "latest_model_version": LATEST_META.get("model_version"),
        "mlflow_run_id": LATEST_META.get("mlflow_run_id"),
        "latest_path": str(LATEST_PATH),
    }


def get_pipeline_and_model():
    # uncalibrated pipeline
    if hasattr(MODEL, "named_steps"):
        if "model" not in MODEL.named_steps:
            raise HTTPException(status_code=400, detail="Pipeline missing model step")
        if "prep" not in MODEL.named_steps:
            raise HTTPException(status_code=400, detail="Pipeline missing prep step")

        prep = MODEL.named_steps["prep"]
        model = MODEL.named_steps["model"]
        return prep, model

    # extract inner pipeline
    if hasattr(MODEL, "estimator") and hasattr(MODEL.estimator, "named_steps"):
        inner = MODEL.estimator

        if "model" not in inner.named_steps:
            raise HTTPException(
                status_code=400, detail="Inner pipeline missing model step"
            )
        if "prep" not in inner.named_steps:
            raise HTTPException(
                status_code=400, detail="Inner pipeline missing prep step"
            )

        prep = inner.named_steps["prep"]
        model = inner.named_steps["model"]
        return prep, model

    raise HTTPException(status_code=400, detail="Model pipeline not found")


def get_transformed_feature_names(prep) -> list[str]:
    if hasattr(prep, "get_feature_names_out"):
        return list(prep.get_feature_names_out())
    raise HTTPException(
        status_code=400, detail="Preprocessor does not support get_feature_names_out"
    )


def normalize_feature_name(raw_name: str) -> str:
    if "__" in raw_name:
        raw_name = raw_name.split("__", 1)[1]

    for i in range(1, 24):
        prefix = f"X{i}"
        if raw_name == prefix or raw_name.startswith(prefix + "_"):
            return prefix

    return raw_name


@app.get("/explain/feature_importance/raw")
def feature_importance_raw():
    prep, model = get_pipeline_and_model()

    if not hasattr(model, "feature_importances_"):
        raise HTTPException(
            status_code=400,
            detail="Current model does not support feature_importances_",
        )

    feature_names = get_transformed_feature_names(prep)
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise HTTPException(
            status_code=500,
            detail=f"Feature name count ({len(feature_names)}) does not match importance count ({len(importances)})",
        )

    pairs = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(feature_names, importances)
    ]
    pairs = sorted(pairs, key=lambda x: x["importance"], reverse=True)

    return {
        "model_type": type(model).__name__,
        "view": "raw",
        "feature_importance": pairs,
    }


@app.get("/explain/feature_importance/grouped")
def feature_importance_grouped():
    prep, model = get_pipeline_and_model()

    if not hasattr(model, "feature_importances_"):
        raise HTTPException(
            status_code=400,
            detail="Current model does not support feature_importances_",
        )

    feature_names = get_transformed_feature_names(prep)
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise HTTPException(
            status_code=500,
            detail=f"Feature name count ({len(feature_names)}) does not match importance count ({len(importances)})",
        )

    grouped: Dict[str, float] = {}

    for f, imp in zip(feature_names, importances):
        base_name = normalize_feature_name(f)
        grouped[base_name] = grouped.get(base_name, 0.0) + float(imp)

    pairs = [{"feature": f, "importance": float(i)} for f, i in grouped.items()]
    pairs = sorted(pairs, key=lambda x: x["importance"], reverse=True)

    return {
        "model_type": type(model).__name__,
        "view": "grouped",
        "feature_importance": pairs,
    }


@app.get("/explain/feature_importance")
def feature_importance():
    return feature_importance_grouped()


def build_single_row_dataframe(payload: PredictRequest) -> pd.DataFrame:
    row = payload.model_dump()
    return pd.DataFrame([row])


def get_shap_explainer_and_transformed_data(X: pd.DataFrame):
    prep, model = get_pipeline_and_model()

    if not hasattr(prep, "transform"):
        raise HTTPException(
            status_code=400, detail="Preprocessor does not support transform"
        )

    X_transformed = prep.transform(X)

    feature_names = get_transformed_feature_names(prep)

    model_name = type(model).__name__.lower()
    supported_tree_models = (
        "randomforestclassifier",
        "histgradientboostingclassifier",
        "extratreesclassifier",
        "decisiontreeclassifier",
    )

    if model_name not in supported_tree_models:
        raise HTTPException(
            status_code=400,
            detail=f"SHAP currently supported for tree-based models only. Current model: {type(model).__name__}",
        )

    return model, X_transformed, feature_names


def group_shap_pairs(pairs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[str, float] = {}

    for item in pairs:
        base_name = normalize_feature_name(item["feature"])
        grouped[base_name] = grouped.get(base_name, 0.0) + float(item["shap_value"])

    grouped_pairs = [
        {
            "feature": k,
            "shap_value": float(v),
            "abs_shap_value": float(abs(v)),
        }
        for k, v in grouped.items()
    ]

    grouped_pairs = sorted(
        grouped_pairs, key=lambda x: x["abs_shap_value"], reverse=True
    )
    return grouped_pairs


@app.post("/explain/predict/grouped")
def explain_predict_grouped(
    payload: PredictRequest,
    mode: Literal["recall80", "maxf1", "fixed_0_5"] = DEFAULT_MODE,
):
    import shap
    import numpy as np

    if mode not in THRESHOLDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode={mode}. Available={list(THRESHOLDS.keys())}",
        )

    X = build_single_row_dataframe(payload)

    proba = float(MODEL.predict_proba(X)[:, 1][0])
    thr = float(THRESHOLDS[mode])
    pred = int(proba >= thr)

    model, X_transformed, feature_names = get_shap_explainer_and_transformed_data(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:
            sv = arr[0, :, 1]
        else:
            sv = arr[0]

    raw_pairs = [
        {
            "feature": f,
            "shap_value": float(v),
            "abs_shap_value": float(abs(v)),
        }
        for f, v in zip(feature_names, sv)
    ]

    grouped_pairs = group_shap_pairs(raw_pairs)

    top_positive = [
        {"feature": p["feature"], "shap_value": p["shap_value"]}
        for p in grouped_pairs
        if p["shap_value"] > 0
    ][:5]

    top_negative = [
        {"feature": p["feature"], "shap_value": p["shap_value"]}
        for p in grouped_pairs
        if p["shap_value"] < 0
    ][:5]

    return {
        "prediction": pred,
        "proba_default": proba,
        "threshold": thr,
        "mode": mode,
        "model_type": type(model).__name__,
        "view": "grouped",
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "top_features_overall": grouped_pairs[:10],
    }


@app.post("/explain/predict")
def explain_predict(
    payload: PredictRequest,
    mode: Literal["recall80", "maxf1", "fixed_0_5"] = DEFAULT_MODE,
):
    import shap
    import numpy as np

    if mode not in THRESHOLDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode={mode}. Available={list(THRESHOLDS.keys())}",
        )

    X = build_single_row_dataframe(payload)

    # normal prediction
    proba = float(MODEL.predict_proba(X)[:, 1][0])
    thr = float(THRESHOLDS[mode])
    pred = int(proba >= thr)

    model, X_transformed, feature_names = get_shap_explainer_and_transformed_data(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:
            sv = arr[0, :, 1]
        else:
            sv = arr[0]

    pairs = [
        {
            "feature": f,
            "shap_value": float(v),
            "abs_shap_value": float(abs(v)),
        }
        for f, v in zip(feature_names, sv)
    ]

    pairs_sorted = sorted(pairs, key=lambda x: x["abs_shap_value"], reverse=True)

    top_positive = [
        {"feature": p["feature"], "shap_value": p["shap_value"]}
        for p in pairs_sorted
        if p["shap_value"] > 0
    ][:5]

    top_negative = [
        {"feature": p["feature"], "shap_value": p["shap_value"]}
        for p in pairs_sorted
        if p["shap_value"] < 0
    ][:5]

    return {
        "prediction": pred,
        "proba_default": proba,
        "threshold": thr,
        "mode": mode,
        "model_type": type(model).__name__,
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "top_features_overall": pairs_sorted[:10],
    }


@app.get("/drift/summary")
def drift_summary(limit: int = Query(500, ge=1, le=5000)):
    if not BASELINE_STATS_PATH.exists():
        return {
            "status": "NO_BASELINE",
            "detail": f"Missing {str(BASELINE_STATS_PATH)}",
        }

    baseline_stats = json.loads(BASELINE_STATS_PATH.read_text(encoding="utf-8"))
    prod_df = load_prod_jsonl(PROD_PRED_PATH, max_rows=limit)

    summary = compute_drift_summary(
        baseline_stats=baseline_stats,
        prod_df=prod_df,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    return summary


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    request: Request,
    mode: Literal["recall80", "maxf1", "fixed_0_5"] = DEFAULT_MODE,
):
    t0 = time.perf_counter()
    request_id = getattr(request.state, "request_id", "-")

    PRED_REQUESTS_TOTAL.inc()
    PRED_MODE_TOTAL.labels(mode=mode).inc()

    status_code = "200"  # default

    try:
        if mode not in THRESHOLDS:
            status_code = "400"
            PRED_ERRORS_TOTAL.labels(error_type="bad_mode").inc()
            raise HTTPException(
                status_code=400,
                detail=f"Unknown mode={mode}. Available={list(THRESHOLDS.keys())}",
            )

        # request -> dataframe
        row = payload.model_dump()
        X = pd.DataFrame([row])

        # proba
        proba = float(MODEL.predict_proba(X)[:, 1][0])

        # decision
        thr = float(THRESHOLDS[mode])
        pred = int(proba >= thr)

        if pred == 1:
            PRED_RISKY_TOTAL.inc()

        # ✅ latency hesapla (return'den ÖNCE)
        latency_ms = (time.perf_counter() - t0) * 1000

        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "mode": mode,
            "threshold": thr,
            "proba_default": proba,
            "prediction": pred,
            "features": row,
            "latency_ms": round(latency_ms, 3),
        }

        with open(PROD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "predict",
            extra={
                "event": "predict",
                "request_id": request_id,
                "mode": mode,
                "threshold": thr,
                "proba_default": proba,
                "prediction": pred,
                "latency_ms": round(latency_ms, 3),
            },
        )

        return PredictResponse(
            proba_default=proba,
            threshold=thr,
            mode=mode,
            prediction=pred,
        )

    except HTTPException as e:
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.warning(
            "predict_error",
            extra={
                "event": "predict_error",
                "request_id": request_id,
                "error_type": "http_exception",
                "status_code": e.status_code,
                "detail": e.detail,
                "latency_ms": round(latency_ms, 3),
            },
        )

        status_code = str(e.status_code)
        raise

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000

        status_code = "500"
        PRED_ERRORS_TOTAL.labels(error_type="internal_error").inc()

        logger.exception(
            "predict_exception",
            extra={
                "event": "predict_exception",
                "request_id": request_id,
                "error_type": type(e).__name__,
                "status_code": 500,
                "latency_ms": round(latency_ms, 3),
            },
        )

        raise

    finally:
        PRED_RESPONSES_TOTAL.labels(status_code=status_code).inc()


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_artifact_dir_from_meta(meta: Dict[str, Any]) -> Path:
    artifact_subdir = meta.get("artifact_subdir")
    if not artifact_subdir:
        raise HTTPException(status_code=500, detail="artifact_subdir missing in meta")
    return PROJECT_ROOT / "artifacts" / artifact_subdir


def get_pr_auc_from_metrics(metrics_path: Path, split: str = "test") -> float | None:
    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    try:
        return float(metrics[split]["recall>=0.80"]["pr_auc"])
    except Exception:
        return None


@app.get("/registry/status")
def registry_status() -> Dict[str, Any]:
    return {
        "model_select_mode": MODEL_SELECT_MODE,
        "configured_model_version": MODEL_VERSION,
        "active_model_version": artifact_subdir,
        "latest_model_version": LATEST_META.get("model_version"),
        "champion_model_version": CHAMPION_META.get("model_version"),
        "latest_artifact_subdir": LATEST_META.get("artifact_subdir"),
        "champion_artifact_subdir": CHAMPION_META.get("artifact_subdir"),
    }


@app.get("/models/compare")
def models_compare() -> Dict[str, Any]:
    if not LATEST_META:
        raise HTTPException(status_code=404, detail="latest.json not found or empty")
    if not CHAMPION_META:
        raise HTTPException(status_code=404, detail="champion.json not found or empty")

    latest_dir = get_artifact_dir_from_meta(LATEST_META)
    champion_dir = get_artifact_dir_from_meta(CHAMPION_META)

    latest_metrics_path = latest_dir / "metrics.json"
    champion_metrics_path = champion_dir / "metrics.json"

    latest_score = get_pr_auc_from_metrics(latest_metrics_path, split="test")
    champion_score = get_pr_auc_from_metrics(champion_metrics_path, split="test")

    improvement = None
    decision = "unknown"

    if latest_score is not None and champion_score is not None:
        improvement = latest_score - champion_score
        decision = "promote" if latest_score > champion_score else "keep_champion"

    return {
        "active_model_version": artifact_subdir,
        "model_select_mode": MODEL_SELECT_MODE,
        "latest": {
            "model_version": LATEST_META.get("model_version"),
            "artifact_subdir": LATEST_META.get("artifact_subdir"),
            "mlflow_run_id": LATEST_META.get("mlflow_run_id"),
            "test_pr_auc_recall80": latest_score,
        },
        "champion": {
            "model_version": CHAMPION_META.get("model_version"),
            "artifact_subdir": CHAMPION_META.get("artifact_subdir"),
            "mlflow_run_id": CHAMPION_META.get("mlflow_run_id"),
            "test_pr_auc_recall80": champion_score,
        },
        "improvement": improvement,
        "decision": decision,
    }


def get_active_artifact_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / artifact_subdir


@app.get("/model/calibration")
def model_calibration():
    calibration_path = get_active_artifact_dir() / "calibration.json"

    if not calibration_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"calibration.json not found for active model: {artifact_subdir}",
        )

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    return {
        "active_model_version": artifact_subdir,
        "model_select_mode": MODEL_SELECT_MODE,
        "calibration": calibration,
    }


@app.get("/debug/data")
def debug_data():
    df = load_raw_data()
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "target_counts": df["default"].value_counts().to_dict(),
    }
