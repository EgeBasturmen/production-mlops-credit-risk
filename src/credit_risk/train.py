from __future__ import annotations
import platform
import subprocess
from datetime import datetime
import sklearn
import json
from pathlib import Path
from typing import Dict, Any
from credit_risk.config import PROJECT_ROOT
import numpy as np
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

from credit_risk.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from credit_risk.schema import TARGET_COLUMN
from credit_risk.config import PROCESSED_DIR
from credit_risk.baseline_stats import compute_baseline_feature_stats


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def make_ohe():
    """
    scikit-learn version compatibility:
    - older: OneHotEncoder(..., sparse=...)
    - newer: OneHotEncoder(..., sparse_output=...)
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_split(name: str) -> pd.DataFrame:
    """
    Loads a parquet split from data/processed/ (or wherever PROCESSED_DIR points).
    """
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return pd.read_parquet(path)


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.80
) -> float:
    """
    Finds a threshold that achieves at least min_recall.
    We align thresholds with recall/precision using recall[:-1].
    Scan from high threshold -> low threshold and return first meeting recall >= min_recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5

    recall_t = recall[:-1]        # aligned to thresholds
    thresholds_t = thresholds

    for r, t in zip(recall_t[::-1], thresholds_t[::-1]):
        if r >= min_recall:
            return float(t)

    # If nothing meets min_recall, return the minimum threshold (most permissive)
    return float(thresholds_t.min())


def threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds threshold maximizing F1 on PR curve.
    Must align thresholds with precision/recall (use precision[:-1], recall[:-1]).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5

    p = precision[:-1]
    r = recall[:-1]
    f1s = (2 * p * r) / (p + r + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(thresholds[idx])


def _json_dump(path: Path, obj: Any) -> None:
    """
    JSON dump with numpy-safe casting.
    """
    def _default(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return str(x)

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    # -----------------------------------------------------
    # 1) Load data
    # -----------------------------------------------------
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN].astype(int).values

    X_val = val.drop(columns=[TARGET_COLUMN])
    y_val = val[TARGET_COLUMN].astype(int).values

    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN].astype(int).values

    # -----------------------------------------------------
    # 2) Pipeline
    # -----------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_ohe(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    # -----------------------------------------------------
    # 3) Output dir
    # -----------------------------------------------------
    out_dir = PROJECT_ROOT / "artifacts" / "model_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 4) MLFLOW
    # -----------------------------------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk-default")

    with mlflow.start_run(run_name="logreg_onehot_baseline_v2") as run:
        mlflow.set_tags({
            "stage": "candidate",
            "model_family": "logreg",
            "version": "v2",
            "target": TARGET_COLUMN,
            "data_splits": "train/val/test",
        })

        mlflow.log_params({
            "numeric_scaler": "StandardScaler",
            "categorical_encoder": "OneHotEncoder",
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "n_numeric_features": len(NUMERIC_FEATURES),
            "n_categorical_features": len(CATEGORICAL_FEATURES),
        })

        # -------------------------------------------------
        # 5) Train
        # -------------------------------------------------
        pipe.fit(X_train, y_train)

        # -------------------------------------------------
        # 6) Thresholds on VAL
        # -------------------------------------------------
        val_prob = pipe.predict_proba(X_val)[:, 1]

        t_recall = threshold_for_recall(y_val, val_prob, min_recall=0.80)
        t_f1 = threshold_max_f1(y_val, val_prob)

        # -------------------------------------------------
        # 7) Evaluate
        # -------------------------------------------------
        val_metrics_recall = evaluate_at_threshold(y_val, val_prob, t_recall)
        val_metrics_f1 = evaluate_at_threshold(y_val, val_prob, t_f1)
        val_metrics_05 = evaluate_at_threshold(y_val, val_prob, 0.5)

        test_prob = pipe.predict_proba(X_test)[:, 1]

        test_metrics_recall = evaluate_at_threshold(y_test, test_prob, t_recall)
        test_metrics_f1 = evaluate_at_threshold(y_test, test_prob, t_f1)
        test_metrics_05 = evaluate_at_threshold(y_test, test_prob, 0.5)

        # -------------------------------------------------
        # 8) Save local artifacts
        # -------------------------------------------------
        joblib.dump(pipe, out_dir / "model.joblib")

        metrics_payload = {
            "val": {
                "recall>=0.80": val_metrics_recall,
                "max_f1": val_metrics_f1,
                "0.5": val_metrics_05,
            },
            "test": {
                "recall>=0.80": test_metrics_recall,
                "max_f1": test_metrics_f1,
                "0.5": test_metrics_05,
            },
        }
        _json_dump(out_dir / "metrics.json", metrics_payload)

        thresholds_payload = {
            "recall80": float(t_recall),
            "maxf1": float(t_f1),
            "fixed_0_5": 0.5,
        }
        _json_dump(out_dir / "thresholds.json", thresholds_payload)

        # ✅ Baseline stats for drift monitoring
        baseline_stats = compute_baseline_feature_stats(X_train)
        baseline_stats["target_mean"] = float(y_train.mean())
        baseline_stats["n_train_rows"] = int(len(y_train))
        _json_dump(out_dir / "baseline_feature_stats.json", baseline_stats)

        # ✅ run metadata (model provenance)
        run_meta = {
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "git_commit": _get_git_commit(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "mlflow_run_id": run.info.run_id,
            "artifact_dir": str(out_dir.resolve()),
            "data": {
                "train_rows": int(X_train.shape[0]),
                "val_rows": int(X_val.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "target_mean_train": float(y_train.mean()),
            },
        }

        _json_dump(out_dir / "run_meta.json", run_meta)

        # -------------------------------------------------
        # 9) MLFLOW LOGGING
        # -------------------------------------------------
        mlflow.log_metrics({
            "val_pr_auc": val_metrics_recall["pr_auc"],
            "test_pr_auc": test_metrics_recall["pr_auc"],

            "val_f1_recall80": val_metrics_recall["f1"],
            "test_f1_recall80": test_metrics_recall["f1"],
            "threshold_recall80": float(t_recall),

            "val_f1_maxf1": val_metrics_f1["f1"],
            "test_f1_maxf1": test_metrics_f1["f1"],
            "threshold_maxf1": float(t_f1),

            "val_f1_05": val_metrics_05["f1"],
            "test_f1_05": test_metrics_05["f1"],
        })

        # log local artifacts folder (includes baseline + thresholds)
        mlflow.log_artifacts(str(out_dir))

        # signature + input_example
        input_example = X_train.head(200).copy()

        # If integer columns may have missing at inference time, cast to float
        for c in input_example.columns:
            if pd.api.types.is_integer_dtype(input_example[c]):
                input_example[c] = input_example[c].astype("float64")

        signature = infer_signature(
            input_example,
            pipe.predict_proba(input_example)[:, 1],
        )

        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=input_example,
        )

    # -----------------------------------------------------
    # 10) Print summary
    # -----------------------------------------------------
    print("\n--- VALIDATION ---")
    print("recall>=0.80:", val_metrics_recall)
    print("max_f1:", val_metrics_f1)
    print("0.5:", val_metrics_05)

    print("\n--- TEST ---")
    print("recall>=0.80:", test_metrics_recall)
    print("max_f1:", test_metrics_f1)
    print("0.5:", test_metrics_05)

    print(f"\nArtifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
