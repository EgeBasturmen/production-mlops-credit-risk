from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from credit_risk.baseline_stats import compute_baseline_feature_stats
from credit_risk.config import PROCESSED_DIR, PROJECT_ROOT
from credit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.schema import TARGET_COLUMN


def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_split(name: str) -> pd.DataFrame:
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


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "calibration_curve": {
            "mean_predicted_value": [float(x) for x in mean_pred],
            "fraction_of_positives": [float(x) for x in frac_pos],
        },
        "n_bins": int(n_bins),
    }


def threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.80
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5

    recall_t = recall[:-1]
    thresholds_t = thresholds

    for r, t in zip(recall_t[::-1], thresholds_t[::-1]):
        if r >= min_recall:
            return float(t)

    return float(thresholds_t.min())


def threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5

    p = precision[:-1]
    r = recall[:-1]
    f1s = (2 * p * r) / (p + r + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(thresholds[idx])


def _json_dump(path: Path, obj: Any) -> None:
    def _default(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return str(x)

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def main() -> None:
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN].astype(int).values

    X_val = val.drop(columns=[TARGET_COLUMN])
    y_val = val[TARGET_COLUMN].astype(int).values

    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN].astype(int).values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    out_dir = PROJECT_ROOT / "artifacts" / "model_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk-default")

    with mlflow.start_run(run_name="hist_gradient_boosting_candidate_v4") as run:
        mlflow.set_tags(
            {
                "stage": "candidate",
                "model_family": "hist_gradient_boosting",
                "version": "v4",
                "target": TARGET_COLUMN,
                "data_splits": "train/val/test",
            }
        )

        mlflow.log_params(
            {
                "numeric_scaler": "StandardScaler",
                "categorical_encoder": "OrdinalEncoder",
                "model": "HistGradientBoostingClassifier",
                "learning_rate": 0.05,
                "max_depth": 6,
                "max_iter": 300,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "random_state": 42,
                "n_numeric_features": len(NUMERIC_FEATURES),
                "n_categorical_features": len(CATEGORICAL_FEATURES),
            }
        )

        pipe.fit(X_train, y_train)

        val_prob = pipe.predict_proba(X_val)[:, 1]
        t_recall = threshold_for_recall(y_val, val_prob, min_recall=0.80)
        t_f1 = threshold_max_f1(y_val, val_prob)

        val_metrics_recall = evaluate_at_threshold(y_val, val_prob, t_recall)
        val_metrics_f1 = evaluate_at_threshold(y_val, val_prob, t_f1)
        val_metrics_05 = evaluate_at_threshold(y_val, val_prob, 0.5)

        test_prob = pipe.predict_proba(X_test)[:, 1]
        test_metrics_recall = evaluate_at_threshold(y_test, test_prob, t_recall)
        test_metrics_f1 = evaluate_at_threshold(y_test, test_prob, t_f1)
        test_metrics_05 = evaluate_at_threshold(y_test, test_prob, 0.5)

        val_calibration = compute_calibration_metrics(y_val, val_prob, n_bins=10)
        test_calibration = compute_calibration_metrics(y_test, test_prob, n_bins=10)

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

        calibration_payload = {
            "val": val_calibration,
            "test": test_calibration,
        }
        _json_dump(out_dir / "calibration.json", calibration_payload)

        baseline_stats = compute_baseline_feature_stats(X_train)
        baseline_stats["target_mean"] = float(y_train.mean())
        baseline_stats["n_train_rows"] = int(len(y_train))
        _json_dump(out_dir / "baseline_feature_stats.json", baseline_stats)

        run_meta = {
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "git_commit": _get_git_commit(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "mlflow_run_id": run.info.run_id,
            "model_version": out_dir.name,
            "artifact_dir": str(out_dir.resolve()),
            "data": {
                "train_rows": int(X_train.shape[0]),
                "val_rows": int(X_val.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "target_mean_train": float(y_train.mean()),
            },
        }
        _json_dump(out_dir / "run_meta.json", run_meta)

        latest_payload = {
            "model_version": out_dir.name,
            "artifact_subdir": out_dir.name,
            "mlflow_run_id": run.info.run_id,
            "created_at_utc": run_meta["created_at_utc"],
        }
        latest_path = PROJECT_ROOT / "artifacts" / "latest.json"
        _json_dump(latest_path, latest_payload)

        mlflow.log_metrics(
            {
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
                "val_brier_score": val_calibration["brier_score"],
                "test_brier_score": test_calibration["brier_score"],
            }
        )

        mlflow.log_artifacts(str(out_dir))

        input_example = X_train.head(200).copy()
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
