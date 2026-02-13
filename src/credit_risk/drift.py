from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def psi_numeric(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """
    PSI for numeric features using quantile bins from expected.
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if len(expected) < 50 or len(actual) < 50:
        return 0.0  # too little data

    qs = np.linspace(0, 1, n_bins + 1)
    bins = expected.quantile(qs).values
    bins = np.unique(bins)

    if len(bins) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, None)
    act_perc = np.clip(act_perc, eps, None)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def psi_categorical(expected: pd.Series, actual: pd.Series, top_k: int = 20) -> float:
    """
    PSI for categorical features:
    - take top_k categories from expected
    - bucket the rest into "__OTHER__"
    """
    expected = expected.astype(str).fillna("__NA__")
    actual = actual.astype(str).fillna("__NA__")

    exp_dist = expected.value_counts(normalize=True)
    top = exp_dist.head(top_k).index.tolist()

    def bucket(s: pd.Series) -> pd.Series:
        return s.where(s.isin(top), "__OTHER__")

    exp_b = bucket(expected)
    act_b = bucket(actual)

    exp_p = exp_b.value_counts(normalize=True)
    act_p = act_b.value_counts(normalize=True)

    cats = sorted(set(exp_p.index).union(set(act_p.index)))
    eps = 1e-6

    psi = 0.0
    for c in cats:
        e = float(exp_p.get(c, 0.0))
        a = float(act_p.get(c, 0.0))
        e = max(e, eps)
        a = max(a, eps)
        psi += (a - e) * math.log(a / e)

    return float(psi)


def load_prod_jsonl(path: Path, max_rows: int = 500) -> pd.DataFrame:
    """
    Reads last max_rows lines of jsonl. If file is large, this is still OK for now.
    """
    if not path.exists():
        return pd.DataFrame()

    lines: List[str] = path.read_text(encoding="utf-8").strip().splitlines()
    lines = lines[-max_rows:]

    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # expand features dict into columns
    if "features" in df.columns:
        feat = pd.json_normalize(df["features"])
        df = pd.concat([df.drop(columns=["features"]), feat], axis=1)

    return df


def compute_drift_summary(
    baseline_stats: Dict[str, Any],
    prod_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
) -> Dict[str, Any]:
    """
    baseline_stats: from baseline_feature_stats.json (train distribution)
    prod_df: last N prod predictions with feature columns
    """
    if prod_df.empty:
        return {
            "status": "NO_DATA",
            "n_prod_rows": 0,
            "feature_drift": {},
            "prediction_drift": {},
            "features_in_warn": [],
            "features_in_alert": [],
        }

    # prediction drift (default rate)
    if "prediction" in prod_df.columns:
        prod_default_rate = float(pd.to_numeric(prod_df["prediction"], errors="coerce").dropna().mean())
    else:
        prod_default_rate = None

    # baseline default rate (from stats json if present)
    train_default_rate = None
    try:
        train_default_rate = float(baseline_stats.get("target_mean", baseline_stats.get("default_rate", None)))
    except Exception:
        train_default_rate = None

    pred_status = "OK"
    pred_diff = None
    if prod_default_rate is not None and train_default_rate is not None:
        pred_diff = abs(prod_default_rate - train_default_rate)
        if pred_diff >= 0.10:
            pred_status = "ALERT"
        elif pred_diff >= 0.05:
            pred_status = "WARN"

    # feature drift
    feature_drift: Dict[str, Any] = {}
    features_in_warn: List[str] = []
    features_in_alert: List[str] = []

    # numeric
    for col in numeric_features:
        if col not in prod_df.columns:
            continue
        exp_col = None
        # if baseline has quantiles we can't reconstruct full dist, so we use train sample stats approach:
        # for PSI we need expected sample; in this simplified version we'll approximate using prod vs prod baseline bins derived from baseline quantiles not possible.
        # Instead: use prod bins from baseline quantiles stored in baseline_stats[col] if available.
        # If not available -> skip PSI for that col.
        # (We'll improve later by saving train sample snapshot.)
        bs = baseline_stats.get("numeric", {}).get(col, None)
        if not bs or "q01" not in bs:
            continue

        # create synthetic expected sample using quantiles (approx)
        # We'll use a cheap approach: sample values at stored quantiles
        exp = pd.Series([bs["q01"], bs["q05"], bs["q50"], bs["q95"], bs["q99"]]).repeat(200)
        act = pd.to_numeric(prod_df[col], errors="coerce").dropna()

        psi = psi_numeric(exp, act, n_bins=10)

        st = "OK"
        if psi >= psi_alert:
            st = "ALERT"
            features_in_alert.append(col)
        elif psi >= psi_warn:
            st = "WARN"
            features_in_warn.append(col)

        feature_drift[col] = {"psi": float(psi), "status": st, "type": "numeric"}

    # categorical
    for col in categorical_features:
        if col not in prod_df.columns:
            continue
        bs = baseline_stats.get("categorical", {}).get(col, None)
        if not bs or "distribution" not in bs:
            continue

        # build expected sample by distribution
        dist = bs["distribution"]
        exp_vals = []
        for k, p in dist.items():
            exp_vals.extend([k] * max(int(p * 1000), 1))
        exp = pd.Series(exp_vals)

        act = prod_df[col].astype(str).fillna("__NA__")
        psi = psi_categorical(exp, act, top_k=20)

        st = "OK"
        if psi >= psi_alert:
            st = "ALERT"
            features_in_alert.append(col)
        elif psi >= psi_warn:
            st = "WARN"
            features_in_warn.append(col)

        feature_drift[col] = {"psi": float(psi), "status": st, "type": "categorical"}

    # overall status
    status = "OK"
    if pred_status == "ALERT" or len(features_in_alert) > 0:
        status = "ALERT"
    elif pred_status == "WARN" or len(features_in_warn) > 0:
        status = "WARN"

    return {
        "status": status,
        "n_prod_rows": int(len(prod_df)),
        "prediction_drift": {
            "status": pred_status,
            "prod_default_rate": prod_default_rate,
            "train_default_rate": train_default_rate,
            "abs_diff": pred_diff,
        },
        "feature_drift": feature_drift,
        "features_in_warn": features_in_warn,
        "features_in_alert": features_in_alert,
    }
