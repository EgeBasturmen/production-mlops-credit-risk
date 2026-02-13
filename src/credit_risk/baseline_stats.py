from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from credit_risk.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def compute_baseline_feature_stats(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"numeric": {}, "categorical": {}}

    # Numeric stats
    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        out["numeric"][col] = {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "q01": float(s.quantile(0.01)),
            "q05": float(s.quantile(0.05)),
            "q50": float(s.quantile(0.50)),
            "q95": float(s.quantile(0.95)),
            "q99": float(s.quantile(0.99)),
        }

    # Categorical distributions
    for col in CATEGORICAL_FEATURES:
        vc = df[col].astype(str).value_counts(dropna=False)
        dist = (vc / vc.sum()).to_dict()
        out["categorical"][col] = {
            "count": int(vc.sum()),
            "n_unique": int(df[col].nunique(dropna=True)),
            "distribution": {k: float(v) for k, v in dist.items()},
        }

    return out
