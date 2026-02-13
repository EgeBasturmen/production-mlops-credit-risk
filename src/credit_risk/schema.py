from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from credit_risk.config import TARGET_COLUMN


def infer_column_type(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        # numeric ama integer mı float mı?
        return "int" if pd.api.types.is_integer_dtype(s) else "float"
    return "categorical"


def build_schema(df: pd.DataFrame) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target": TARGET_COLUMN,
        "columns": {},
    }

    for col in df.columns:
        s = df[col]
        col_type = infer_column_type(s)
        col_info: Dict[str, Any] = {
            "type": col_type,
            "nullable": bool(s.isna().any()),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }

        if col == TARGET_COLUMN:
            col_info["value_counts"] = s.value_counts(dropna=False).to_dict()

        if col_type in {"int", "float"}:
            s2 = pd.to_numeric(s, errors="coerce")
            col_info.update(
                {
                    "min": float(s2.min()),
                    "max": float(s2.max()),
                    "mean": float(s2.mean()),
                }
            )
        else:
            # kategorik gibi davran: en sık 5 değer
            top = s.astype(str).value_counts().head(5).to_dict()
            col_info["top_values"] = top

        schema["columns"][col] = col_info

    return schema


def save_schema(schema: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
