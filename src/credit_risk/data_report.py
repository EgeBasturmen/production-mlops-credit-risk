from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from credit_risk.config import TARGET_COLUMN


@dataclass
class DataReport:
    n_rows: int
    n_cols: int
    target_counts: Dict[int, int]
    missing_top: List[Tuple[str, int, float]]
    numeric_outliers: List[Tuple[str, float, float, float]]  # col, p1, p99, outlier_rate
    high_cardinality: List[Tuple[str, int]]  # col, n_unique


def build_data_report(df: pd.DataFrame) -> DataReport:
    n_rows, n_cols = df.shape

    # Target distribution
    tc = df[TARGET_COLUMN].value_counts().to_dict()

    # Missing
    missing = df.isna().sum()
    missing_top = []
    for col, cnt in missing.sort_values(ascending=False).head(15).items():
        if cnt > 0:
            missing_top.append((col, int(cnt), float(cnt / n_rows)))

    # Outliers (numeric) - simple robust check via 1% and 99% percentiles
    numeric_outliers = []
    num_cols = [c for c in df.columns if c != TARGET_COLUMN and pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        p1 = float(np.percentile(s, 1))
        p99 = float(np.percentile(s, 99))
        # outlier rate beyond [p1, p99]
        out_rate = float(((df[c] < p1) | (df[c] > p99)).mean())
        numeric_outliers.append((c, p1, p99, out_rate))

    # High-cardinality categoricals (risk: too many categories)
    cat_cols = [c for c in df.columns if c != TARGET_COLUMN and not pd.api.types.is_numeric_dtype(df[c])]
    high_card = []
    for c in cat_cols:
        high_card.append((c, int(df[c].nunique(dropna=True))))
    high_card = sorted(high_card, key=lambda x: x[1], reverse=True)[:10]

    return DataReport(
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        target_counts={int(k): int(v) for k, v in tc.items()},
        missing_top=missing_top,
        numeric_outliers=sorted(numeric_outliers, key=lambda x: x[3], reverse=True)[:15],
        high_cardinality=high_card,
    )


def render_markdown(r: DataReport) -> str:
    lines = []
    lines.append("# Data Quality Report\n")
    lines.append(f"- Rows: **{r.n_rows}**")
    lines.append(f"- Cols: **{r.n_cols}**")
    lines.append(f"- Target: **{TARGET_COLUMN}**")
    lines.append("")

    lines.append("## Target distribution")
    lines.append("")
    for k, v in sorted(r.target_counts.items()):
        ratio = v / r.n_rows
        lines.append(f"- {k}: {v} ({ratio:.2%})")
    lines.append("")

    lines.append("## Missing values (top)")
    lines.append("")
    if not r.missing_top:
        lines.append("- ✅ No missing values found (after numeric coercion).")
    else:
        for col, cnt, rate in r.missing_top:
            lines.append(f"- **{col}**: {cnt} ({rate:.2%})")
    lines.append("")

    lines.append("## Numeric outliers (1%–99% range)")
    lines.append("")
    lines.append("> Outlier rate = fraction of rows outside [p1, p99].")
    lines.append("")
    for col, p1, p99, out_rate in r.numeric_outliers:
        lines.append(f"- **{col}**: p1={p1:.2f}, p99={p99:.2f}, outlier_rate={out_rate:.2%}")
    lines.append("")

    lines.append("## High-cardinality categorical columns")
    lines.append("")
    for col, nunique in r.high_cardinality:
        lines.append(f"- **{col}**: {nunique} unique")

    lines.append("")
    lines.append("## Notes / next checks")
    lines.append("")
    lines.append("- Leakage check: confirm what timestamp the score is generated (application-time vs behavior-time).")
    lines.append("- Consider mapping X1..X23 to meaningful names for readability.")
    lines.append("- Next step: stratified train/val/test split and write to `data/processed/`.")
    lines.append("")

    return "\n".join(lines)
