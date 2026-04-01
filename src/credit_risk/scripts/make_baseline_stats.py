import json
from pathlib import Path

import pandas as pd

from credit_risk.baseline_stats import compute_baseline_feature_stats
from credit_risk.config import PROJECT_ROOT


def main() -> None:
    train_path = Path(PROJECT_ROOT) / "data" / "processed" / "train.parquet"
    df = pd.read_parquet(train_path)

    stats = compute_baseline_feature_stats(df)

    out_path = (
        Path(PROJECT_ROOT) / "artifacts" / "model_v1" / "baseline_feature_stats.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"✅ baseline_feature_stats saved to: {out_path}")


if __name__ == "__main__":
    main()
