from pathlib import Path

from credit_risk.config import PROJECT_ROOT
from credit_risk.data_load import load_raw_data
from credit_risk.split import split_train_val_test


def main() -> None:
    df = load_raw_data()
    res = split_train_val_test(df, test_size=0.20, val_size=0.10)

    out_dir = Path(PROJECT_ROOT) / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    res.train.to_parquet(out_dir / "train.parquet", index=False)
    res.val.to_parquet(out_dir / "val.parquet", index=False)
    res.test.to_parquet(out_dir / "test.parquet", index=False)

    print("✅ Saved splits to data/processed/")
    print(f"train: {res.train.shape} | val: {res.val.shape} | test: {res.test.shape}")
    print("Target ratios:")
    for name, d in [("train", res.train), ("val", res.val), ("test", res.test)]:
        r = d["default"].mean()
        print(f" - {name}: default_rate={r:.4f}")


if __name__ == "__main__":
    main()
