from pathlib import Path

from credit_risk.data_load import load_raw_data
from credit_risk.schema import build_schema, save_schema
from credit_risk.config import PROJECT_ROOT


def main() -> None:
    df = load_raw_data()
    schema = build_schema(df)
    out_path = Path(PROJECT_ROOT) / "artifacts" / "data" / "schema.json"
    save_schema(schema, out_path)
    print(f"✅ schema saved to: {out_path}")


if __name__ == "__main__":
    main()
