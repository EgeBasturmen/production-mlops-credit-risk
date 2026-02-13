from pathlib import Path

from credit_risk.config import PROJECT_ROOT
from credit_risk.data_load import load_raw_data
from credit_risk.data_report import build_data_report, render_markdown


def main() -> None:
    df = load_raw_data()
    report = build_data_report(df)
    md = render_markdown(report)

    out_path = Path(PROJECT_ROOT) / "artifacts" / "data" / "data_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print(f"✅ report saved to: {out_path}")


if __name__ == "__main__":
    main()
