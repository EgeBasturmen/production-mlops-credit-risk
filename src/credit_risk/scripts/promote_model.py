from __future__ import annotations

import json
from datetime import datetime

from credit_risk.config import LATEST_PATH, PROJECT_ROOT


def main() -> None:
    if not LATEST_PATH.exists():
        raise RuntimeError(f"latest.json not found: {LATEST_PATH}")
    latest = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
    required_keys = {
        "model_version",
        "artifact_subdir",
        "mlflow_run_id",
        "created_at_utc",
    }

    missing = required_keys - set(latest.keys())
    if missing:
        raise RuntimeError(f"latest.json missing keys:{sorted(missing)}")
    champion_payload = {
        "model_version": latest["model_version"],
        "artifact_subdir": latest["artifact_subdir"],
        "mlflow_run_id": latest["mlflow_run_id"],
        "promoted_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_latest_created_at_utc": latest["created_at_utc"],
    }

    champion_path = PROJECT_ROOT / "artifacts" / "champion.json"
    champion_path.write_text(
        json.dumps(champion_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"✅ champion.json updated: {champion_path}")
    print(json.dumps(champion_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
