from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
PROMOTION_LOG_PATH = ARTIFACTS_ROOT / "promotion_log.json1"
LATEST_PATH = ARTIFACTS_ROOT / "latest.json"
CHAMPION_PATH = ARTIFACTS_ROOT / "champion.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_model_dir(pointer_path: Path) -> Path:
    meta = read_json(pointer_path)

    if "artifact_subdir" not in meta:
        raise RuntimeError(f"{pointer_path.name} missing 'artifact_subdir'")

    return ARTIFACTS_ROOT / meta["artifact_subdir"]


def get_metric(metrics_path: Path, split: str, metric: str) -> float:
    metrics = read_json(metrics_path)

    try:
        return float(metrics[split]["recall>=0.80"][metric])
    except KeyError as e:
        raise RuntimeError(f"metrics.json missing key: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and promote model")

    parser.add_argument(
        "--metric", default="pr_auc", help="Metric name (default: pr_auc)"
    )
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"latest.json not found: {LATEST_PATH}")

    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(f"champion.json not found: {CHAMPION_PATH}")

    latest_meta = read_json(LATEST_PATH)
    champion_meta = read_json(CHAMPION_PATH)

    latest_dir = get_model_dir(LATEST_PATH)
    champion_dir = get_model_dir(CHAMPION_PATH)

    latest_metrics_path = latest_dir / "metrics.json"
    champion_metrics_path = champion_dir / "metrics.json"

    latest_score = get_metric(latest_metrics_path, args.split, args.metric)
    champion_score = get_metric(champion_metrics_path, args.split, args.metric)

    improvement = latest_score - champion_score

    decision = latest_score > champion_score + args.min_improvement

    result = {
        "event": "model_evaluation",
        "metric": args.metric,
        "split": args.split,
        "min_improvement": args.min_improvement,
        "latest_model": latest_meta["model_version"],
        "champion_model": champion_meta["model_version"],
        "latest_score": latest_score,
        "champion_score": champion_score,
        "improvement": improvement,
        "decision": "promote" if decision else "keep_champion",
        "dry_run": args.dry_run,
        "ts": datetime.utcnow().isoformat() + "Z",
    }

    print(json.dumps(result, indent=2))
    append_jsonl(PROMOTION_LOG_PATH, result)

    if decision and not args.dry_run:
        new_champion = {
            "model_version": latest_meta["model_version"],
            "artifact_subdir": latest_meta["artifact_subdir"],
            "mlflow_run_id": latest_meta.get("mlflow_run_id"),
            "promoted_at_utc": datetime.utcnow().isoformat() + "Z",
            "source_latest_created_at_utc": latest_meta.get("created_at_utc"),
            "promotion_reason": (
                f"{args.split}.{args.metric}: "
                f"{latest_score:.6f} >= {champion_score:.6f} + {args.min_improvement}"
            ),
        }

        CHAMPION_PATH.write_text(
            json.dumps(new_champion, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        append_jsonl(
            PROMOTION_LOG_PATH,
            {
                "event": "promotion_applied",
                "promoted_model": latest_meta["model_version"],
                "previous_champion": champion_meta["model_version"],
                "metric": args.metric,
                "split": args.split,
                "min_improvement": args.min_improvement,
                "latest_score": latest_score,
                "champion_score": champion_score,
                "ts": datetime.utcnow().isoformat() + "Z",
            },
        )

        print("\nPromotion applied.")
    elif decision and args.dry_run:
        print("\nDry-run: promotion would happen.")
    else:
        print("\nNo promotion.")


if __name__ == "__main__":
    main()
