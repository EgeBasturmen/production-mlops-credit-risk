import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

APP_ENV = os.getenv("APP_ENV", "dev")
MODEL_VERSION = os.getenv("MODEL_VERSION", "model_v2")
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "recall80")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / MODEL_VERSION
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
THRESH_PATH = ARTIFACTS_DIR / "thresholds.json"
BASELINE_STATS_PATH = ARTIFACTS_DIR / "baseline_feature_stats.json"
RUN_META_PATH = ARTIFACTS_DIR / "run_meta.json"
CHAMPION_PATH = PROJECT_ROOT / "artifacts" / "champion.json"
PROD_PRED_PATH = PROJECT_ROOT / "data" / "prod" / "predictions.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"

# data paths (eski olanlar)
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

LATEST_PATH = PROJECT_ROOT / "artifacts" / "latest.json"
MODEL_SELECT_MODE = os.getenv("MODEL_SELECT_MODE", "latest")

RAW_DATA_FILE = "default_of_credit_card_clients.xls"
TARGET_COLUMN = "default"
RANDOM_STATE = 42
