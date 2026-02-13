from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Data paths
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

# Dataset
RAW_DATA_FILE = "default_of_credit_card_clients.xls"  # gerekirse değiştireceğiz

# Target column
TARGET_COLUMN = "default"

# Random seed
RANDOM_STATE = 42

PROD_PRED_PATH = Path("data/prod/predictions.jsonl")