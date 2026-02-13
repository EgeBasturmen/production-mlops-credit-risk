import pandas as pd
from credit_risk.config import DATA_RAW_PATH, RAW_DATA_FILE, TARGET_COLUMN


def load_raw_data() -> pd.DataFrame:
    file_path = DATA_RAW_PATH / RAW_DATA_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_excel(file_path)



    # 1) Gereksiz index kolonu
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    numeric_cols = ["X1", "X5"] + [f"X{i}" for i in range(12, 24)]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Target kolonunu standardize et
    if "Y" in df.columns and TARGET_COLUMN not in df.columns:
        df = df.rename(columns={"Y": TARGET_COLUMN})

    # 3) Target sadece 0/1 olmalı (bozuk satırları temizle)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df
