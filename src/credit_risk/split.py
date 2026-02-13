from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import TARGET_COLUMN, RANDOM_STATE


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,  # train içinden ayrılacak val oranı
) -> SplitResult:
    # 1) test ayır
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    # 2) train_val içinden val ayır
    # val_size train_val'a göre uygulanır
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET_COLUMN],
    )

    return SplitResult(train=train, val=val, test=test)