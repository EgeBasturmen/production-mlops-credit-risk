from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import RANDOM_STATE, TARGET_COLUMN


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
) -> SplitResult:
    # test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    # train_val içinden val ayır

    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET_COLUMN],
    )

    return SplitResult(train=train, val=val, test=test)
