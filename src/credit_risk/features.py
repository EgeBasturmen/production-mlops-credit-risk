from __future__ import annotations

# Target
TARGET = "default"

# Numeric features
NUMERIC_FEATURES = ["X1", "X5"] + [f"X{i}" for i in range(12, 24)]

# Categorical features
CATEGORICAL_FEATURES = [f"X{i}" for i in range(2, 5)] + [f"X{i}" for i in range(6, 12)]
