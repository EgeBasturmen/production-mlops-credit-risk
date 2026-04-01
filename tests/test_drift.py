import numpy as np
import pandas as pd

from credit_risk.drift import psi_categorical, psi_numeric


def test_psi_numeric_same_distribution_near_zero():
    rng = np.random.default_rng(42)
    expected = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))
    actual = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))

    psi = psi_numeric(expected, actual, n_bins=10)

    assert psi < 0.05


def test_psi_numeric_shifted_distribution_high():
    rng = np.random.default_rng(42)
    expected = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))
    actual = pd.Series(rng.normal(loc=3.0, scale=1.0, size=2000))

    psi = psi_numeric(expected, actual, n_bins=10)

    assert psi > 0.25


def test_psi_categorical_same_distribution_near_zero():
    expected = pd.Series(["A"] * 500 + ["B"] * 300 + ["C"] * 200)
    actual = pd.Series(["A"] * 500 + ["B"] * 300 + ["C"] * 200)

    psi = psi_categorical(expected, actual, top_k=20)
    assert psi < 0.01


def test_psi_categorical_distribution_changed_high():
    expected = pd.Series(["A"] * 800 + ["B"] * 150 + ["C"] * 50)

    actual = pd.Series(["A"] * 200 + ["B"] * 700 + ["C"] * 100)

    psi = psi_categorical(expected, actual, top_k=20)
    assert psi > 0.25
