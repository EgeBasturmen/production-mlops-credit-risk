import numpy as np
import pandas as pd

from credit_risk.drift import psi_numeric, psi_categorical


def test_psi_numeric_same_distribution_near_zero():
    # aynı dağılımdan iki sample üret
    rng = np.random.default_rng(42)
    expected = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))
    actual = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))

    psi = psi_numeric(expected, actual, n_bins=10)

    # aynı dağılım => PSI çok küçük olmalı
    assert psi < 0.05


def test_psi_numeric_shifted_distribution_high():
    # expected: 0 merkezli
    # actual: 3 birim sağa kaymış (drift)
    rng = np.random.default_rng(42)
    expected = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))
    actual = pd.Series(rng.normal(loc=3.0, scale=1.0, size=2000))

    psi = psi_numeric(expected, actual, n_bins=10)

    # ciddi shift => PSI belirgin yüksek olmalı
    assert psi > 0.25


def test_psi_categorical_same_distribution_near_zero():
    # kategorik dağılım aynıysa PSI ~ 0
    expected = pd.Series(["A"] * 500 + ["B"] * 300 + ["C"] * 200)
    actual = pd.Series(["A"] * 500 + ["B"] * 300 + ["C"] * 200)

    psi = psi_categorical(expected, actual, top_k=20)
    assert psi < 0.01


def test_psi_categorical_distribution_changed_high():
    # expected: A ağırlıklı
    expected = pd.Series(["A"] * 800 + ["B"] * 150 + ["C"] * 50)

    # actual: B ağırlıklı (drift)
    actual = pd.Series(["A"] * 200 + ["B"] * 700 + ["C"] * 100)

    psi = psi_categorical(expected, actual, top_k=20)
    assert psi > 0.25
