"""Tests cho statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stat_tests import (
    diebold_mariano,
    dm_pairwise_table,
    friedman_nemenyi,
)


def test_dm_identical_forecasts_no_significance() -> None:
    np.random.seed(0)
    y = np.random.randn(100)
    p = np.random.randn(100)
    r = diebold_mariano(y, p, p, h=1)
    # Identical → d=0 → dm_stat=0 → p=1
    assert r["dm_stat"] == pytest.approx(0.0, abs=1e-9)
    assert r["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_dm_perfect_vs_naive() -> None:
    np.random.seed(0)
    y = np.random.randn(200)
    perfect = y.copy()
    naive = np.zeros(200)
    r = diebold_mariano(y, perfect, naive, h=1, loss="MSE")
    # Perfect (loss=0) should be significantly better than naive (loss>0)
    assert r["dm_stat"] < 0  # A (perfect) better
    assert r["p_value"] < 0.001
    assert "***" in r["significance"]


def test_dm_pairwise_table_shape() -> None:
    np.random.seed(0)
    y = np.random.randn(100)
    preds = {
        "A": y + np.random.randn(100) * 0.5,
        "B": y + np.random.randn(100) * 1.0,
        "C": y + np.random.randn(100) * 0.3,
    }
    table = dm_pairwise_table(y, preds, h=1)
    assert table.shape == (3, 3)
    # Diagonal = 1.0
    assert all(table.iloc[i, i] == 1.0 for i in range(3))


def test_friedman_nemenyi_basic() -> None:
    np.random.seed(0)
    # 5 folds × 4 models — model A is best (lowest), D is worst
    df = pd.DataFrame({
        "A": np.random.randn(5) * 0.1 + 1.0,
        "B": np.random.randn(5) * 0.1 + 2.0,
        "C": np.random.randn(5) * 0.1 + 3.0,
        "D": np.random.randn(5) * 0.1 + 5.0,
    })
    r = friedman_nemenyi(df)
    assert "friedman_stat" in r
    assert "p_value" in r
    assert "mean_ranks" in r
    assert "nemenyi_cd" in r
    # A should have lowest mean rank
    ranks = r["mean_ranks"]
    assert ranks["A"] < ranks["D"]


def test_friedman_too_few_datasets() -> None:
    df = pd.DataFrame({"A": [1.0]})
    r = friedman_nemenyi(df)
    assert "error" in r
