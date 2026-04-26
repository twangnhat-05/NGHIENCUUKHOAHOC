"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_daily_df() -> pd.DataFrame:
    """Synthetic time-series daily DataFrame ~ 1500 rows (~6 năm) with trend + seasonality + noise."""
    rng = np.random.default_rng(42)
    n = 1500
    dates = pd.bdate_range(start="2018-01-02", periods=n, freq="B")
    t = np.arange(n)
    trend = 60 + 0.02 * t
    season = 5 * np.sin(2 * np.pi * t / 252) + 2 * np.cos(2 * np.pi * t / 30)
    noise = rng.normal(scale=1.0, size=n)
    sjc_ban_ra = trend + season + noise
    return pd.DataFrame({
        "Date": dates,
        "SJC_ban_ra": sjc_ban_ra,
        "Gold_Close": 1300 + 5 * t / n + rng.normal(scale=20, size=n),
        "USD_Index_Close": 95 + 0.5 * np.sin(2 * np.pi * t / 252) + rng.normal(scale=0.5, size=n),
    })


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Tiny DF cho edge cases (200 rows)."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=n),
        "value": rng.normal(loc=100, scale=2, size=n).cumsum(),
    })
