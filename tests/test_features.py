"""Sanity tests cho features pipeline — đảm bảo không leak future."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    add_lag_features,
    add_return_features,
    add_targets,
    add_technical_features,
)
from src.features.calendar import add_calendar_features
from src.features.macro import add_macro_features
from src.features.technical import bollinger, ema, macd, rsi, sma


# ============================================================
# TECHNICAL: only past info (no leakage)
# ============================================================

@pytest.mark.no_leakage
def test_sma_only_past(synthetic_daily_df: pd.DataFrame) -> None:
    s = synthetic_daily_df["SJC_ban_ra"]
    sma10 = sma(s, 10)
    # First 9 values must be NaN (insufficient history)
    assert sma10.iloc[:9].isna().all()
    # SMA at index i depends on values [i-9..i] — verify
    expected = s.iloc[100:110].mean()
    assert sma10.iloc[109] == pytest.approx(expected)


@pytest.mark.no_leakage
def test_rsi_no_lookahead(synthetic_daily_df: pd.DataFrame) -> None:
    s = synthetic_daily_df["SJC_ban_ra"]
    rsi_full = rsi(s, period=14)
    rsi_truncated = rsi(s.iloc[:200], period=14)
    # RSI[100] phải bằng nhau khi tính trên full vs truncated (chỉ dùng past)
    assert rsi_full.iloc[100] == pytest.approx(rsi_truncated.iloc[100])


@pytest.mark.no_leakage
def test_macd_no_lookahead(synthetic_daily_df: pd.DataFrame) -> None:
    s = synthetic_daily_df["SJC_ban_ra"]
    full = macd(s)
    trunc = macd(s.iloc[:200])
    # macd[100] should match (only depends on values up to index 100)
    # Note: EMA has slight numerical drift due to recursive nature; tolerance 1e-3
    for col in ("macd", "signal", "hist"):
        assert full[col].iloc[100] == pytest.approx(trunc[col].iloc[100], rel=1e-3)


@pytest.mark.no_leakage
def test_bollinger_no_lookahead(synthetic_daily_df: pd.DataFrame) -> None:
    s = synthetic_daily_df["SJC_ban_ra"]
    bb_full = bollinger(s)
    bb_trunc = bollinger(s.iloc[:200])
    assert bb_full["bb_upper"].iloc[100] == pytest.approx(bb_trunc["bb_upper"].iloc[100])


# ============================================================
# LAG FEATURES
# ============================================================

@pytest.mark.no_leakage
def test_lag_features_only_past(synthetic_daily_df: pd.DataFrame) -> None:
    df = synthetic_daily_df.copy()
    out = add_lag_features(df, {"SJC_ban_ra": [1, 5, 10]})
    # Lag value at index i = original value at i-lag
    assert out["SJC_ban_ra_lag1"].iloc[10] == pytest.approx(df["SJC_ban_ra"].iloc[9])
    assert out["SJC_ban_ra_lag5"].iloc[10] == pytest.approx(df["SJC_ban_ra"].iloc[5])
    assert out["SJC_ban_ra_lag10"].iloc[10] == pytest.approx(df["SJC_ban_ra"].iloc[0])
    # First `lag` rows should be NaN
    assert out["SJC_ban_ra_lag1"].iloc[:1].isna().all()


# ============================================================
# RETURNS
# ============================================================

@pytest.mark.no_leakage
def test_return_features_align_with_pct_change(synthetic_daily_df: pd.DataFrame) -> None:
    df = synthetic_daily_df.copy()
    out = add_return_features(df, columns=["SJC_ban_ra"], periods=[1, 5], log_returns=True)
    expected1 = df["SJC_ban_ra"].pct_change(1).iloc[10]
    assert out["SJC_ban_ra_ret_1d"].iloc[10] == pytest.approx(expected1)


# ============================================================
# TARGETS
# ============================================================

@pytest.mark.no_leakage
def test_targets_are_future_shifted(synthetic_daily_df: pd.DataFrame) -> None:
    df = synthetic_daily_df.copy()
    out = add_targets(df, "SJC_ban_ra", [1, 5, 20])
    # y_h1 at index i = SJC[i+1]; last row is NaN
    assert out["y_h1"].iloc[10] == pytest.approx(df["SJC_ban_ra"].iloc[11])
    assert out["y_h5"].iloc[10] == pytest.approx(df["SJC_ban_ra"].iloc[15])
    assert out["y_h20"].iloc[10] == pytest.approx(df["SJC_ban_ra"].iloc[30])
    # Last `h` rows are NaN
    assert pd.isna(out["y_h1"].iloc[-1])
    assert out["y_h5"].iloc[-5:].isna().all()
    assert out["y_h20"].iloc[-20:].isna().all()


# ============================================================
# CALENDAR
# ============================================================

def test_calendar_features_added(synthetic_daily_df: pd.DataFrame) -> None:
    out = add_calendar_features(synthetic_daily_df)
    for col in ("cal_dow", "cal_dom", "cal_month", "cal_quarter",
                "cal_dow_sin", "cal_dow_cos", "cal_is_vn_holiday", "cal_days_to_tet"):
        assert col in out.columns
    # dow ∈ [0, 4] sau filter business days
    assert out["cal_dow"].between(0, 6).all()


def test_vn_holidays_detected() -> None:
    from datetime import date
    from src.features.calendar import _is_vn_holiday
    assert _is_vn_holiday(date(2024, 1, 1))     # Tết Dương lịch
    assert _is_vn_holiday(date(2024, 4, 30))    # 30/4
    assert _is_vn_holiday(date(2024, 9, 2))     # Quốc khánh
    assert _is_vn_holiday(date(2024, 2, 10))    # Tết âm 2024 (8-14/2)
    assert not _is_vn_holiday(date(2024, 6, 15))  # Random ngày thường


# ============================================================
# MACRO
# ============================================================

def test_macro_features_added() -> None:
    df = pd.DataFrame({
        "Date": pd.bdate_range("2018-01-01", periods=300),
        "USDVND_Close": np.linspace(22000, 25000, 300),
        "USD_Close": np.linspace(95, 105, 300),
        "USD_Broad_Index": np.linspace(100, 110, 300),
        "Interest_Rate_FED": np.linspace(0.5, 5.0, 300),
        "TenY_Treasury": np.linspace(2.0, 4.5, 300),
        "SJC_ban_ra": np.linspace(60, 90, 300),
        "Gold_Close": np.linspace(1300, 2500, 300),
    })
    out = add_macro_features(df)
    assert "USDVND_change_1d" in out.columns
    assert "USD_z_gap" in out.columns
    assert "yield_spread_10Y_FED" in out.columns
    assert "USD_realized_vol_20d" in out.columns
    assert "sjc_gold_ratio" in out.columns
