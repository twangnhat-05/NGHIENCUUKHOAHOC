"""Macro derivatives: yield spread, real rate, USD/VND change rate, etc."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived macro features. Yêu cầu df có:
    - USD_Close, USD_Broad_Index, USDVND_Close (USD-related)
    - Interest_Rate_FED, TenY_Treasury (rates)
    """
    out = df.copy()

    # USD/VND change (1, 5, 20 day)
    if "USDVND_Close" in out.columns:
        for p in [1, 5, 20]:
            out[f"USDVND_change_{p}d"] = out["USDVND_Close"].pct_change(p)

    # USD index dispersion (yfinance vs FRED official)
    if "USD_Close" in out.columns and "USD_Broad_Index" in out.columns:
        # Normalize and take ratio (z-score gap)
        usd_yf_z = (out["USD_Close"] - out["USD_Close"].rolling(252, min_periods=252).mean()) \
            / out["USD_Close"].rolling(252, min_periods=252).std()
        usd_fred_z = (out["USD_Broad_Index"] - out["USD_Broad_Index"].rolling(252, min_periods=252).mean()) \
            / out["USD_Broad_Index"].rolling(252, min_periods=252).std()
        out["USD_z_gap"] = usd_yf_z - usd_fred_z

    # Yield spread proxy (10Y - FED funds, both monthly forward-filled)
    if "TenY_Treasury" in out.columns and "Interest_Rate_FED" in out.columns:
        out["yield_spread_10Y_FED"] = out["TenY_Treasury"] - out["Interest_Rate_FED"]

    # Realized USD volatility (rolling std of returns)
    if "USD_Close" in out.columns:
        usd_ret = out["USD_Close"].pct_change()
        out["USD_realized_vol_20d"] = usd_ret.rolling(20, min_periods=20).std() * np.sqrt(252)

    # SJC - Gold spread (% premium so với gold thế giới quy đổi)
    # Đơn giản: ratio (không quy đổi USD vì cần USD/VND + tỷ lệ tael→ounce phức tạp)
    if "SJC_ban_ra" in out.columns and "Gold_Close" in out.columns:
        # Chỉ bound check, không tính premium thực
        out["sjc_gold_ratio"] = out["SJC_ban_ra"] / out["Gold_Close"]

    return out
