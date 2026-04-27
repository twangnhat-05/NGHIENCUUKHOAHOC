"""Regime detection cho time series volatile (gold rally 2024 example).

Strategy: rolling 20-day std of log returns → 2 regimes:
- "stable": vol < threshold (low volatility, mean-reverting)
- "volatile": vol >= threshold (rally / crash, momentum-driven)

Threshold fit chỉ trên train fold (no leakage).

References:
- Hamilton (1989) Markov regime-switching
- Đơn giản hóa: threshold rule thay vì HMM (đủ cho gold market 2 regime rõ rệt)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


class VolatilityRegimeDetector:
    """Detect regime dựa trên rolling volatility threshold.

    Parameters
    ----------
    window : int
        Rolling window cho std (20 = ~1 tháng business)
    threshold_quantile : float
        Quantile của vol distribution trên train để định ngưỡng (default 0.7 = 70th percentile)
    """

    def __init__(self, window: int = 20, threshold_quantile: float = 0.7) -> None:
        self.window = window
        self.threshold_quantile = threshold_quantile
        self.threshold_value: float | None = None

    @staticmethod
    def _compute_log_return_vol(series: pd.Series, window: int) -> pd.Series:
        log_ret = np.log(series / series.shift(1))
        return log_ret.rolling(window, min_periods=window).std() * np.sqrt(252)

    def fit(self, train_target: pd.Series) -> "VolatilityRegimeDetector":
        """Fit threshold trên train target series."""
        vol = self._compute_log_return_vol(train_target, self.window).dropna()
        if len(vol) < 50:
            log.warning(f"Train vol có {len(vol)} obs, có thể thiếu cho threshold ổn định")
        self.threshold_value = float(vol.quantile(self.threshold_quantile))
        log.info(f"Regime threshold (vol annualized) = {self.threshold_value:.4f} "
                 f"(at q={self.threshold_quantile} of {len(vol)} train obs)")
        return self

    def detect(self, target_series: pd.Series) -> pd.Series:
        """Detect regime cho mỗi point. Returns series of {'stable', 'volatile'}."""
        if self.threshold_value is None:
            raise RuntimeError("Detector chưa fit")
        vol = self._compute_log_return_vol(target_series, self.window)
        regime = pd.Series(
            np.where(vol >= self.threshold_value, "volatile", "stable"),
            index=target_series.index,
        )
        # First `window` rows: NaN vol → default "stable"
        regime[vol.isna()] = "stable"
        return regime

    def detect_single(self, recent_series: pd.Series) -> Literal["stable", "volatile"]:
        """Detect regime cho điểm cuối cùng dùng recent history."""
        regime_series = self.detect(recent_series)
        return regime_series.iloc[-1]
