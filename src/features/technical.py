"""Technical indicators thuần — không có lookahead.

Mọi indicator chỉ dùng giá quá khứ. Index là position-based; signature mọi hàm:
    f(series: pd.Series, ...) -> pd.Series  (cùng index)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def returns(series: pd.Series, period: int = 1, log: bool = False) -> pd.Series:
    """Simple hoặc log return chu kỳ `period`."""
    if log:
        return np.log(series / series.shift(period))
    return series.pct_change(periods=period)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing approximation = SMA)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(100.0)  # avg_loss=0 → max RSI


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD: trả về DataFrame ['macd', 'signal', 'hist']."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: trả về DataFrame ['bb_mid', 'bb_upper', 'bb_lower', 'bb_width', 'bb_pct']."""
    mid = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid
    pct = (series - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_width": width, "bb_pct": pct,
    })


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range — cần OHLC. Wilder smoothing (EMA-style)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator: trả về ['stoch_k', 'stoch_d']."""
    low_min = low.rolling(k_period, min_periods=k_period).min()
    high_max = high.rolling(k_period, min_periods=k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def realized_volatility(returns_series: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility từ returns (rolling std × sqrt(252))."""
    return returns_series.rolling(window, min_periods=window).std() * np.sqrt(252)


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Momentum = price_t - price_{t-period}."""
    return series - series.shift(period)


def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    """ROC = (price_t / price_{t-period}) - 1."""
    return series.pct_change(periods=period)
