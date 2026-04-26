"""Classical baselines: Naive, Seasonal Naive, AutoARIMA, AutoETS, Theta, Prophet, MLForecast.

Tất cả wrap trong interface BaseForecaster để chạy thống nhất qua trainer.
Sử dụng StatsForecast cho ARIMA/ETS/Theta (nhanh, Numba).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================
# TIER 0: TRIVIAL BASELINES
# ============================================================

class NaiveForecaster(BaseForecaster):
    """Mode-A Naive: ŷ_{T+1}, ŷ_{T+2}, ... = y_T (last train value, constant).

    Đây là baseline chuẩn academic: model fit trên train, trả về n_val predictions
    đều bằng giá trị cuối train. So sánh với val[target] cho 1-step-ahead error.
    """
    name = "Naive"

    def __init__(self) -> None:
        self.last_value: float | None = None
        self.train_target: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str, **kwargs) -> "NaiveForecaster":
        self.train_target = train_df[target_col].copy().reset_index(drop=True)
        self.last_value = float(self.train_target.iloc[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        # Mode A: forecast n steps from train-end, mọi step = last train value
        return np.full(len(test_df), self.last_value)


class SeasonalNaiveForecaster(BaseForecaster):
    """Mode-A Seasonal Naive: cycle last `season_length` train values.

    y_{T+i} = y_{T - season_length + (i mod season_length)}
    """
    name = "SeasonalNaive"

    def __init__(self, season_length: int = 5) -> None:
        self.season_length = season_length
        self.train_target: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str, **kwargs) -> "SeasonalNaiveForecaster":
        self.train_target = train_df[target_col].copy().reset_index(drop=True)
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self.train_target is None or len(self.train_target) < self.season_length:
            return np.full(len(test_df), float(self.train_target.iloc[-1]))
        season = self.train_target.iloc[-self.season_length:].to_numpy()
        n = len(test_df)
        idx = (np.arange(n) % self.season_length)
        return season[idx]


class RollingNaiveForecaster(BaseForecaster):
    """Mode-B Rolling Naive: ŷ_{t+1} = y_t (always shift-1 of OBSERVED val).

    Khác `NaiveForecaster`: ở đây ta giả định mỗi val row bộc lộ giá trị thật
    của ngày đó (mode B / rolling-1-step). Predict cho row i = y[i-1] (val past).
    Đây là baseline nhiều paper financial dùng để so với "1-step recursive forecast".
    """
    name = "RollingNaive"

    def __init__(self) -> None:
        self.last_train_value: float | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str, **kwargs) -> "RollingNaiveForecaster":
        self.last_train_value = float(train_df[target_col].iloc[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        # Cần "y_observed" trong test_df = val[target_col]
        if "y_observed" not in test_df.columns:
            return np.full(len(test_df), self.last_train_value or 0.0)
        s = test_df["y_observed"].shift(h)
        s.iloc[:h] = self.last_train_value  # boundary
        return s.to_numpy()


# ============================================================
# TIER 1: STATISTICAL — wrap StatsForecast
# ============================================================

class StatsForecastWrapper(BaseForecaster):
    """Wrap StatsForecast models (AutoARIMA, AutoETS, Theta, etc.)

    StatsForecast yêu cầu format (unique_id, ds, y) — convert internally.
    """

    def __init__(self, model_name: str, model_obj: Any, freq: str = "B") -> None:
        self.name = model_name
        self.model_obj = model_obj
        self.freq = freq
        self._sf = None
        self._train_y: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str, **kwargs) -> "StatsForecastWrapper":
        from statsforecast import StatsForecast
        df_sf = pd.DataFrame({
            "unique_id": "sjc",
            "ds": pd.to_datetime(train_df["Date"]).reset_index(drop=True),
            "y": train_df[target_col].reset_index(drop=True),
        })
        self._sf = StatsForecast(models=[self.model_obj], freq=self.freq, n_jobs=1)
        self._sf.fit(df_sf)
        self._train_y = train_df[target_col].to_numpy()
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self._sf is None:
            raise RuntimeError("Model chưa fit")
        n = len(test_df)
        # forecast n steps cho test horizon
        try:
            fcst = self._sf.predict(h=n)
        except Exception as e:
            log.warning(f"{self.name}.predict failed: {e}; fallback to naive")
            return np.full(n, float(self._train_y[-1]))
        col = self.model_obj.alias if hasattr(self.model_obj, "alias") else self.name
        # Find the model column (first non-id/ds column)
        pred_cols = [c for c in fcst.columns if c not in ("unique_id", "ds")]
        if not pred_cols:
            raise RuntimeError(f"StatsForecast fcst không có cột prediction: {list(fcst.columns)}")
        return fcst[pred_cols[0]].to_numpy()[:n]


def _make_statsforecast_models(freq: str = "B") -> list[BaseForecaster]:
    """Build danh sách classical models từ StatsForecast."""
    from statsforecast.models import (
        AutoARIMA,
        AutoETS,
        AutoTheta,
        HistoricAverage,
    )
    models: list[BaseForecaster] = []
    # AutoARIMA — season=5 (weekly), max_p/q nhỏ vì data daily ~2k
    try:
        models.append(StatsForecastWrapper("AutoARIMA",
                                            AutoARIMA(season_length=5), freq=freq))
    except Exception as e:
        log.warning(f"AutoARIMA init failed: {e}")
    try:
        models.append(StatsForecastWrapper("AutoETS",
                                            AutoETS(season_length=5), freq=freq))
    except Exception as e:
        log.warning(f"AutoETS init failed: {e}")
    try:
        models.append(StatsForecastWrapper("AutoTheta",
                                            AutoTheta(season_length=5), freq=freq))
    except Exception as e:
        log.warning(f"AutoTheta init failed: {e}")
    try:
        models.append(StatsForecastWrapper("HistoricAverage",
                                            HistoricAverage(), freq=freq))
    except Exception as e:
        log.warning(f"HistoricAverage init failed: {e}")
    return models


# ============================================================
# TIER 1b: PROPHET (Meta)
# ============================================================

class ProphetForecaster(BaseForecaster):
    """Prophet wrapper. Univariate target only (W2). Có thể thêm exog regressors sau."""
    name = "Prophet"

    def __init__(self, daily_seasonality: bool = False, weekly_seasonality: bool = True) -> None:
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self._model = None
        self._last_train_date: pd.Timestamp | None = None
        self._last_train_value: float | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str, **kwargs) -> "ProphetForecaster":
        from prophet import Prophet
        df = pd.DataFrame({
            "ds": pd.to_datetime(train_df["Date"]).reset_index(drop=True),
            "y": train_df[target_col].reset_index(drop=True),
        })
        self._model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=True,
        )
        # Prophet log lai nhiều — silence
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self._model.fit(df)
        self._last_train_date = df["ds"].iloc[-1]
        self._last_train_value = float(df["y"].iloc[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model chưa fit")
        future = pd.DataFrame({"ds": pd.to_datetime(test_df["Date"]).reset_index(drop=True)})
        try:
            fcst = self._model.predict(future)
            return fcst["yhat"].to_numpy()
        except Exception as e:
            log.warning(f"Prophet.predict failed: {e}; fallback to naive")
            return np.full(len(test_df), self._last_train_value)


# ============================================================
# TIER 2 (preview): MLForecast LightGBM (full-feature, nhanh)
# ============================================================

class MLForecastLGBM(BaseForecaster):
    """MLForecast với LightGBM regressor — auto lag features."""
    name = "MLForecast_LGBM"

    def __init__(self, lags: list[int] | None = None, lag_means: list[int] | None = None) -> None:
        self.lags = lags or [1, 2, 3, 5, 7, 14, 21, 30]
        self.lag_means = lag_means or [7, 14, 30]
        self._mf = None
        self._last_train_value: float | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str, **kwargs) -> "MLForecastLGBM":
        from mlforecast import MLForecast
        from lightgbm import LGBMRegressor
        from mlforecast.lag_transforms import RollingMean

        # MLForecast cần freq nhất quán; data của ta có gaps (lễ VN, holidays).
        # Workaround: dùng integer index sequential để tránh "missing dates" check.
        n = len(train_df)
        df = pd.DataFrame({
            "unique_id": "sjc",
            "ds": np.arange(n, dtype=np.int64),  # integer index — bypass freq check
            "y": train_df[target_col].reset_index(drop=True).to_numpy(),
        })
        models = {"lgbm": LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31,
            random_state=42, verbose=-1, n_jobs=1,
        )}
        self._mf = MLForecast(
            models=models,
            freq=1,                  # integer freq for integer index
            lags=self.lags,
            lag_transforms={1: [RollingMean(window_size=w) for w in self.lag_means]},
        )
        self._mf.fit(df, id_col="unique_id", time_col="ds", target_col="y")
        self._last_train_value = float(df["y"].iloc[-1])
        self._train_n = n
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self._mf is None:
            raise RuntimeError("Chưa fit")
        n = len(test_df)
        try:
            fcst = self._mf.predict(h=n)
            return fcst["lgbm"].to_numpy()[:n]
        except Exception as e:
            log.warning(f"MLForecast_LGBM predict failed: {e}; naive")
            return np.full(n, self._last_train_value)


# ============================================================
# REGISTRY
# ============================================================

def build_classical_models(freq: str = "B", include_prophet: bool = True) -> list[BaseForecaster]:
    """Trả về danh sách model classical sẵn-sàng-chạy.

    Bao gồm cả `RollingNaive` (mode-B) để so sánh với `Naive` (mode-A): cho thấy
    hiệu quả "biết y[t]" lớn cỡ nào — gợi ý floor cho mọi mô hình mode-A.
    """
    models: list[BaseForecaster] = []
    models.append(NaiveForecaster())
    models.append(SeasonalNaiveForecaster(season_length=5))
    models.append(RollingNaiveForecaster())  # mode-B floor
    models.extend(_make_statsforecast_models(freq=freq))
    if include_prophet:
        models.append(ProphetForecaster())
    models.append(MLForecastLGBM())
    return models
