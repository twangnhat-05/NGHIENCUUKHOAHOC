"""Regime-aware ensemble: chọn ensemble khác nhau theo volatility regime.

Stable regime → ML linear (Ridge, ElasticNet, AutoARIMA) — predict trend
Volatile regime → RollingNaive + ElasticNet weighted heavy on RollingNaive
                  (bound dưới — assume price gần như random walk khi rally)

Prediction:
1. Detect regime tại train end (latest vol)
2. Use appropriate ensemble for forecast
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.models.classical import (
    NaiveForecaster,
    RollingNaiveForecaster,
    SeasonalNaiveForecaster,
)
from src.models.ml import ElasticNetForecaster, LightGBMForecaster, RidgeForecaster
from src.models.regime import VolatilityRegimeDetector
from src.utils.logging import get_logger

log = get_logger(__name__)


class RegimeAwareEnsemble(BaseForecaster):
    """Switch ensemble dựa trên detected regime.

    Stable: weighted (Ridge 0.4, ElasticNet 0.4, SeasonalNaive 0.2)
    Volatile: weighted (RollingNaive 0.5, ElasticNet 0.3, LightGBM 0.2)
    """
    name = "RegimeAwareEnsemble"
    use_features = True   # ML members cần features

    def __init__(
        self,
        horizon: int = 1,
        regime_window: int = 20,
        regime_threshold_q: float = 0.7,
    ) -> None:
        self.horizon = horizon
        self.regime_window = regime_window
        self.regime_threshold_q = regime_threshold_q

        # Stable regime models
        self.stable_models = {
            "ridge": (RidgeForecaster(horizon=horizon), 0.45),
            "enet":  (ElasticNetForecaster(horizon=horizon), 0.45),
            "snaive": (SeasonalNaiveForecaster(season_length=5), 0.10),
        }
        # Volatile regime models
        self.volatile_models = {
            "rollnaive": (RollingNaiveForecaster(), 0.50),
            "enet":  (ElasticNetForecaster(horizon=horizon), 0.30),
            "lgbm":  (LightGBMForecaster(horizon=horizon, n_estimators=200), 0.20),
        }

        self.detector = VolatilityRegimeDetector(window=regime_window, threshold_quantile=regime_threshold_q)
        self._train_target: pd.Series | None = None
        self._fit_regime: str | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "RegimeAwareEnsemble":
        # Fit regime detector
        self._train_target = train_df[target_col].copy()
        self.detector.fit(self._train_target)

        # Detect regime tại train end
        self._fit_regime = self.detector.detect_single(self._train_target)
        log.info(f"Train end regime detected: {self._fit_regime}")

        # Fit cả 2 ensembles (stable + volatile) vì regime có thể switch khi predict
        all_models = list(self.stable_models.values()) + list(self.volatile_models.values())
        for model, _w in all_models:
            try:
                if hasattr(model, "horizon"):
                    model.horizon = self.horizon
                model.fit(train_df, target_col=target_col)
            except Exception as e:
                log.warning(f"Member {model.name} fit failed: {e}")
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1, target_col: str = "SJC_ban_ra") -> np.ndarray:
        """Rolling regime detection: re-detect mỗi val row dùng past target.

        For each val row i:
          history = train_target ∪ val[target][:i]  (only past, no leakage)
          regime_i = detector.detect_single(history)
          Use stable/volatile ensemble accordingly to predict val row i.

        Optimize: predict ALL val rows với cả 2 ensembles 1 lần,
        sau đó chọn output tương ứng theo regime per row.
        """
        n = len(test_df)
        test_with_obs = test_df.copy()
        if target_col in test_with_obs.columns and "y_observed" not in test_with_obs.columns:
            test_with_obs["y_observed"] = test_with_obs[target_col]

        # Predict với CẢ 2 ensemble, mỗi ensemble cho ra 1 array length n
        def _ensemble_predict(ensemble_dict: dict) -> np.ndarray:
            preds_matrix = []
            weights = []
            for name, (model, w) in ensemble_dict.items():
                try:
                    p = np.asarray(model.predict(test_with_obs, h=h))
                    if len(p) < n:
                        p = np.concatenate([p, np.full(n - len(p), p[-1] if len(p) else 0.0)])
                    preds_matrix.append(p[:n])
                    weights.append(w)
                except Exception as e:
                    log.warning(f"Member {name} predict failed: {e}")
            if not preds_matrix:
                return np.zeros(n)
            preds_matrix = np.array(preds_matrix)
            weights = np.array(weights) / np.array(weights).sum()
            return weights @ preds_matrix

        stable_preds = _ensemble_predict(self.stable_models)
        volatile_preds = _ensemble_predict(self.volatile_models)

        # Rolling regime: cho mỗi val row i, dùng (train + val[:i]) để detect
        if target_col not in test_with_obs.columns:
            # Không có observed target → fallback train-end regime
            return stable_preds if self._fit_regime == "stable" else volatile_preds

        val_target = test_with_obs[target_col].to_numpy()
        # Build extended target: train + val (cumulative past at each val row)
        extended = pd.concat([self._train_target, pd.Series(val_target)], ignore_index=True)
        n_train = len(self._train_target)

        regimes = []
        out = np.zeros(n)
        for i in range(n):
            # History up through val row i (inclusive; predict at row i uses past only)
            history_end = n_train + i
            history = extended.iloc[: history_end + 1]
            try:
                regime_i = self.detector.detect_single(history)
            except Exception:
                regime_i = self._fit_regime
            regimes.append(regime_i)
            out[i] = stable_preds[i] if regime_i == "stable" else volatile_preds[i]

        # Cache regime trace for debugging
        self._last_regime_trace = regimes
        n_volatile = sum(1 for r in regimes if r == "volatile")
        if n_volatile > 0:
            log.info(f"Rolling regimes: {n_volatile}/{n} volatile, {n - n_volatile} stable")
        return out
