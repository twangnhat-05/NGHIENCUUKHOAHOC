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

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        # Use train-end regime cho all val (assumption: regime stable trong val period 90 days)
        # Nâng cao hơn: re-detect mỗi val row, nhưng cần past data — phức tạp
        ensemble = self.stable_models if self._fit_regime == "stable" else self.volatile_models
        log.debug(f"Using {self._fit_regime} ensemble: {list(ensemble.keys())}")

        n = len(test_df)
        # Cần truyền y_observed cho RollingNaive
        test_with_obs = test_df.copy()
        if "SJC_ban_ra" in test_with_obs.columns and "y_observed" not in test_with_obs.columns:
            test_with_obs["y_observed"] = test_with_obs["SJC_ban_ra"]

        preds_matrix = []
        weights = []
        for name, (model, w) in ensemble.items():
            try:
                p = model.predict(test_with_obs, h=h)
                p = np.asarray(p)
                if len(p) < n:
                    p = np.concatenate([p, np.full(n - len(p), p[-1] if len(p) else 0.0)])
                preds_matrix.append(p[:n])
                weights.append(w)
            except Exception as e:
                log.warning(f"Member {name} predict failed: {e}")

        if not preds_matrix:
            return np.zeros(n)
        preds_matrix = np.array(preds_matrix)
        weights = np.array(weights)
        weights = weights / weights.sum()
        return weights @ preds_matrix
