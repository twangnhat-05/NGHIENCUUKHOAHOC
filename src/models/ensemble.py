"""Ensemble forecasters: simple average, inverse-RMSE weighted, stacking.

Workflow:
1. Bộ chứa list của BaseForecaster (đã constructed)
2. fit(): refit từng model trên train
3. predict(): lấy predictions từ mỗi model, kết hợp theo strategy
4. Optional: cross-validate trên train để tính weights inverse-RMSE
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import rmse
from src.models.base import BaseForecaster
from src.utils.logging import get_logger

log = get_logger(__name__)


class EnsembleForecaster(BaseForecaster):
    """Combine predictions từ multiple base models.

    strategy: "mean" | "median" | "inverse_rmse" | "weighted" (custom weights)
    """
    name = "Ensemble"
    use_features = False  # actual dispatch depends on member models — handled in fit/predict

    def __init__(
        self,
        base_models: list[BaseForecaster],
        strategy: str = "inverse_rmse",
        weights: list[float] | None = None,
        cv_split_size: int = 90,
        horizon: int = 1,
    ) -> None:
        if not base_models:
            raise ValueError("Cần ≥ 1 base model")
        self.base_models = base_models
        self.strategy = strategy
        self.weights = weights
        self.cv_split_size = cv_split_size
        self.horizon = horizon
        self._fitted_weights: np.ndarray | None = None
        self.name = f"Ensemble[{strategy}]_{len(base_models)}models"

    def _detect_use_features(self, model: BaseForecaster) -> bool:
        return getattr(model, "use_features", False)

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "EnsembleForecaster":
        # Sync horizon trên các base model nếu họ có attribute
        for m in self.base_models:
            if hasattr(m, "horizon"):
                m.horizon = self.horizon

        if self.strategy == "inverse_rmse":
            # Internal split để tính weights
            n = len(train_df)
            cv_split = min(self.cv_split_size, n // 4)
            internal_train = train_df.iloc[:n - cv_split].copy()
            internal_val = train_df.iloc[n - cv_split:].copy()
            target_h_col = f"y_h{self.horizon}"
            if target_h_col not in internal_val.columns:
                internal_val[target_h_col] = internal_val[target_col].shift(-self.horizon)

            rmses = []
            for m in self.base_models:
                try:
                    m.fit(internal_train, target_col=target_col)
                    if self._detect_use_features(m):
                        eval_df = internal_val.dropna(subset=[target_h_col])
                        y_pred = m.predict(eval_df, h=self.horizon)
                        y_true = eval_df[target_h_col].to_numpy()
                    else:
                        eval_df = internal_val.copy()
                        eval_df["y_observed"] = eval_df[target_col]
                        y_pred = m.predict(eval_df, h=self.horizon)
                        y_true = eval_df[target_col].to_numpy()
                    if len(y_pred) != len(y_true):
                        n_min = min(len(y_pred), len(y_true))
                        y_pred = y_pred[:n_min]
                        y_true = y_true[:n_min]
                    err = rmse(y_true, y_pred)
                    rmses.append(max(err, 1e-6))
                except Exception as e:
                    log.warning(f"Ensemble member {m.name} internal-CV failed: {e}")
                    rmses.append(np.inf)
            inv = 1.0 / np.array(rmses)
            inv = np.where(np.isinf(inv) | np.isnan(inv), 0.0, inv)
            if inv.sum() > 0:
                self._fitted_weights = inv / inv.sum()
            else:
                self._fitted_weights = np.ones(len(self.base_models)) / len(self.base_models)
            log.info(f"Ensemble inverse_rmse weights: "
                     + ", ".join(f"{m.name}={w:.3f}" for m, w in zip(self.base_models, self._fitted_weights)))
        elif self.strategy == "weighted":
            if self.weights is None or len(self.weights) != len(self.base_models):
                raise ValueError("strategy=weighted cần `weights` cùng số lượng base_models")
            w = np.array(self.weights, dtype=float)
            self._fitted_weights = w / w.sum()
        else:
            self._fitted_weights = np.ones(len(self.base_models)) / len(self.base_models)

        # Refit all models trên FULL train
        for m in self.base_models:
            try:
                m.fit(train_df, target_col=target_col)
            except Exception as e:
                log.warning(f"Ensemble final refit {m.name} failed: {e}")
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        n = len(test_df)
        preds_matrix = np.zeros((len(self.base_models), n))
        for i, m in enumerate(self.base_models):
            try:
                p = m.predict(test_df, h=h)
                p = np.asarray(p)
                if len(p) < n:
                    p = np.concatenate([p, np.full(n - len(p), p[-1] if len(p) else 0.0)])
                preds_matrix[i] = p[:n]
            except Exception as e:
                log.warning(f"Ensemble predict {m.name} failed: {e}")
                preds_matrix[i] = 0.0

        if self.strategy == "median":
            return np.median(preds_matrix, axis=0)
        # Weighted average
        weights = self._fitted_weights if self._fitted_weights is not None else \
                  np.ones(len(self.base_models)) / len(self.base_models)
        return weights @ preds_matrix


def build_ensemble(
    base_models: list[BaseForecaster],
    strategy: str = "inverse_rmse",
    horizon: int = 1,
) -> EnsembleForecaster:
    return EnsembleForecaster(base_models=base_models, strategy=strategy, horizon=horizon)
