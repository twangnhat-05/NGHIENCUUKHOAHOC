"""Tier-2 Tabular ML wrappers — sử dụng features_v2 (108 features) trực tiếp.

Khác với classical (mode-A: 1 fit train, n-step forecast), ML models dùng
mode-B per-row prediction:
- Mỗi val row có vector feature (lags + technical + macro + calendar)
- Predict y_h{horizon} cho row đó (single point forecast)
- Horizon "baked in" model: 1 model per horizon (tuned with Optuna riêng)

Lý do: ML cần exogenous features để vượt baseline; classical không. Đây là
cách so sánh CÔNG BẰNG — classical có disadvantage không dùng exog, nhưng
ML có overhead retrain per fold.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================
# FEATURE SELECTION HELPER
# ============================================================

def select_feature_columns(
    df: pd.DataFrame,
    target_col: str = "SJC_ban_ra",
    exclude_prefixes: tuple[str, ...] = ("y_h", "Date"),
    exclude_exact: tuple[str, ...] = ("SJC_mua_vao",),
) -> list[str]:
    """Chọn feature columns từ features_v2: bỏ Date, các y_h*, target, SJC_mua_vao."""
    cols = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if c == target_col:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        # Chỉ lấy numeric
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


# ============================================================
# BASE ML WRAPPER (mode-B per-row)
# ============================================================

class _MLBaseWrapper(BaseForecaster):
    """Base cho XGB/LGBM/CatBoost/RF/SVR/Ridge.

    Subclass override `_make_estimator()`.
    """
    name = "ml_base"
    use_features = True   # đánh dấu để trainer dispatch đúng path

    def __init__(self, horizon: int = 1, scale_features: bool = False, **estimator_kwargs) -> None:
        self.horizon = horizon
        self.scale_features = scale_features
        self.estimator_kwargs = estimator_kwargs
        self._estimator = None
        self._feature_cols: list[str] = []
        self._scaler = None
        self._last_train_target: float | None = None

    def _make_estimator(self):
        raise NotImplementedError

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "_MLBaseWrapper":
        target_h_col = f"y_h{self.horizon}"
        if target_h_col not in train_df.columns:
            train_df = train_df.copy()
            train_df[target_h_col] = train_df[target_col].shift(-self.horizon)

        # Drop rows với NaN target (last `horizon` rows)
        valid = train_df.dropna(subset=[target_h_col]).copy()
        self._feature_cols = select_feature_columns(valid, target_col=target_col)
        # Fill NaN trong features (sentiment lags / first rows) — same logic as predict
        X_df = valid[self._feature_cols].ffill().bfill().fillna(0.0)
        X = X_df.values
        y = valid[target_h_col].values

        if self.scale_features:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        self._estimator = self._make_estimator()
        self._estimator.fit(X, y)
        self._last_train_target = float(valid[target_col].iloc[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        """Predict y_h{self.horizon} for each row of test_df.

        Note: argument `h` ignored (horizon baked in via self.horizon).
        Trainer should call evaluate_ml_one_fold which passes correct val + horizon.
        """
        if self._estimator is None:
            raise RuntimeError(f"{self.name} chưa fit")
        # Use ffill cho NaN trong test (vì mỗi val row có thể có NaN từ feature lag)
        X_df = test_df[self._feature_cols].ffill().bfill().fillna(0.0)
        X = X_df.values
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._estimator.predict(X)


# ============================================================
# CONCRETE MODELS
# ============================================================

class XGBoostForecaster(_MLBaseWrapper):
    name = "XGBoost"

    def __init__(self, horizon: int = 1, **kwargs) -> None:
        defaults = dict(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
            objective="reg:squarederror",
        )
        defaults.update(kwargs)
        super().__init__(horizon=horizon, scale_features=False, **defaults)

    def _make_estimator(self):
        from xgboost import XGBRegressor
        return XGBRegressor(**self.estimator_kwargs)


class LightGBMForecaster(_MLBaseWrapper):
    name = "LightGBM"

    def __init__(self, horizon: int = 1, **kwargs) -> None:
        defaults = dict(
            n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        defaults.update(kwargs)
        super().__init__(horizon=horizon, scale_features=False, **defaults)

    def _make_estimator(self):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**self.estimator_kwargs)


class CatBoostForecaster(_MLBaseWrapper):
    name = "CatBoost"

    def __init__(self, horizon: int = 1, **kwargs) -> None:
        defaults = dict(
            iterations=500, learning_rate=0.05, depth=6,
            random_state=42, verbose=False, allow_writing_files=False,
        )
        defaults.update(kwargs)
        super().__init__(horizon=horizon, scale_features=False, **defaults)

    def _make_estimator(self):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**self.estimator_kwargs)


class RandomForestForecaster(_MLBaseWrapper):
    name = "RandomForest"

    def __init__(self, horizon: int = 1, **kwargs) -> None:
        defaults = dict(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        defaults.update(kwargs)
        super().__init__(horizon=horizon, scale_features=False, **defaults)

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**self.estimator_kwargs)


class SVRForecaster(_MLBaseWrapper):
    name = "SVR_RBF"

    def __init__(self, horizon: int = 1, **kwargs) -> None:
        defaults = dict(kernel="rbf", C=10.0, gamma="scale", epsilon=0.05)
        defaults.update(kwargs)
        super().__init__(horizon=horizon, scale_features=True, **defaults)

    def _make_estimator(self):
        from sklearn.svm import SVR
        return SVR(**self.estimator_kwargs)


class RidgeForecaster(_MLBaseWrapper):
    name = "Ridge"

    def __init__(self, horizon: int = 1, **kwargs) -> None:
        defaults = dict(alpha=1.0, random_state=42)
        defaults.update(kwargs)
        super().__init__(horizon=horizon, scale_features=True, **defaults)

    def _make_estimator(self):
        from sklearn.linear_model import Ridge
        return Ridge(**self.estimator_kwargs)


class ElasticNetForecaster(_MLBaseWrapper):
    name = "ElasticNet"

    def __init__(self, horizon: int = 1, **kwargs) -> None:
        defaults = dict(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
        defaults.update(kwargs)
        super().__init__(horizon=horizon, scale_features=True, **defaults)

    def _make_estimator(self):
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**self.estimator_kwargs)


# ============================================================
# STACKING: XGB + LGBM + CatBoost → Ridge meta
# ============================================================

class StackingForecaster(_MLBaseWrapper):
    """Stacking ensemble: XGB + LGBM + CatBoost → Ridge meta-learner.

    Sử dụng `cross_val_predict` với 3 folds để generate OOF predictions,
    rồi fit meta-learner trên đó. Standard scikit-learn StackingRegressor.
    """
    name = "Stacking_XGB_LGBM_Cat"

    def __init__(self, horizon: int = 1) -> None:
        super().__init__(horizon=horizon, scale_features=False)

    def _make_estimator(self):
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        base = [
            ("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  random_state=42, n_jobs=1, verbosity=0)),
            ("lgbm", LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    random_state=42, n_jobs=1, verbose=-1)),
            ("cat", CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                                       random_state=42, verbose=False, allow_writing_files=False)),
        ]
        return StackingRegressor(
            estimators=base,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=1,
        )


# ============================================================
# REGISTRY
# ============================================================

def build_ml_models(horizon: int = 1, include_stacking: bool = False) -> list[BaseForecaster]:
    """Trả về list ML models cho 1 horizon. Để chạy multi-horizon, lặp build."""
    models: list[BaseForecaster] = [
        RidgeForecaster(horizon=horizon),
        ElasticNetForecaster(horizon=horizon),
        SVRForecaster(horizon=horizon),
        RandomForestForecaster(horizon=horizon),
        XGBoostForecaster(horizon=horizon),
        LightGBMForecaster(horizon=horizon),
        CatBoostForecaster(horizon=horizon),
    ]
    if include_stacking:
        models.append(StackingForecaster(horizon=horizon))
    return models
