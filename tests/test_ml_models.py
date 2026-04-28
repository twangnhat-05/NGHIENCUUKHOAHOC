"""Unit tests cho ML wrappers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ml import (
    ElasticNetForecaster,
    LightGBMForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
    select_feature_columns,
)


@pytest.fixture
def small_features_df() -> pd.DataFrame:
    """Synthetic features dataset cho tests."""
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    sjc = 60 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "Date": dates,
        "SJC_ban_ra": sjc,
        "SJC_mua_vao": sjc - 0.2,
        "Gold_Close": 1500 + np.cumsum(rng.normal(0, 5, n)),
        "USD_Close": 95 + rng.normal(0, 0.5, n),
        "feat_1": rng.normal(0, 1, n),
        "feat_2": rng.normal(0, 1, n),
    })
    # Add target columns
    df["y_h1"] = df["SJC_ban_ra"].shift(-1)
    df["y_h5"] = df["SJC_ban_ra"].shift(-5)
    df["y_h20"] = df["SJC_ban_ra"].shift(-20)
    return df


def test_select_features_excludes_targets_and_date(small_features_df: pd.DataFrame) -> None:
    cols = select_feature_columns(small_features_df, target_col="SJC_ban_ra")
    assert "Date" not in cols
    assert "y_h1" not in cols
    assert "y_h5" not in cols
    assert "y_h20" not in cols
    assert "SJC_ban_ra" not in cols
    assert "SJC_mua_vao" not in cols  # excluded by default
    assert "Gold_Close" in cols


def test_ridge_fit_predict(small_features_df: pd.DataFrame) -> None:
    train = small_features_df.iloc[:200]
    val = small_features_df.iloc[200:].dropna(subset=["y_h1"])
    model = RidgeForecaster(horizon=1)
    model.fit(train, target_col="SJC_ban_ra")
    preds = model.predict(val)
    assert len(preds) == len(val)
    assert not np.isnan(preds).any()


def test_xgboost_fit_predict(small_features_df: pd.DataFrame) -> None:
    train = small_features_df.iloc[:200]
    val = small_features_df.iloc[200:].dropna(subset=["y_h1"])
    model = XGBoostForecaster(horizon=1, n_estimators=50)
    model.fit(train, target_col="SJC_ban_ra")
    preds = model.predict(val)
    assert len(preds) == len(val)


def test_lightgbm_horizon_5(small_features_df: pd.DataFrame) -> None:
    train = small_features_df.iloc[:200]
    val = small_features_df.iloc[200:].dropna(subset=["y_h5"])
    model = LightGBMForecaster(horizon=5, n_estimators=50)
    model.fit(train, target_col="SJC_ban_ra")
    preds = model.predict(val)
    assert len(preds) == len(val)


def test_random_forest_use_features_attr() -> None:
    model = RandomForestForecaster(horizon=1)
    assert model.use_features is True


@pytest.mark.no_leakage
def test_ml_scaler_fit_only_on_train(small_features_df: pd.DataFrame) -> None:
    """Verify Ridge scaler is fit only on train (not on val)."""
    train = small_features_df.iloc[:200]
    model = RidgeForecaster(horizon=1)
    model.fit(train, target_col="SJC_ban_ra")
    # The scaler.mean_ should match train statistics
    assert model._scaler is not None
    feature_cols = model._feature_cols
    train_features = train[feature_cols].ffill().fillna(0).values
    expected_mean = train_features.mean(axis=0)
    np.testing.assert_allclose(model._scaler.mean_, expected_mean, rtol=1e-6)


@pytest.mark.no_leakage
def test_elastic_net_no_target_in_features(small_features_df: pd.DataFrame) -> None:
    model = ElasticNetForecaster(horizon=1)
    model.fit(small_features_df, target_col="SJC_ban_ra")
    # Verify y_h1, y_h5, y_h20, SJC_ban_ra not in feature_cols → no target leakage
    for forbidden in ("y_h1", "y_h5", "y_h20", "SJC_ban_ra"):
        assert forbidden not in model._feature_cols, f"{forbidden} leaked into features"
