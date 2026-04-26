"""Optuna tuning helpers.

Strategy: tune-once-on-first-fold (or on full train_val), use best params
for all subsequent folds. Tránh overhead 5x retune.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import optuna

from src.evaluation.metrics import rmse
from src.models.ml import select_feature_columns
from src.utils.logging import get_logger

log = get_logger(__name__)

# Silence optuna verbose
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _rolling_split_for_tune(df: pd.DataFrame, val_size: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take last val_size rows as val, rest as train. Avoid using actual test."""
    n = len(df)
    if n < val_size + 100:
        raise ValueError(f"DF too small ({n}) for val_size={val_size}")
    train = df.iloc[: n - val_size].copy()
    val = df.iloc[n - val_size :].copy()
    return train, val


def tune_xgboost(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "SJC_ban_ra",
    n_trials: int = 30,
    timeout_seconds: int = 120,
) -> dict:
    """Optuna tune XGBoost trên train_df (split internally). Trả về best params."""
    from xgboost import XGBRegressor

    target_h_col = f"y_h{horizon}"
    if target_h_col not in train_df.columns:
        train_df = train_df.copy()
        train_df[target_h_col] = train_df[target_col].shift(-horizon)
    df = train_df.dropna(subset=[target_h_col]).copy()
    train, val = _rolling_split_for_tune(df, val_size=90)

    feature_cols = select_feature_columns(df, target_col=target_col)
    X_tr = train[feature_cols].values
    y_tr = train[target_h_col].values
    X_va = val[feature_cols].ffill().bfill().fillna(0).values
    y_va = val[target_h_col].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1500, step=100),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
            "objective":        "reg:squarederror",
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        return rmse(y_va, y_pred)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
    log.info(f"XGB best RMSE={study.best_value:.4f}, params={study.best_params}")
    return study.best_params


def tune_lightgbm(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "SJC_ban_ra",
    n_trials: int = 30,
    timeout_seconds: int = 120,
) -> dict:
    from lightgbm import LGBMRegressor

    target_h_col = f"y_h{horizon}"
    if target_h_col not in train_df.columns:
        train_df = train_df.copy()
        train_df[target_h_col] = train_df[target_col].shift(-horizon)
    df = train_df.dropna(subset=[target_h_col]).copy()
    train, val = _rolling_split_for_tune(df, val_size=90)

    feature_cols = select_feature_columns(df, target_col=target_col)
    X_tr = train[feature_cols].values
    y_tr = train[target_h_col].values
    X_va = val[feature_cols].ffill().bfill().fillna(0).values
    y_va = val[target_h_col].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1500, step=100),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", -1, 12),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 255),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":     42,
            "n_jobs":           -1,
            "verbose":          -1,
        }
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        return rmse(y_va, y_pred)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
    log.info(f"LGBM best RMSE={study.best_value:.4f}, params={study.best_params}")
    return study.best_params
