"""Walk-forward trainer: fit models trên train fold, predict val fold, collect metrics.

Mỗi (model × horizon × fold) là 1 record. Output → list[dict] feed cho leaderboard.
"""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import summary
from src.models.base import BaseForecaster
from src.training.cv import WalkForwardCV
from src.utils.logging import get_logger

log = get_logger(__name__)


def evaluate_one_fold(
    model: BaseForecaster,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    target_col: str = "SJC_ban_ra",
) -> dict:
    """Fit model trên train, predict val (mode A — single fit, multi-step forecast).

    Evaluation:
      - y_pred[i] = forecast cho thời điểm val.iloc[i].Date (i+1 steps ahead from train-end)
      - y_true[i] = val[target_col].iloc[i]
      - Horizon parameter chỉ ảnh hưởng SHIFT của y_pred / y_true:
          * h=1: predict val[0:n], compare val[target][0:n]   (n_eval = n)
          * h=5: predict val[4:n+4], compare val[target][0:n]  → bỏ 4 row đầu
          * Practical: align bằng cách forecast n+h-1 steps, slice [h-1:h-1+n]

    Để giữ đơn giản W2: chạy h=1 baseline (forecast horizon = step ahead = 1 day each row).
    Để compute multi-horizon đúng (h=5, h=20), cần forecast (n + h - 1) bước rồi shift.
    Code dưới handle generic h theo logic align:
    """
    train = train_df.copy()
    val = val_df.copy()
    n_val = len(val)
    if n_val < horizon + 5:
        log.warning(f"{model.name} fold val ({n_val}) too small for h={horizon}")
        return {}

    # y_observed cho mode-B Naive
    val["y_observed"] = val[target_col]

    # Fit
    t0 = time.time()
    try:
        model.fit(train, target_col=target_col)
    except Exception as e:
        log.error(f"{model.name} fit failed: {e}")
        return {}
    fit_time = time.time() - t0

    # Predict n_val + (horizon - 1) bước → align: pred_at_index_i predicts time T + i + 1
    # Ta cần h-step-ahead cho mỗi origin t_i ∈ {T, T+1, ..., T+n_val-h}
    # Origin T+i predicts T+i+h → pred index = i+h-1 (0-based: forecast(n) returns indices 0..n-1)
    n_forecast = n_val   # mode A: chỉ fit 1 lần, forecast n_val steps, sau đó shift cho h
    t0 = time.time()
    try:
        y_pred_raw = model.predict(val, h=horizon)
    except Exception as e:
        log.error(f"{model.name} predict failed: {e}")
        return {}
    pred_time = time.time() - t0
    y_pred_raw = np.asarray(y_pred_raw).flatten()

    # Align: for h=1, y_pred[i] vs val[target][i]
    #        for h>1, y_pred[i] vs val[target][i+h-1]  (drop first h-1 rows from y_true)
    if len(y_pred_raw) < n_val:
        # Pad with last value if model returned fewer
        y_pred_raw = np.concatenate([y_pred_raw, np.full(n_val - len(y_pred_raw), y_pred_raw[-1])])
    if horizon == 1:
        y_pred = y_pred_raw[:n_val]
        y_true = val[target_col].to_numpy()[:n_val]
        y_prev = np.concatenate([[train[target_col].iloc[-1]],
                                  val[target_col].to_numpy()[:n_val - 1]])
    else:
        # Slice: predict[h-1:h-1+n_eval], target[0:n_eval], n_eval = n_val - (h-1)
        n_eval = n_val - (horizon - 1)
        y_pred = y_pred_raw[horizon - 1: horizon - 1 + n_eval]
        y_true = val[target_col].to_numpy()[horizon - 1: horizon - 1 + n_eval]
        # y_prev cho horizon h: giá trị h ngày trước y_true
        prev_idx_start = max(0, 0)
        y_prev = val[target_col].to_numpy()[:n_eval]

    y_train = train[target_col].to_numpy()
    metrics = summary(y_true, y_pred, y_train=y_train, y_prev=y_prev, seasonal_period=5)
    metrics["fit_seconds"] = fit_time
    metrics["pred_seconds"] = pred_time
    metrics["n_eval"] = len(y_true)
    return metrics


def run_walk_forward(
    df: pd.DataFrame,
    models: Iterable[BaseForecaster],
    cv: WalkForwardCV,
    horizons: Iterable[int],
    target_col: str = "SJC_ban_ra",
) -> list[dict]:
    """Run mọi (model × horizon × fold). Trả về list of records."""
    results = []
    folds = list(cv.split(df))
    log.info(f"{len(folds)} folds × {len(list(horizons))} horizons × {len(list(models))} models")
    horizons = list(horizons)
    models = list(models)

    for fold in folds:
        train_df, val_df = cv.get_train_val(df, fold)
        log.info(f"=== Fold {fold.fold_id}: {fold.train_dates[0].date()}→{fold.train_dates[1].date()} "
                 f"| val {fold.val_dates[0].date()}→{fold.val_dates[1].date()} ===")
        for h in horizons:
            for model in models:
                t0 = time.time()
                metrics = evaluate_one_fold(model, train_df, val_df, h, target_col)
                if not metrics:
                    continue
                results.append({
                    "model": model.name,
                    "horizon": h,
                    "fold_id": fold.fold_id,
                    "metrics": metrics,
                })
                log.info(f"  {model.name:18s} h={h:2d} → MAPE={metrics.get('MAPE', float('nan')):.2f}% "
                         f"RMSE={metrics.get('RMSE', float('nan')):.4f} "
                         f"DA={metrics.get('DA', float('nan')):5.1f}%  "
                         f"({time.time()-t0:.1f}s)")
    return results
