"""Benchmark fine-tuned Chronos-Bolt vs zero-shot vs Ridge per fold.

Reads `models/chronos_finetuned/fold_{k}/` checkpoints. Skips folds without
a saved checkpoint. Saves `reports/leaderboard/chronos_finetuned_long.csv` and
`chronos_finetuned_summary.csv`.

Usage: python -m scripts.benchmark_finetuned_chronos --folds 0 --horizons 1 5 20
       python -m scripts.benchmark_finetuned_chronos                  # all folds, all h
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import pandas as pd

from src.evaluation.metrics import mape, rmse
from src.models.foundation import ChronosBoltForecaster, FineTunedChronosBoltForecaster
from src.models.ml import RidgeForecaster
from src.training.cv import build_cv_from_config
from src.utils.io import read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)

CHECKPOINT_ROOT = Path("models/chronos_finetuned")


def evaluate_chronos(
    model_factory,
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    target_col: str,
) -> tuple[float, float, int]:
    """Run a Chronos-style forecaster: fit on train series, forecast next `n` values
    where `n = max(horizon, len(val))`. Then compare prediction at index horizon-1
    against shifted target."""
    target_h_col = f"y_h{horizon}"
    val_eval = val_df.dropna(subset=[target_h_col]).copy()
    if len(val_eval) < 5:
        return float("nan"), float("nan"), 0

    model = model_factory()
    if hasattr(model, "fold_id"):
        model.fold_id = fold_id
    if hasattr(model, "horizon"):
        model.horizon = horizon

    model.fit(train_df, target_col=target_col)
    # Chronos predicts next n future values from train end; align with val rows
    # For mode-A the evaluation: at row i in val_eval, true y is val_eval[target_h_col][i]
    # Chronos returns a flat array of forecasts of length n; we use position min(horizon-1, n-1)
    # because the model predicts the FUTURE relative to train end, and val rows are sequential
    n = len(val_eval)
    preds = model.predict(val_eval, h=horizon)
    preds = np.asarray(preds).flatten()
    if len(preds) < n:
        preds = np.concatenate([preds, np.full(n - len(preds), preds[-1])])
    preds = preds[:n]

    y_true = val_eval[target_h_col].to_numpy()
    return float(mape(y_true, preds)), float(rmse(y_true, preds)), n


def evaluate_ridge(
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    target_col: str,
) -> tuple[float, float, int]:
    """Mirror the evaluation harness used in run_ml_baselines (mode-B per-row)."""
    target_h_col = f"y_h{horizon}"
    train_with_t = train_df.copy()
    if target_h_col not in train_with_t.columns:
        train_with_t[target_h_col] = train_with_t[target_col].shift(-horizon)
    val_eval = val_df.dropna(subset=[target_h_col]).copy()
    if len(val_eval) < 5:
        return float("nan"), float("nan"), 0

    model = RidgeForecaster(horizon=horizon)
    model.fit(train_with_t, target_col=target_col)
    y_pred = model.predict(val_eval, h=horizon)
    y_pred = np.asarray(y_pred)[: len(val_eval)]
    y_true = val_eval[target_h_col].to_numpy()
    return float(mape(y_true, y_pred)), float(rmse(y_true, y_pred)), len(val_eval)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/processed/features_v2_with_sentiment.parquet")
    p.add_argument("--target-col", default="SJC_ban_ra")
    p.add_argument("--cv-config", default="configs/cv.yaml")
    p.add_argument("--folds", nargs="*", type=int, default=None,
                   help="Fold ids to benchmark (default: every fold with a saved checkpoint)")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    p.add_argument("--out-prefix", default="reports/leaderboard/chronos_finetuned")
    args = p.parse_args()

    set_global_seed(42)
    df = read_parquet(args.features)
    cv = build_cv_from_config(args.cv_config)
    folds = list(cv.split(df))

    if args.folds is None:
        available = [
            f.fold_id for f in folds
            if (CHECKPOINT_ROOT / f"fold_{f.fold_id}").exists()
        ]
        if not available:
            log.error(f"No fold checkpoints found under {CHECKPOINT_ROOT}/. Run finetune first.")
            return 1
        args.folds = available
    log.info(f"Benchmarking folds: {args.folds}, horizons: {args.horizons}")

    # Make sure y_h cols exist
    for h in args.horizons:
        col = f"y_h{h}"
        if col not in df.columns:
            df[col] = df[args.target_col].shift(-h)

    long_records: list[dict] = []
    for k in args.folds:
        fold = folds[k]
        train_df, val_df = cv.get_train_val(df, fold)

        for h in args.horizons:
            log.info(f"[fold {k} h={h}] Ridge ...")
            ridge_mape, ridge_rmse, n = evaluate_ridge(k, train_df, val_df, h, args.target_col)
            log.info(f"  Ridge MAPE={ridge_mape:.3f}% RMSE={ridge_rmse:.3f} n={n}")

            log.info(f"[fold {k} h={h}] Chronos zero-shot ...")
            zs_mape, zs_rmse, _ = evaluate_chronos(
                lambda: ChronosBoltForecaster(horizon=h),
                k, train_df, val_df, h, args.target_col,
            )
            log.info(f"  ZeroShot MAPE={zs_mape:.3f}% RMSE={zs_rmse:.3f}")

            log.info(f"[fold {k} h={h}] Chronos fine-tuned ...")
            ft_mape, ft_rmse, _ = evaluate_chronos(
                lambda: FineTunedChronosBoltForecaster(horizon=h),
                k, train_df, val_df, h, args.target_col,
            )
            log.info(f"  FineTuned MAPE={ft_mape:.3f}% RMSE={ft_rmse:.3f}")

            long_records.extend([
                {"fold": k, "horizon": h, "model": "Ridge", "MAPE": ridge_mape, "RMSE": ridge_rmse, "n": n},
                {"fold": k, "horizon": h, "model": "Chronos-Bolt-ZeroShot", "MAPE": zs_mape, "RMSE": zs_rmse, "n": n},
                {"fold": k, "horizon": h, "model": "Chronos-Bolt-FineTuned", "MAPE": ft_mape, "RMSE": ft_rmse, "n": n},
            ])

    long_df = pd.DataFrame(long_records)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(f"{args.out_prefix}_long.csv", index=False)
    summary = (
        long_df.groupby(["horizon", "model"])
        .agg(MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
             RMSE_mean=("RMSE", "mean"), n_folds=("fold", "count"))
        .reset_index()
        .sort_values(["horizon", "MAPE_mean"])
    )
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    log.info(f"\n{summary.to_string(index=False)}")
    log.info(f"Saved {args.out_prefix}_long.csv and _summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
