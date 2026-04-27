"""Feature-family ablation study (cumulative add).

For paper Section 4.5 — quantifies marginal contribution of each feature family.
Runs Ridge + ElasticNet on 7 nested feature subsets across 5 walk-forward folds × 3 horizons.

Output: reports/ablation/ablation_summary.csv + ablation_long.csv

Usage: python -m scripts.run_ablation_features
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import mape
from src.training.cv import build_cv_from_config
from src.utils.io import read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)

FEATURES_PATH = "data/processed/features_v2_with_sentiment.parquet"
TARGET_COL = "SJC_ban_ra"
HORIZONS = (1, 5, 20)
OUT_DIR = Path("reports/ablation")


def family_of(col: str) -> str:
    """Map a feature column to its family."""
    if col.startswith("sentiment"):
        return "sentiment"
    if col.startswith("cal_"):
        return "calendar"
    if re.search(r"_(ret|logret)_\d+d$", col):
        return "returns"
    if re.match(r"(sma|ema|rsi|macd|signal|hist|bb_|realvol)\d*", col):
        return "technical"
    if "_lag" in col:
        return "lag"
    if col in {
        "USDVND_change_1d", "USDVND_change_5d", "USDVND_change_20d",
        "USD_z_gap", "yield_spread_10Y_FED", "USD_realized_vol_20d",
        "sjc_gold_ratio",
    }:
        return "macro_derived"
    return "raw_macro"  # Gold_Close, USD_Close, USDVND_Close, BTC_Close, etc.


SUBSETS: list[tuple[str, set[str]]] = [
    ("S1_lag_only", {"raw_macro", "lag"}),
    ("S2_plus_returns", {"raw_macro", "lag", "returns"}),
    ("S3_plus_technical", {"raw_macro", "lag", "returns", "technical"}),
    ("S4_plus_macro", {"raw_macro", "lag", "returns", "technical", "macro_derived"}),
    ("S5_plus_calendar", {"raw_macro", "lag", "returns", "technical", "macro_derived", "calendar"}),
    ("S6_full", {"raw_macro", "lag", "returns", "technical", "macro_derived", "calendar", "sentiment"}),
]


def select_columns(df: pd.DataFrame, families: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in {"Date", TARGET_COL, "SJC_mua_vao"}:
            continue
        if c.startswith("y_h"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if family_of(c) in families:
            cols.append(c)
    return cols


def fit_predict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_h_col: str,
    estimator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit + predict on train/val using only `feature_cols`. Returns (y_true, y_pred)."""
    valid_train = train_df.dropna(subset=[target_h_col]).copy()
    X_train = valid_train[feature_cols].ffill().bfill().fillna(0.0).values
    y_train = valid_train[target_h_col].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    estimator.fit(X_train, y_train)

    val_eval = val_df.dropna(subset=[target_h_col]).copy()
    X_val = val_eval[feature_cols].ffill().bfill().fillna(0.0).values
    X_val = scaler.transform(X_val)
    y_pred = estimator.predict(X_val)
    y_true = val_eval[target_h_col].values
    return y_true, y_pred


def main() -> int:
    set_global_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_parquet(FEATURES_PATH)
    log.info(f"Loaded features: {df.shape}")

    # Pre-compute target columns for each horizon
    for h in HORIZONS:
        col = f"y_h{h}"
        if col not in df.columns:
            df[col] = df[TARGET_COL].shift(-h)

    cv = build_cv_from_config("configs/cv.yaml")
    folds = list(cv.split(df))
    log.info(f"CV folds: {len(folds)}")

    # Family count summary
    log.info("Feature counts per subset (h=1):")
    for name, fams in SUBSETS:
        n = len(select_columns(df, fams))
        log.info(f"  {name}: {n} features (families={sorted(fams)})")

    long_records: list[dict] = []
    for h in HORIZONS:
        target_h_col = f"y_h{h}"
        for subset_name, families in SUBSETS:
            feature_cols = select_columns(df, families)
            n_feat = len(feature_cols)
            for est_name, est_factory in [
                ("Ridge", lambda: Ridge(alpha=1.0, random_state=42)),
                ("ElasticNet",
                 lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)),
            ]:
                for fold in folds:
                    train_df, val_df = cv.get_train_val(df, fold)
                    y_true, y_pred = fit_predict(
                        train_df, val_df, feature_cols, target_h_col, est_factory(),
                    )
                    if len(y_true) == 0:
                        continue
                    m = mape(y_true, y_pred)
                    long_records.append({
                        "horizon": h,
                        "subset": subset_name,
                        "model": est_name,
                        "fold_id": fold.fold_id,
                        "n_features": n_feat,
                        "mape": m,
                        "n_eval": len(y_true),
                    })
            log.info(f"  done h={h} subset={subset_name} ({n_feat} feat) ×2 models ×{len(folds)} folds")

    long_df = pd.DataFrame(long_records)
    long_path = OUT_DIR / "ablation_long.csv"
    long_df.to_csv(long_path, index=False)
    log.info(f"Saved {long_path} ({len(long_df)} records)")

    summary = (
        long_df.groupby(["horizon", "subset", "model"])
        .agg(mape_mean=("mape", "mean"),
             mape_std=("mape", "std"),
             n_features=("n_features", "first"),
             n_folds=("fold_id", "count"))
        .reset_index()
        .sort_values(["horizon", "model", "subset"])
    )
    summary_path = OUT_DIR / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved {summary_path}")
    log.info(f"Summary preview:\n{summary.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
