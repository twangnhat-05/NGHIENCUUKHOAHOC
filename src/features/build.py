"""Feature engineering V2 — orchestrate technical + macro + calendar + sentiment.

Đầu vào: data/interim/merged.parquet (sau merge + ffill + business-day filter)
Đầu ra: data/processed/features_v2.parquet (rich features cho ML/DL)

QUAN TRỌNG về data leakage:
- Lag/return/technical đều chỉ dùng quá khứ — không leak.
- Outlier winsorization KHÔNG làm ở đây — phải fit-on-train-fold (xem src/training).
- Target shift(-h) tạo y trong tương lai — đó là TARGET, không phải feature leak.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.calendar import add_calendar_features
from src.features.macro import add_macro_features
from src.features.technical import (
    bollinger,
    ema,
    macd,
    realized_volatility,
    returns,
    rsi,
    sma,
)
from src.utils.io import load_yaml, project_root, read_parquet, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================
# LAGS & RETURNS
# ============================================================

def add_lag_features(df: pd.DataFrame, lag_config: dict[str, list[int]]) -> pd.DataFrame:
    """Thêm các cột lag theo config.

    Parameters
    ----------
    df : DataFrame đã sort theo Date.
    lag_config : {column_name: [list of lag periods]}
    """
    out = df.copy()
    for col, lags in lag_config.items():
        if col not in out.columns:
            log.debug(f"Bỏ qua lag — cột không có: {col}")
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_return_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int],
    log_returns: bool = True,
) -> pd.DataFrame:
    """Thêm simple + (optional) log returns."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for p in periods:
            out[f"{col}_ret_{p}d"] = returns(out[col], period=p, log=False)
            if log_returns:
                out[f"{col}_logret_{p}d"] = returns(out[col], period=p, log=True)
    return out


# ============================================================
# TECHNICAL ON SJC + GOLD
# ============================================================

def add_technical_features(df: pd.DataFrame, tech_config: dict) -> pd.DataFrame:
    """Apply technical indicators trên SJC_ban_ra + Gold_Close (nếu có).

    Indicators áp dụng cho cả 2 cột; suffix `_sjc` hoặc `_gold`.
    """
    out = df.copy()
    target_cols = []
    if "SJC_ban_ra" in out.columns:
        target_cols.append(("SJC_ban_ra", "sjc"))
    if "Gold_Close" in out.columns:
        target_cols.append(("Gold_Close", "gold"))

    for col, suffix in target_cols:
        s = out[col]

        # SMA & EMA
        for w in tech_config.get("sma", []):
            out[f"sma{w}_{suffix}"] = sma(s, w)
        for w in tech_config.get("ema", []):
            out[f"ema{w}_{suffix}"] = ema(s, w)

        # RSI
        for p in tech_config.get("rsi", []):
            out[f"rsi{p}_{suffix}"] = rsi(s, p)

        # MACD
        macd_cfg = tech_config.get("macd")
        if macd_cfg:
            macd_df = macd(s, fast=macd_cfg["fast"], slow=macd_cfg["slow"], signal=macd_cfg["signal"])
            for c in macd_df.columns:
                out[f"{c}_{suffix}"] = macd_df[c]

        # Bollinger
        bb_cfg = tech_config.get("bollinger")
        if bb_cfg:
            bb_df = bollinger(s, window=bb_cfg["window"], n_std=bb_cfg["n_std"])
            for c in bb_df.columns:
                out[f"{c}_{suffix}"] = bb_df[c]

        # Realized vol on returns
        ret = returns(s, period=1, log=False)
        for w in tech_config.get("realized_vol", []):
            out[f"realvol{w}_{suffix}"] = realized_volatility(ret, window=w)

    return out


# ============================================================
# TARGETS
# ============================================================

def add_targets(df: pd.DataFrame, target_col: str, horizons: list[int]) -> pd.DataFrame:
    """Thêm cột target cho mỗi horizon: y_h{h} = target_col.shift(-h)."""
    out = df.copy()
    if target_col not in out.columns:
        raise KeyError(f"target_col {target_col} không có trong df")
    for h in horizons:
        out[f"y_h{h}"] = out[target_col].shift(-h)
    return out


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_features(
    df: pd.DataFrame,
    features_config: dict,
    drop_na_targets: bool = True,
) -> pd.DataFrame:
    """Pipeline đầy đủ: lag → return → technical → calendar → macro → target."""
    log.info(f"Input: {df.shape}, columns: {list(df.columns)}")

    # 1) Lags
    out = add_lag_features(df, features_config.get("lags", {}))
    log.info(f"After lags: {out.shape}")

    # 2) Returns
    ret_cfg = features_config.get("returns", {})
    out = add_return_features(
        out,
        columns=ret_cfg.get("columns", []),
        periods=ret_cfg.get("periods", [1]),
        log_returns=ret_cfg.get("log_returns", False),
    )
    log.info(f"After returns: {out.shape}")

    # 3) Technical
    out = add_technical_features(out, features_config.get("technical", {}))
    log.info(f"After technical: {out.shape}")

    # 4) Calendar
    out = add_calendar_features(out)
    log.info(f"After calendar: {out.shape}")

    # 5) Macro derivatives
    out = add_macro_features(out)
    log.info(f"After macro: {out.shape}")

    # 6) Targets
    target_col = features_config["target_column"]
    horizons = features_config["target_horizons"]
    out = add_targets(out, target_col, horizons)
    log.info(f"After targets (horizons={horizons}): {out.shape}")

    # 7) Drop rows với target NaN (cuối chuỗi cho h lớn nhất)
    if drop_na_targets:
        max_h = max(horizons)
        before = len(out)
        # Drop rows mà TẤT CẢ horizon target đều NaN
        target_cols = [f"y_h{h}" for h in horizons]
        out = out.dropna(subset=target_cols, how="all")
        log.info(f"After drop NaN targets: {len(out)} rows (drop {before - len(out)} ở cuối, max horizon={max_h})")

    # 8) Drop rows đầu chuỗi với feature NaN do lag/SMA/etc
    before = len(out)
    out = out.dropna()
    log.info(f"After drop NaN features (warm-up): {len(out)} rows (drop {before - len(out)})")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build features V2")
    parser.add_argument("--input", default="data/interim/merged.parquet")
    parser.add_argument("--output", default="data/processed/features_v2.parquet")
    parser.add_argument("--config", default="configs/features.yaml")
    args = parser.parse_args()

    df_in = read_parquet(args.input)
    cfg = load_yaml(args.config)
    df_out = build_features(df_in, cfg, drop_na_targets=True)
    out_path = project_root() / args.output
    write_parquet(df_out, out_path)
    log.info(f"Saved: {out_path}, shape={df_out.shape}, n_features={len(df_out.columns) - 4}")
    log.info(f"Date range: {df_out['Date'].min().date()} → {df_out['Date'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
