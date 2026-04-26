"""Merge raw data sources → interim/merged.parquet.

Outer join trên Date, ffill (no bfill — tránh look-ahead bias).
Outlier xử lý KHÔNG ở đây — defer cho features layer (fit-on-train-fold).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_yaml, project_root, read_csv_safe, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)


def _normalize_date(series: pd.Series) -> pd.Series:
    """Bỏ timezone, chỉ giữ date (YYYY-MM-DD)."""
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.to_datetime(s.dt.date)


def _load_sjc(path: Path) -> pd.DataFrame:
    df = read_csv_safe(path)
    df = df.rename(columns={"date": "Date"})
    df["Date"] = _normalize_date(df["Date"])
    df = df.rename(columns={"mua_vao": "SJC_mua_vao", "ban_ra": "SJC_ban_ra"})
    return df[["Date", "SJC_mua_vao", "SJC_ban_ra"]]


def _load_ohlcv(path: Path, prefix: str, keep: tuple[str, ...] = ("Close",)) -> pd.DataFrame:
    """Load OHLCV file, keep selected columns with prefix_<col>."""
    df = read_csv_safe(path)
    df["Date"] = _normalize_date(df["Date"])
    cols_keep = ["Date"] + [c for c in keep if c in df.columns]
    df = df[cols_keep]
    rename = {c: f"{prefix}_{c}" for c in keep if c in df.columns}
    df = df.rename(columns=rename)
    return df


def _load_fred(path: Path, value_col: str, rename: str) -> pd.DataFrame:
    df = read_csv_safe(path)
    df["Date"] = _normalize_date(df["Date"])
    if value_col not in df.columns:
        # fallback: lấy cột số đầu tiên không phải Date
        candidates = [c for c in df.columns if c != "Date"]
        if not candidates:
            raise ValueError(f"FRED file {path.name} chỉ có cột Date")
        value_col = candidates[0]
    return df[["Date", value_col]].rename(columns={value_col: rename}).drop_duplicates("Date")


# ============================================================
# MAIN MERGE
# ============================================================

def load_all_raw(config_path: str = "configs/data.yaml") -> dict[str, pd.DataFrame]:
    """Load mọi raw file vào dict {name: DataFrame}."""
    cfg = load_yaml(config_path)
    raw_dir = project_root() / cfg["paths"]["raw_dir"]

    loaders: dict[str, pd.DataFrame] = {}

    if (raw_dir / "SJC_gold_historical.csv").exists():
        loaders["sjc"] = _load_sjc(raw_dir / "SJC_gold_historical.csv")

    # OHLCV files: keep Close (simple) + Volume cho Gold/Oil/VN-Index
    ohlcv_map = {
        "Gold_Futures":  ("Gold",  ("Close", "Volume")),
        "USD_Index":     ("USD",   ("Close",)),
        "VN_Index":      ("VNI",   ("Close", "Volume")),
        "Oil_WTI":       ("Oil",   ("Close", "Volume")),
        "USD_VND":       ("USDVND", ("Close",)),
        "GLD":           ("GLD",   ("Close",)),
        "BTC":           ("BTC",   ("Close",)),
    }
    for fname, (prefix, keep) in ohlcv_map.items():
        p = raw_dir / f"{fname}_ohlcv.csv"
        if p.exists():
            loaders[fname.lower()] = _load_ohlcv(p, prefix=prefix, keep=keep)

    # FRED indicators
    fred_map = {
        "interest_rate.csv": ("Interest_Rate", "Interest_Rate_FED"),
        "dtwexbgs.csv":      ("DTWEXBGS",      "USD_Broad_Index"),
        "teny_treasury.csv": ("TenY_Treasury", "TenY_Treasury"),
    }
    for fname, (val_col, rename) in fred_map.items():
        p = raw_dir / fname
        if p.exists():
            loaders[fname.split(".")[0]] = _load_fred(p, value_col=val_col, rename=rename)

    return loaders


def merge_on_date(loaders: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer join tất cả DataFrames trên cột Date."""
    merged: pd.DataFrame | None = None
    for name, df in loaders.items():
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="Date", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def forward_fill_no_bfill(df: pd.DataFrame, target_col: str = "SJC_ban_ra") -> pd.DataFrame:
    """ffill toàn bộ + dropna bằng target.

    KHÔNG dùng bfill (sẽ leak future).
    Drop rows nơi target = NaN (đầu chuỗi hoặc thiếu).
    """
    df = df.ffill()
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    # Drop rows còn NaN bất kỳ (sau khi ffill thì chỉ là đầu chuỗi)
    df = df.dropna()
    return df.reset_index(drop=True)


def filter_business_days(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Giữ chỉ business days (T2-T6) — SJC không giao dịch weekend.

    Weekend rows tồn tại do BTC trade 7/7; ffill weekend SJC sẽ gây stale value.
    Quyết định: drop weekends để align với SJC scraper schedule.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    is_bday = df[date_col].dt.dayofweek < 5  # 0=Mon ... 4=Fri
    return df[is_bday].reset_index(drop=True)


def merge_all(
    config_path: str = "configs/data.yaml",
    start_date: str | None = None,
    output_path: str | None = None,
    business_days_only: bool = True,
) -> pd.DataFrame:
    """Pipeline đầy đủ: load → merge → filter business days → filter date → ffill → save parquet."""
    cfg = load_yaml(config_path)
    if start_date is None:
        start_date = cfg["start_date"]

    loaders = load_all_raw(config_path)
    if not loaders:
        raise RuntimeError("Không load được nguồn nào — check data/raw/")
    log.info(f"Loaded {len(loaders)} sources: {list(loaders.keys())}")

    merged = merge_on_date(loaders)
    log.info(f"After merge: {len(merged)} rows, {len(merged.columns)} cols, "
             f"range={merged['Date'].min().date()}→{merged['Date'].max().date()}")

    if business_days_only:
        before = len(merged)
        merged = filter_business_days(merged)
        log.info(f"After filter business days: {len(merged)} rows (drop {before - len(merged)} weekend rows)")

    # Filter time range
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged[merged["Date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    log.info(f"After filter start_date={start_date}: {len(merged)} rows")

    # Forward fill (no bfill — never look ahead)
    merged = forward_fill_no_bfill(merged)
    log.info(f"After ffill + dropna: {len(merged)} rows")

    # Save
    out_path = Path(output_path) if output_path else (
        project_root() / cfg["paths"]["interim_dir"] / "merged.parquet"
    )
    write_parquet(merged, out_path)
    log.info(f"Saved to {out_path}")
    return merged


def main() -> int:
    """CLI entry."""
    import argparse
    parser = argparse.ArgumentParser(description="Merge raw → interim/merged.parquet")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    df = merge_all(args.config, args.start_date, args.output)
    log.info(f"Done. Final shape={df.shape}, columns={list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
