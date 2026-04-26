"""CLI: refresh raw data — chỉ tải DELTA (từ ngày cuối + 1 đến today).

Usage:
    python -m src.data.refresh                       # delta refresh, all sources
    python -m src.data.refresh --full                # full re-download (chậm, ~1h vì SJC)
    python -m src.data.refresh --sources yfinance    # chỉ refresh yfinance
    python -m src.data.refresh --skip sjc            # bỏ qua SJC (scraping chậm)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data.fetch import (
    fetch_all_full,
    fetch_fred_series,
    fetch_sjc_range,
    fetch_vnindex_vnstock,
    fetch_yfinance,
)
from src.utils.io import load_yaml, project_root, read_csv_safe
from src.utils.logging import get_logger

log = get_logger(__name__)


def _last_date_in(path: Path, date_col_candidates: tuple[str, ...] = ("Date", "date")) -> pd.Timestamp | None:
    """Lấy ngày cuối cùng trong file CSV (nếu có)."""
    if not path.exists():
        return None
    try:
        df = read_csv_safe(path)
    except Exception as e:
        log.warning(f"Không đọc được {path.name}: {e}")
        return None
    date_col = next((c for c in date_col_candidates if c in df.columns), None)
    if date_col is None:
        return None
    s = pd.to_datetime(df[date_col], utc=True, errors="coerce").dropna()
    return s.max().tz_localize(None) if len(s) else None


def refresh_delta(config_path: str = "configs/data.yaml", skip: tuple[str, ...] = ()) -> dict[str, str]:
    """Refresh delta: chỉ tải từ ngày cuối + 1 → today, MERGE vào raw file hiện có."""
    cfg = load_yaml(config_path)
    out_dir = project_root() / cfg["paths"]["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    summary: dict[str, str] = {}

    # ----- yfinance: full re-download for those nguồn (yfinance fast, idempotent) -----
    if "yfinance" not in skip:
        for ticker, name in cfg["yfinance"].items():
            if ticker in ("retry", "delay_seconds"):
                continue
            last = _last_date_in(out_dir / f"{name}_ohlcv.csv")
            if last is not None and last.date() >= today - timedelta(days=1):
                summary[name] = f"up-to-date ({last.date()})"
                continue
            ok = fetch_yfinance(
                ticker=ticker, name=name,
                start_date=cfg["start_date"], end_date=today_str, out_dir=out_dir,
                retries=cfg["yfinance"]["retry"], delay=cfg["yfinance"]["delay_seconds"],
            )
            new_last = _last_date_in(out_dir / f"{name}_ohlcv.csv")
            summary[name] = f"refreshed → {new_last.date() if new_last is not None else 'FAIL'}" if ok else "FAIL"

    # ----- vnstock VN-Index -----
    if "vnstock" not in skip:
        last = _last_date_in(out_dir / "VN_Index_ohlcv.csv")
        if last is not None and last.date() >= today - timedelta(days=1):
            summary["VN_Index"] = f"up-to-date ({last.date()})"
        else:
            ok = fetch_vnindex_vnstock(cfg["start_date"], today_str, out_dir)
            new_last = _last_date_in(out_dir / "VN_Index_ohlcv.csv")
            summary["VN_Index"] = f"refreshed → {new_last.date() if new_last is not None else 'FAIL'}" if ok else "FAIL"

    # ----- FRED -----
    if "fred" not in skip:
        for name, code in cfg["fred"].items():
            out_path = out_dir / "interest_rate.csv" if name == "FEDFUNDS" else out_dir / f"{name.lower()}.csv"
            rename = {"FEDFUNDS": "Interest_Rate"}.get(name, name)
            last = _last_date_in(out_path)
            # FRED là monthly cho FEDFUNDS — chỉ refresh nếu > 30 ngày stale
            if last is not None and last.date() >= today - timedelta(days=15):
                summary[f"FRED_{name}"] = f"up-to-date ({last.date()})"
                continue
            ok = fetch_fred_series(code, cfg["start_date"], today_str, out_path, rename_to=rename)
            new_last = _last_date_in(out_path)
            summary[f"FRED_{name}"] = f"refreshed → {new_last.date() if new_last is not None else 'FAIL'}" if ok else "FAIL"

    # ----- SJC (delta only — scraping is expensive) -----
    if "sjc" not in skip:
        sjc_path = out_dir / "SJC_gold_historical.csv"
        last = _last_date_in(sjc_path, date_col_candidates=("date", "Date"))
        if last is None:
            log.info("SJC chưa có file → tải full từ start_date.")
            sjc_start = datetime.strptime(cfg["start_date"], "%Y-%m-%d")
        else:
            if last.date() >= today - timedelta(days=1):
                summary["SJC"] = f"up-to-date ({last.date()})"
                return summary
            sjc_start = (last + timedelta(days=1)).to_pydatetime()
            log.info(f"SJC delta refresh: {sjc_start.date()} → {today}")
        sjc_cfg = cfg["sjc"]
        df_new = fetch_sjc_range(
            start_date=sjc_start, end_date=datetime.combine(today, datetime.min.time()),
            base_url=sjc_cfg["base_url"], user_agent=sjc_cfg["user_agent"],
            delay=sjc_cfg["delay_seconds"], weekday_only=sjc_cfg["weekday_only"],
        )
        if df_new.empty:
            summary["SJC"] = f"no new data (last={last.date() if last else 'N/A'})"
        else:
            if sjc_path.exists() and last is not None:
                df_old = read_csv_safe(sjc_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            else:
                df_combined = df_new
            df_combined.to_csv(sjc_path, index=False, encoding="utf-8-sig")
            new_last = _last_date_in(sjc_path, date_col_candidates=("date", "Date"))
            summary["SJC"] = f"appended {len(df_new)} rows → {new_last.date() if new_last else 'N/A'}"

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh raw data (delta or full)")
    parser.add_argument("--full", action="store_true", help="Full re-download (slow, especially SJC)")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--skip", nargs="*", default=[], choices=["yfinance", "vnstock", "fred", "sjc"],
                        help="Skip specific source(s)")
    args = parser.parse_args()

    if args.full:
        log.info("FULL re-download (có thể mất ~1 giờ vì SJC scraper)")
        results = fetch_all_full(args.config)
        for k, v in results.items():
            log.info(f"  {k}: {'OK' if v else 'FAIL'}")
    else:
        log.info("DELTA refresh (chỉ tải dữ liệu mới)")
        summary = refresh_delta(args.config, skip=tuple(args.skip))
        log.info("=" * 60)
        log.info("REFRESH SUMMARY:")
        for k, v in summary.items():
            log.info(f"  {k:20s} {v}")
        log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
