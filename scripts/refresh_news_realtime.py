"""Real-time news refresh orchestrator (Phase 9).

Single-shot CLI suitable for invocation from a cron job. Fetches the latest
news from every configured free-tier source, deduplicates against the local
cache, and writes an updated parquet.

Usage:
    python -m scripts.refresh_news_realtime
    python -m scripts.refresh_news_realtime --keep-days 30 --score-recent 30
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src.features.news_realtime import fetch_all_realtime
from src.utils.io import project_root, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_CACHE = "data/external/news_realtime.parquet"
DEFAULT_SCORED = "data/external/news_realtime_scored.parquet"


def aggregate_daily_tone(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GDELT tone (the only native-sentiment source) per day.

    Reddit and RSS rows are counted but not scored here — that requires the
    heavier mDeBERTa pipeline run separately."""
    if df.empty:
        return pd.DataFrame()
    g = df[df["source"] == "gdelt"].copy()
    if g.empty:
        return pd.DataFrame()
    g["tone"] = pd.to_numeric(g["tone"], errors="coerce")
    daily = (
        g.groupby("date")
        .agg(
            tone_mean=("tone", "mean"),
            tone_std=("tone", "std"),
            tone_count=("tone", "count"),
        )
        .reset_index()
        .sort_values("date")
    )
    return daily


def maybe_score_recent(
    df: pd.DataFrame,
    n_recent: int,
    output_path: str,
) -> None:
    """Score the most recent N un-scored items via mDeBERTa zero-shot.

    Heavy: ~0.3s per headline on CPU. Run only when explicitly requested via
    --score-recent so the regular cron stays fast (<10s)."""
    if n_recent <= 0:
        return
    try:
        from src.features.news_score import score_news_zero_shot
    except Exception as e:
        log.warning(f"Cannot import scorer: {e}")
        return

    out_full = project_root() / output_path
    if out_full.exists():
        scored = pd.read_parquet(out_full)
        already = set(scored["url"].dropna()) if "url" in scored.columns else set()
    else:
        scored = pd.DataFrame()
        already = set()

    candidates = (
        df[~df["url"].isin(already)]
        .sort_values("ts", ascending=False)
        .head(n_recent)
        .copy()
    )
    if candidates.empty:
        log.info("No new items to score")
        return

    log.info(f"Scoring {len(candidates)} headlines via mDeBERTa (zero-shot, CPU)…")
    new_scored = score_news_zero_shot(candidates, batch_size=4, text_col="title")
    combined = pd.concat([scored, new_scored], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"], keep="last")
    write_parquet(combined, out_full)
    log.info(f"Scored cache: {out_full.name} ({len(combined)} total)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--scored", default=DEFAULT_SCORED)
    p.add_argument("--keep-days", type=int, default=90)
    p.add_argument("--score-recent", type=int, default=0,
                   help="Score N most-recent un-scored headlines via mDeBERTa "
                        "(slow; default 0 = skip)")
    args = p.parse_args()

    df = fetch_all_realtime(args.cache, keep_days=args.keep_days)
    if df.empty:
        # No cache and no fetch — first-run failure. Exit 0 so cron job is
        # marked successful (will retry next slot). Errors already logged.
        log.warning("Empty news cache and no successful fetch this run")
        return 0

    by_source = df["source"].value_counts().to_dict() if "source" in df else {}
    log.info(f"Sources: {by_source}")

    daily = aggregate_daily_tone(df)
    if not daily.empty:
        last7 = daily.tail(7)
        log.info("Last-7-day GDELT tone:")
        for _, row in last7.iterrows():
            log.info(f"  {row['date']}: tone={row['tone_mean']:+.2f} "
                     f"std={row['tone_std']:.2f} n={int(row['tone_count'])}")

    if args.score_recent > 0:
        maybe_score_recent(df, args.score_recent, args.scored)

    return 0


if __name__ == "__main__":
    sys.exit(main())
