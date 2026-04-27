"""Live-news tone adjustment for the SJC forecast (Phase 10).

Reads the cached news_realtime.parquet, takes the most recent GDELT articles
within a configurable look-back window, returns the mean tone (-100..+100),
and produces a small calibrated multiplicative shift that the dashboard can
apply on top of the base forecast.

Calibration prior is intentionally tiny (alpha = 1e-3 per tone-point) — the
historical CV window does not overlap the live news archive (see paper §4.5),
so we cannot retro-fit the coefficient yet. The dashboard surfaces both the
base and adjusted forecast plus a short explanation so the user can judge
the residual uncertainty themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DEFAULT_CACHE = Path("data/external/news_realtime.parquet")
DEFAULT_WINDOW_HOURS = 24
DEFAULT_ALPHA = 1e-3  # tiny prior; see module docstring


@dataclass(frozen=True)
class ToneSnapshot:
    """Aggregated live-news tone for a single look-back window."""

    tone_mean: float | None      # mean GDELT V2Tone, -100..+100, or None if no data
    tone_std: float | None       # std across articles in the window
    n_articles: int              # number of GDELT articles aggregated
    window_hours: int            # look-back window
    latest_ts: pd.Timestamp | None
    age_minutes: float | None    # minutes since latest article
    fallback_reason: str | None  # filled if no usable data was found


def compute_live_tone(
    cache_path: str | Path = DEFAULT_CACHE,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
) -> ToneSnapshot:
    """Aggregate GDELT tone from the most recent `window_hours` of cached news.

    Returns a ToneSnapshot. If the cache is missing or has no GDELT rows in
    the window, returns a snapshot with `tone_mean=None` and a fallback reason.
    """
    now = now or datetime.now(timezone.utc)
    p = Path(cache_path)
    if not p.exists():
        return ToneSnapshot(None, None, 0, window_hours, None, None,
                            "news cache not found")

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        return ToneSnapshot(None, None, 0, window_hours, None, None,
                            f"cache read failed: {e}")

    if df.empty or "source" not in df.columns:
        return ToneSnapshot(None, None, 0, window_hours, None, None,
                            "cache empty")

    df = df[df["source"] == "gdelt"].copy()
    if df.empty:
        return ToneSnapshot(None, None, 0, window_hours, None, None,
                            "no GDELT articles in cache")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    cutoff = pd.Timestamp(now) - timedelta(hours=window_hours)
    recent = df[df["ts"] >= cutoff].copy()
    if recent.empty:
        latest = df["ts"].max()
        age = (pd.Timestamp(now) - latest).total_seconds() / 60.0
        return ToneSnapshot(None, None, 0, window_hours, latest, age,
                            f"no articles within last {window_hours}h "
                            f"(latest is {age/60:.1f}h old)")

    recent["tone"] = pd.to_numeric(recent["tone"], errors="coerce")
    recent = recent.dropna(subset=["tone"])
    if recent.empty:
        return ToneSnapshot(None, None, 0, window_hours, None, None,
                            "GDELT articles in window have no tone field")

    latest = recent["ts"].max()
    age = (pd.Timestamp(now) - latest).total_seconds() / 60.0
    return ToneSnapshot(
        tone_mean=float(recent["tone"].mean()),
        tone_std=float(recent["tone"].std()) if len(recent) > 1 else 0.0,
        n_articles=int(len(recent)),
        window_hours=window_hours,
        latest_ts=latest,
        age_minutes=age,
        fallback_reason=None,
    )


def apply_tone_adjustment(
    base_prediction: float,
    tone: float | None,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Apply a small multiplicative shift to a base prediction.

    Returns (adjusted_prediction, delta_pct) where:
        adjusted = base * (1 + alpha * tone / 100)
        delta_pct = (adjusted - base) / base * 100
    If `tone` is None or NaN, returns (base, 0.0) — no-op.
    """
    if tone is None or pd.isna(tone):
        return float(base_prediction), 0.0
    factor = 1.0 + alpha * (tone / 100.0)
    adjusted = float(base_prediction) * factor
    delta_pct = (adjusted - base_prediction) / base_prediction * 100.0 \
        if base_prediction else 0.0
    return adjusted, delta_pct
