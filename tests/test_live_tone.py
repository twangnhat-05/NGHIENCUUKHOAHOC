"""Unit tests for src.forecast.live_tone (Phase 10)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.forecast.live_tone import (
    apply_tone_adjustment,
    compute_live_tone,
    DEFAULT_ALPHA,
    ToneSnapshot,
)


def _write_tone_cache(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "gdelt_tone_timeline.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(p, index=False)
    return p


def test_compute_live_tone_returns_fallback_when_cache_missing(tmp_path):
    snap = compute_live_tone(cache_path=tmp_path / "missing.parquet")
    assert snap.tone_mean is None
    assert snap.n_articles == 0
    assert snap.fallback_reason is not None


def test_compute_live_tone_filters_to_window(tmp_path):
    now = datetime(2026, 4, 27, 10, 0, tzinfo=timezone.utc)
    rows = [
        {"ts": (now - timedelta(hours=2)).isoformat(),  "tone": 5.0},
        {"ts": (now - timedelta(hours=20)).isoformat(), "tone": -3.0},
        # too old — excluded by 24h window
        {"ts": (now - timedelta(hours=48)).isoformat(), "tone": 100.0},
    ]
    p = _write_tone_cache(tmp_path, rows)
    snap = compute_live_tone(cache_path=p, window_hours=24, now=now)
    assert isinstance(snap, ToneSnapshot)
    assert snap.tone_mean == pytest.approx(1.0)  # mean of 5, -3
    assert snap.n_articles == 2
    assert snap.fallback_reason is None


def test_compute_live_tone_handles_no_recent_articles(tmp_path):
    now = datetime(2026, 4, 27, 10, 0, tzinfo=timezone.utc)
    rows = [
        {"ts": (now - timedelta(days=5)).isoformat(), "tone": 7.0},
    ]
    p = _write_tone_cache(tmp_path, rows)
    snap = compute_live_tone(cache_path=p, window_hours=24, now=now)
    assert snap.tone_mean is None
    assert "no tone data" in snap.fallback_reason.lower()


def test_apply_tone_adjustment_noop_when_tone_none():
    out, delta = apply_tone_adjustment(100.0, None)
    assert out == 100.0
    assert delta == 0.0


def test_apply_tone_adjustment_positive_tone_lifts():
    base = 95.0
    out, delta = apply_tone_adjustment(base, 50.0, alpha=DEFAULT_ALPHA)
    assert out > base
    assert delta == pytest.approx(0.05, abs=1e-6)  # alpha=1e-3 * 50/100 = 5e-4 -> 0.05%


def test_apply_tone_adjustment_negative_tone_drops():
    base = 95.0
    out, delta = apply_tone_adjustment(base, -100.0, alpha=DEFAULT_ALPHA)
    assert out < base
    assert delta == pytest.approx(-0.1, abs=1e-6)
