"""Real-time gold-relevant news ingester (Phase 9).

Multi-source pipeline that extends the Phase 2 batch fetcher with:
  • GDELT 2.0 DOC API — global news with built-in V2Tone sentiment
  • Reddit JSON   — /r/Gold, /r/wallstreetbets, /r/Forex public feeds
  • Investing.com Gold RSS / Kitco RSS — financial news
  • Existing Google News + CafeF + yfinance (re-used from Phase 2)

All sources are 100% free tier. No API keys required.
Time granularity: ISO timestamps (not just date) for true real-time aggregation.

Output: data/external/news_realtime.parquet (append-only with dedupe)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from urllib.parse import quote
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from src.utils.io import project_root, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; NCKH-TDTU-research/1.0)"
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT = 20

NEWS_SCHEMA = [
    "ts",        # ISO 8601 UTC timestamp
    "date",      # YYYY-MM-DD UTC
    "source",    # gdelt / reddit / google / cafef / kitco / investing / yfinance
    "title",
    "summary",
    "url",
    "domain",
    "tone",      # native sentiment from source if available, else NaN
    "lang",      # en / vi / unknown
]


# ============================================================
# GDELT 2.0 DOC API — global news with built-in tone
# ============================================================

def fetch_gdelt_doc_api(
    query: str = "(gold OR \"gold price\" OR bullion OR \"precious metals\")",
    timespan: str = "1d",
    max_records: int = 100,
    sort: str = "datedesc",
) -> pd.DataFrame:
    """Fetch articles from GDELT 2.0 DOC API.

    Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    Free, no auth, ~15-min update cadence.
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "timespan": timespan,
        "maxrecords": str(max_records),
        "sort": sort,
    }
    url = base + "?" + "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"GDELT DOC API failed: {e}")
        return pd.DataFrame(columns=NEWS_SCHEMA)

    rows = []
    for art in data.get("articles", []):
        ts_raw = art.get("seendate", "")  # YYYYMMDDTHHMMSSZ
        try:
            ts = datetime.strptime(ts_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append({
            "ts": ts.isoformat(),
            "date": ts.date().isoformat(),
            "source": "gdelt",
            "title": art.get("title", "").strip(),
            "summary": "",
            "url": art.get("url", ""),
            "domain": art.get("domain", ""),
            "tone": float(art.get("tone", 0.0)) if art.get("tone") is not None else None,
            "lang": art.get("language", "unknown")[:2].lower(),
        })
    log.info(f"GDELT: {len(rows)} articles (timespan={timespan})")
    return pd.DataFrame(rows, columns=NEWS_SCHEMA)


# ============================================================
# REDDIT JSON — public feeds, no auth
# ============================================================

def fetch_reddit_json(subreddits: list[str] | None = None, limit: int = 50) -> pd.DataFrame:
    """Fetch new posts from public subreddits via JSON endpoint."""
    subreddits = subreddits or ["Gold", "wallstreetbets", "Forex", "preciousmetals"]
    rows = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit={limit}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            posts = r.json().get("data", {}).get("children", [])
        except Exception as e:
            log.warning(f"Reddit /r/{sub} failed: {e}")
            continue
        for p in posts:
            d = p.get("data", {})
            title = (d.get("title") or "").strip()
            if not title:
                continue
            ts = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc)
            rows.append({
                "ts": ts.isoformat(),
                "date": ts.date().isoformat(),
                "source": "reddit",
                "title": title,
                "summary": (d.get("selftext") or "")[:500],
                "url": "https://reddit.com" + d.get("permalink", ""),
                "domain": f"reddit.com/r/{sub}",
                "tone": None,
                "lang": "en",
            })
        time.sleep(0.5)  # be polite
    log.info(f"Reddit: {len(rows)} posts across {len(subreddits)} subs")
    return pd.DataFrame(rows, columns=NEWS_SCHEMA)


# ============================================================
# Generic RSS fetcher (Investing.com, Kitco, etc.)
# ============================================================

RSS_SOURCES: list[tuple[str, str, str]] = [
    # (source_id, feed_url, lang)
    ("kitco", "https://www.kitco.com/rss/KitcoNews.xml", "en"),
    ("investing", "https://www.investing.com/rss/news_25.rss", "en"),  # commodities
    ("cafef_vang", "https://cafef.vn/thi-truong-chung-khoan.rss", "vi"),
]


def fetch_rss(source_id: str, url: str, lang: str,
              gold_filter: bool = True) -> pd.DataFrame:
    """Generic RSS fetcher. If `gold_filter=True`, keep only gold-related items
    (broad keyword match) to reduce noise from non-gold feeds like cafef-broad."""
    rows = []
    keywords = ["gold", "vàng", "sjc", "bullion", "precious metal",
                "kim loại quý", "fed", "usd"]
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning(f"RSS {source_id} failed: {e}")
        return pd.DataFrame(columns=NEWS_SCHEMA)

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        if gold_filter and not any(k.lower() in title.lower() for k in keywords):
            continue
        pub = item.findtext("pubDate") or ""
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "")[:500]
        # Try multiple date formats
        ts = None
        for fmt in ("%a, %d %b %Y %H:%M:%S %z",
                    "%a, %d %b %Y %H:%M:%S %Z",
                    "%a, %d %b %Y %H:%M:%S GMT"):
            try:
                ts = datetime.strptime(pub, fmt)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        if ts is None:
            continue
        domain = link.split("/")[2] if "://" in link else ""
        rows.append({
            "ts": ts.astimezone(timezone.utc).isoformat(),
            "date": ts.astimezone(timezone.utc).date().isoformat(),
            "source": source_id,
            "title": title,
            "summary": desc,
            "url": link,
            "domain": domain,
            "tone": None,
            "lang": lang,
        })
    log.info(f"RSS {source_id}: {len(rows)} gold-relevant items")
    return pd.DataFrame(rows, columns=NEWS_SCHEMA)


# ============================================================
# Aggregator
# ============================================================

def fetch_all_realtime(
    cache_path: str = "data/external/news_realtime.parquet",
    keep_days: int = 90,
) -> pd.DataFrame:
    """Fetch every source, dedupe, append to cache, prune older than `keep_days`."""
    parts: list[pd.DataFrame] = []

    df_gd = fetch_gdelt_doc_api(timespan="1d", max_records=150)
    if not df_gd.empty:
        parts.append(df_gd)

    df_rd = fetch_reddit_json(limit=30)
    if not df_rd.empty:
        parts.append(df_rd)

    for sid, url, lang in RSS_SOURCES:
        df_r = fetch_rss(sid, url, lang)
        if not df_r.empty:
            parts.append(df_r)

    if not parts:
        log.error("No news fetched from any source")
        return pd.DataFrame(columns=NEWS_SCHEMA + ["fetched_at"])

    df_new = pd.concat(parts, ignore_index=True)
    df_new["fetched_at"] = datetime.now(timezone.utc).isoformat()

    cache_full = project_root() / cache_path
    if cache_full.exists():
        try:
            df_old = pd.read_parquet(cache_full)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            log.warning(f"Cache read failed: {e}")
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined = df_combined.drop_duplicates(subset=["url", "title"], keep="last")
    df_combined["ts"] = pd.to_datetime(df_combined["ts"], utc=True, errors="coerce")
    df_combined = df_combined.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=keep_days)
    df_combined = df_combined[df_combined["ts"] >= cutoff].copy()
    df_combined["ts"] = df_combined["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    write_parquet(df_combined, cache_full)
    log.info(f"Saved {cache_full.name}: {len(df_combined)} rows ({len(df_new)} new fetched)")
    log.info(f"By source: {df_combined['source'].value_counts().to_dict()}")
    return df_combined


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/external/news_realtime.parquet")
    p.add_argument("--keep-days", type=int, default=90)
    args = p.parse_args()
    df = fetch_all_realtime(args.cache, keep_days=args.keep_days)
    if df.empty:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
