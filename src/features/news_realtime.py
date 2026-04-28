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
try:
    # defusedxml hardens against XXE / billion-laughs in untrusted RSS feeds
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover
    from xml.etree import ElementTree as ET

import pandas as pd
import requests

from src.utils.io import project_root, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept": "*/*"}
TIMEOUT = 60       # generous — GH runners and GDELT are sometimes slow
MAX_RETRIES = 3


def _http_get_with_retry(url: str, headers: dict | None = None,
                         timeout: int = TIMEOUT) -> requests.Response | None:
    """GET with simple exponential backoff. Returns None on terminal failure."""
    h = headers or HEADERS
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=h, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    log.warning(f"GET {url} failed after {MAX_RETRIES} attempts: {last_err}")
    return None

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
    r = _http_get_with_retry(url)
    if r is None:
        return pd.DataFrame(columns=NEWS_SCHEMA)
    try:
        data = r.json()
    except Exception as e:
        log.warning(f"GDELT JSON parse failed: {e}")
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
            # ArtList does not return per-article tone — use timeline API below
            "tone": None,
            "lang": (art.get("language") or "unknown")[:2].lower(),
        })
    log.info(f"GDELT: {len(rows)} articles (timespan={timespan})")
    return pd.DataFrame(rows, columns=NEWS_SCHEMA)


def fetch_gdelt_timeline_tone(
    query: str = "(gold OR \"gold price\" OR bullion OR \"precious metals\")",
    timespan: str = "1d",
) -> pd.DataFrame:
    """Fetch the GDELT TimelineTone series for a query.

    Returns a DataFrame with `ts` (UTC, 15-min buckets) and `tone` (mean tone
    -100..+100 across all articles in that bucket). Empty DF on failure.
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "TimelineTone",
        "format": "json",
        "timespan": timespan,
    }
    url = base + "?" + "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
    r = _http_get_with_retry(url)
    if r is None:
        return pd.DataFrame(columns=["ts", "tone"])
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame(columns=["ts", "tone"])

    rows = []
    for series in data.get("timeline", []):
        for pt in series.get("data", []):
            ts_raw = pt.get("date", "")
            try:
                ts = datetime.strptime(ts_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            try:
                tone = float(pt.get("value", 0.0))
            except (TypeError, ValueError):
                continue
            rows.append({"ts": ts.isoformat(), "tone": tone})
    log.info(f"GDELT TimelineTone: {len(rows)} buckets (timespan={timespan})")
    return pd.DataFrame(rows, columns=["ts", "tone"])


# ============================================================
# REDDIT JSON — public feeds, no auth
# ============================================================

def fetch_reddit_json(subreddits: list[str] | None = None, limit: int = 50) -> pd.DataFrame:
    """Fetch new posts from public subreddits via JSON endpoint."""
    subreddits = subreddits or ["Gold", "wallstreetbets", "Forex", "preciousmetals"]
    rows = []
    # Reddit aggressively rate-limits / blocks data-centre IPs (GitHub runners
    # routinely get 403). Fall back to old.reddit.com which is more lenient
    # for unauthenticated reads, and skip silently if blocked.
    for sub in subreddits:
        url = f"https://old.reddit.com/r/{sub}/new.json?limit={limit}"
        r = _http_get_with_retry(url)
        if r is None:
            continue
        try:
            posts = r.json().get("data", {}).get("children", [])
        except Exception:
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
    ("investing", "https://www.investing.com/rss/news_25.rss", "en"),  # commodities
    ("yahoo_gold", "https://finance.yahoo.com/news/rssindex", "en"),
    ("cafef_kd", "https://cafef.vn/thi-truong-chung-khoan.rss", "vi"),
    ("cafef_taichinh", "https://cafef.vn/tai-chinh-ngan-hang.rss", "vi"),
    ("vnexpress_kd", "https://vnexpress.net/rss/kinh-doanh.rss", "vi"),
    ("tuoitre_kt", "https://tuoitre.vn/rss/kinh-te.rss", "vi"),
    ("thanhnien_kt", "https://thanhnien.vn/rss/kinh-te.rss", "vi"),
    ("vneconomy_taichinh", "https://vneconomy.vn/tai-chinh.rss", "vi"),
]


def fetch_google_news_rss(
    query: str,
    lang: str = "en",
    region: str = "US",
    source_id: str | None = None,
) -> pd.DataFrame:
    """Google News RSS — free, no API key, supports any query + locale."""
    from urllib.parse import quote_plus
    src = source_id or f"gnews_{lang}"
    url = (
        f"https://news.google.com/rss/search?q={quote_plus(query)}"
        f"&hl={lang}-{region}&gl={region}&ceid={region}:{lang}"
    )
    r = _http_get_with_retry(url)
    if r is None:
        return pd.DataFrame(columns=NEWS_SCHEMA)
    try:
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning(f"Google News RSS '{query}' parse failed: {e}")
        return pd.DataFrame(columns=NEWS_SCHEMA)

    rows = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        pub = item.findtext("pubDate") or ""
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "")[:500]
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
        domain = link.split("/")[2] if "://" in link else "news.google.com"
        rows.append({
            "ts": ts.astimezone(timezone.utc).isoformat(),
            "date": ts.astimezone(timezone.utc).date().isoformat(),
            "source": src,
            "title": title,
            "summary": desc,
            "url": link,
            "domain": domain,
            "tone": None,
            "lang": lang,
        })
    log.info(f"Google News '{query}' [{lang}/{region}]: {len(rows)} items")
    return pd.DataFrame(rows, columns=NEWS_SCHEMA)


def fetch_rss(source_id: str, url: str, lang: str,
              gold_filter: bool = True) -> pd.DataFrame:
    """Generic RSS fetcher. If `gold_filter=True`, keep only gold-related items
    (broad keyword match) to reduce noise from non-gold feeds like cafef-broad."""
    rows = []
    keywords = [
        # English
        "gold", "bullion", "precious metal", "fed", "usd", "inflation",
        "treasury", "interest rate", "rate cut", "rate hike",
        # Vietnamese
        "vàng", "sjc", "kim loại quý", "lãi suất", "ngân hàng nhà nước",
        "giá vàng", "đô la", "tỷ giá", "tài chính", "kinh tế vĩ mô",
        "ngân hàng trung ương", "lạm phát", "vàng miếng",
    ]
    r = _http_get_with_retry(url)
    if r is None:
        return pd.DataFrame(columns=NEWS_SCHEMA)
    try:
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning(f"RSS {source_id} parse failed: {e}")
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

DEFAULT_TONE_CACHE = "data/external/gdelt_tone_timeline.parquet"


def fetch_all_realtime(
    cache_path: str = "data/external/news_realtime.parquet",
    tone_cache_path: str = DEFAULT_TONE_CACHE,
    keep_days: int = 90,
) -> pd.DataFrame:
    """Fetch every source, dedupe, append to cache, prune older than `keep_days`.

    Also refreshes the GDELT TimelineTone cache (small parquet with 15-min
    bucket tones) so the dashboard can read aggregate tone without making an
    API call at request time.
    """
    parts: list[pd.DataFrame] = []

    # Reduced GDELT cap so VN + other sources are visible in the dashboard mix
    df_gd = fetch_gdelt_doc_api(timespan="1d", max_records=80)
    if not df_gd.empty:
        parts.append(df_gd)

    # Google News RSS — language-specific gold queries (free, no key)
    for q, lang, region, src_id in [
        ("\"gold price\" OR bullion OR \"FED rate\"", "en", "US", "gnews_en"),
        ("giá vàng SJC OR \"vàng miếng\"", "vi", "VN", "gnews_vi_sjc"),
        ("ngân hàng nhà nước OR \"lãi suất\"", "vi", "VN", "gnews_vi_macro"),
    ]:
        df_g = fetch_google_news_rss(q, lang=lang, region=region, source_id=src_id)
        if not df_g.empty:
            parts.append(df_g)

    # Tone timeline (separate from article list — different GDELT mode)
    df_tone = fetch_gdelt_timeline_tone(timespan="1d")
    if not df_tone.empty:
        tone_full = project_root() / tone_cache_path
        try:
            if tone_full.exists():
                old = pd.read_parquet(tone_full)
                df_tone = pd.concat([old, df_tone], ignore_index=True)
            df_tone = df_tone.drop_duplicates(subset=["ts"], keep="last")
            df_tone["ts_dt"] = pd.to_datetime(df_tone["ts"], utc=True, errors="coerce")
            df_tone = df_tone.dropna(subset=["ts_dt"]).sort_values("ts_dt")
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=keep_days)
            df_tone = df_tone[df_tone["ts_dt"] >= cutoff]
            df_tone = df_tone.drop(columns=["ts_dt"])
            write_parquet(df_tone, tone_full)
            log.info(f"Saved {tone_full.name}: {len(df_tone)} tone buckets")
        except Exception as e:
            log.warning(f"Tone cache update failed: {e}")

    df_rd = fetch_reddit_json(limit=30)
    if not df_rd.empty:
        parts.append(df_rd)

    for sid, url, lang in RSS_SOURCES:
        df_r = fetch_rss(sid, url, lang)
        if not df_r.empty:
            parts.append(df_r)

    cache_full = project_root() / cache_path
    df_old = pd.DataFrame(columns=NEWS_SCHEMA + ["fetched_at"])
    if cache_full.exists():
        try:
            df_old = pd.read_parquet(cache_full)
        except Exception as e:
            log.warning(f"Cache read failed: {e}")

    if not parts:
        # Graceful degradation — every source failed (network / rate limit).
        # Keep cache as-is so dashboard stays functional.
        log.warning("No news fetched from any source this run; keeping cache")
        return df_old

    df_new = pd.concat(parts, ignore_index=True)
    df_new["fetched_at"] = datetime.now(timezone.utc).isoformat()
    df_combined = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new

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
