"""News headlines fetcher cho gold market — multi-source.

Sources (free tier):
1. yfinance.Ticker.news — gives ~10 latest news items for ticker, has title/summary/pubDate
2. Google News RSS — free, query "gold price" → RSS feed
3. CafeF RSS (optional, VN news) — free RSS feed

Output: data/external/news_headlines.parquet với columns:
   date, ticker, source, title, summary, url
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from src.utils.io import project_root, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; NCKH-TDTU-research/1.0)"


# ============================================================
# YFINANCE NEWS (English, gold-related tickers)
# ============================================================

def fetch_yfinance_news(tickers: list[str] = None) -> pd.DataFrame:
    """Fetch news từ yfinance cho list tickers.

    yfinance trả về ~10 news mới nhất. Để có history, cần chạy định kỳ + cache.
    """
    import yfinance as yf
    tickers = tickers or ["GLD", "GC=F", "GDX", "IAU", "SLV"]
    rows = []
    for tkr in tickers:
        try:
            news_list = yf.Ticker(tkr).news
            for item in news_list:
                content = item.get("content", item)
                title = content.get("title", "")
                summary = content.get("summary", content.get("description", ""))
                pub = content.get("pubDate", content.get("displayTime", ""))
                if not title or not pub:
                    continue
                rows.append({
                    "date": pub[:10],   # YYYY-MM-DD
                    "ticker": tkr,
                    "source": "yfinance",
                    "title": title,
                    "summary": summary[:500] if summary else "",
                    "url": content.get("clickThroughUrl", {}).get("url", ""),
                })
            log.info(f"yfinance {tkr}: fetched {len(news_list)} news")
        except Exception as e:
            log.warning(f"yfinance {tkr}: {e}")
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "source", "title", "summary", "url"])
    df = pd.DataFrame(rows)
    return df


# ============================================================
# GOOGLE NEWS RSS (free, no auth)
# ============================================================

def fetch_google_news_rss(query: str = "gold price", lang: str = "en", region: str = "US") -> pd.DataFrame:
    """Fetch Google News RSS cho query."""
    base = "https://news.google.com/rss/search"
    url = f"{base}?q={query.replace(' ', '+')}&hl={lang}-{region}&gl={region}&ceid={region}:{lang}"
    rows = []
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            pub = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            desc = item.findtext("description", "")[:500]
            # Parse pubDate "Sun, 27 Apr 2026 01:30:00 GMT" → date
            try:
                pub_dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
                date_str = pub_dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = ""
            if not title or not date_str:
                continue
            rows.append({
                "date": date_str, "ticker": "GLOBAL_GOLD", "source": "google_news",
                "title": title, "summary": desc, "url": link,
            })
        log.info(f"Google News RSS '{query}': fetched {len(rows)} items")
    except Exception as e:
        log.warning(f"Google News RSS failed: {e}")
    return pd.DataFrame(rows)


# ============================================================
# CAFEF RSS (Vietnamese gold news)
# ============================================================

def fetch_cafef_gold_rss() -> pd.DataFrame:
    """CafeF RSS cho thị trường vàng (Vietnamese news)."""
    url = "https://cafef.vn/thi-truong-chung-khoan.rss"  # broader market RSS
    rows = []
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            # Filter chỉ tin liên quan đến vàng
            keywords = ["vàng", "gold", "SJC", "kim loại quý"]
            if not any(k.lower() in title.lower() for k in keywords):
                continue
            pub = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            desc = item.findtext("description", "")[:500]
            try:
                pub_dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
                date_str = pub_dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = ""
            if not title or not date_str:
                continue
            rows.append({
                "date": date_str, "ticker": "VN_GOLD", "source": "cafef",
                "title": title, "summary": desc, "url": link,
            })
        log.info(f"CafeF RSS: fetched {len(rows)} gold-related items")
    except Exception as e:
        log.warning(f"CafeF RSS failed: {e}")
    return pd.DataFrame(rows)


# ============================================================
# AGGREGATE + CACHE (append-only)
# ============================================================

def fetch_all_news(cache_path: str = "data/external/news_headlines.parquet") -> pd.DataFrame:
    """Fetch tất cả nguồn, merge + dedupe + append vào cache."""
    parts = []
    df_yf = fetch_yfinance_news()
    if not df_yf.empty:
        parts.append(df_yf)
    df_gn = fetch_google_news_rss(query="gold price")
    if not df_gn.empty:
        parts.append(df_gn)
    df_gn_vn = fetch_google_news_rss(query="giá vàng SJC", lang="vi", region="VN")
    if not df_gn_vn.empty:
        parts.append(df_gn_vn)
    df_cf = fetch_cafef_gold_rss()
    if not df_cf.empty:
        parts.append(df_cf)

    if not parts:
        log.error("Không lấy được tin nào từ mọi nguồn")
        return pd.DataFrame()

    df_new = pd.concat(parts, ignore_index=True)
    df_new["fetched_at"] = datetime.now().isoformat()

    # Append to cache (dedupe by title + date + ticker)
    cache_full = project_root() / cache_path
    if cache_full.exists():
        try:
            df_old = pd.read_parquet(cache_full)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            log.warning(f"Read cache failed: {e}; overwrite")
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined = df_combined.drop_duplicates(subset=["date", "title", "ticker"], keep="last")
    df_combined = df_combined.sort_values("date").reset_index(drop=True)
    write_parquet(df_combined, cache_full)
    log.info(f"Saved {cache_full.name}: {len(df_combined)} total ({len(df_new)} new)")
    return df_combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch gold news from multi sources")
    parser.add_argument("--cache", default="data/external/news_headlines.parquet")
    args = parser.parse_args()
    df = fetch_all_news(args.cache)
    if df.empty:
        return 1
    log.info(f"By source: {df['source'].value_counts().to_dict()}")
    log.info(f"Date range: {df['date'].min()} → {df['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
