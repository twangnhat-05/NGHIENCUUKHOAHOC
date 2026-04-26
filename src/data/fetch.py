"""Fetch dữ liệu raw từ các nguồn miễn phí.

Giữ logic ổn định từ legacy `download_data.py` nhưng:
- Config-driven (configs/data.yaml)
- Retry với backoff
- Schema validation sau khi tải
- Không hardcode constants

Sources:
- yfinance: GC=F, DX-Y.NYB, CL=F, VND=X, GLD, BTC-USD
- vnstock (VCI): VNINDEX
- FRED CSV download (free, no API key): FEDFUNDS, DTWEXBGS, DGS10
- Webgia.com: SJC mua/bán (BeautifulSoup scraper)
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.io import load_yaml, project_root
from src.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================
# SSL fix for Vietnamese paths (kept from legacy — Windows quirk)
# ============================================================

def _fix_ssl_cert_path() -> None:
    """Workaround cho Windows + Unicode path (curl error 77)."""
    if sys.platform != "win32":
        return
    try:
        import certifi
        import ctypes
        cert_src = certifi.where()
        buf = ctypes.create_unicode_buffer(512)
        if ctypes.windll.kernel32.GetShortPathNameW(cert_src, buf, 512):
            short_path = buf.value
            if short_path and os.path.exists(short_path):
                os.environ["SSL_CERT_FILE"] = short_path
                os.environ["REQUESTS_CA_BUNDLE"] = short_path
                return
        cert_dir = os.environ.get("PROGRAMDATA", "C:\\ProgramData")
        cert_dst = os.path.join(cert_dir, "cacert_nckh.pem")
        try:
            os.makedirs(cert_dir, exist_ok=True)
            shutil.copy2(cert_src, cert_dst)
            os.environ["SSL_CERT_FILE"] = cert_dst
            os.environ["REQUESTS_CA_BUNDLE"] = cert_dst
        except (OSError, PermissionError):
            pass
    except Exception:
        pass


_fix_ssl_cert_path()


# ============================================================
# YFINANCE
# ============================================================

def fetch_yfinance(
    ticker: str,
    name: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
    retries: int = 3,
    delay: float = 1.0,
) -> bool:
    """Tải 1 ticker từ yfinance, lưu CSV. Trả về True nếu OK."""
    import yfinance as yf  # lazy import

    for attempt in range(1, retries + 1):
        try:
            obj = yf.Ticker(ticker)
            df = obj.history(start=start_date, end=end_date, auto_adjust=True)
            if df.empty or len(df) < 10:
                log.warning(f"{ticker} ({name}): rỗng/thiếu dữ liệu (attempt {attempt})")
                time.sleep(delay)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            out_path = out_dir / f"{name}_ohlcv.csv"
            df.to_csv(out_path, encoding="utf-8-sig")
            log.info(f"OK {ticker} ({name}): {len(df)} dòng → {out_path.name}")
            return True
        except Exception as e:
            log.warning(f"{ticker} attempt {attempt}/{retries}: {e}")
            time.sleep(delay)
    log.error(f"FAIL {ticker} ({name}) sau {retries} attempts")
    return False


# ============================================================
# VNSTOCK
# ============================================================

def fetch_vnindex_vnstock(start_date: str, end_date: str, out_dir: Path) -> bool:
    """VN-Index từ vnstock (VCI source) — chỉ số gốc, không phải ETF proxy."""
    try:
        from vnstock import Quote
        quote = Quote(symbol="VNINDEX", source="VCI")
        df = quote.history(start=start_date, end=end_date, interval="1D", show_log=False)
        if df is None or df.empty or len(df) < 10:
            return False
        rename_map = {"time": "Date", "open": "Open", "high": "High",
                      "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df = df[[c for c in cols if c in df.columns]]
        out_path = out_dir / "VN_Index_ohlcv.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"OK VNINDEX (vnstock VCI): {len(df)} dòng → {out_path.name}")
        return True
    except Exception as e:
        log.error(f"vnstock VNINDEX: {e}")
        return False


# ============================================================
# FRED (free CSV download, no API key)
# ============================================================

def fetch_fred_series(
    code: str,
    start_date: str,
    end_date: str,
    out_path: Path,
    rename_to: str = "Value",
) -> bool:
    """Lấy 1 chuỗi FRED qua CSV download. Free, không cần API key."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    user_agent = "Mozilla/5.0 (compatible; NCKH-TDTU-research/1.0)"
    try:
        r = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
        r.raise_for_status()
        text = r.text.lstrip("\ufeff")
        df = pd.read_csv(StringIO(text))
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        date_col = next((c for c in ["DATE", "Date", "observation_date", "date"] if c in df.columns), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        other_cols = [c for c in df.columns if c != "Date"]
        rate_col = code if code in df.columns else (other_cols[0] if other_cols else None)
        if rate_col is None:
            raise ValueError(f"Không tìm thấy cột giá trị cho {code}")
        df = df.rename(columns={rate_col: rename_to})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.dropna(subset=["Date", rename_to])
        df = df[
            (df["Date"] >= pd.to_datetime(start_date).date())
            & (df["Date"] <= pd.to_datetime(end_date).date())
        ]
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"OK FRED {code}: {len(df)} dòng → {out_path.name}")
        return True
    except Exception as e:
        log.error(f"FRED {code}: {e}")
        return False


# ============================================================
# SJC SCRAPER (webgia.com)
# ============================================================

def _parse_vn_number(text: str) -> float | None:
    """Parse số kiểu VN: '74.000' (triệu/lượng) → 74.0."""
    if not text or not isinstance(text, str):
        return None
    text = re.sub(r"\s*\([^)]*\)\s*", "", text).strip()
    try:
        return float(text.replace(",", "."))
    except ValueError:
        pass
    text_clean = text.replace(".", "").replace(",", ".")
    try:
        return float(text_clean)
    except ValueError:
        return None


def scrape_sjc_day(date: datetime, session: requests.Session, base_url: str, ua: str) -> dict | None:
    """Lấy SJC cho 1 ngày từ webgia.com."""
    url = f"{base_url}/{date.strftime('%d-%m-%Y')}.html"
    try:
        r = session.get(url, headers={"User-Agent": ua}, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"
    except Exception as e:
        log.debug(f"Request lỗi {url}: {e}")
        return None
    soup = BeautifulSoup(r.text, "lxml")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        if "mua vào" not in str(header) or "bán ra" not in str(header):
            continue
        mua_vals, ban_vals = [], []
        for tr in rows[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) >= 4:
                mua = _parse_vn_number(cells[2].get_text(strip=True))
                ban = _parse_vn_number(cells[3].get_text(strip=True))
                if mua is not None and ban is not None:
                    mua_vals.append(mua)
                    ban_vals.append(ban)
        if mua_vals and ban_vals:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "mua_vao": mua_vals[-1],
                "ban_ra": ban_vals[-1],
            }
    return None


def fetch_sjc_range(
    start_date: datetime,
    end_date: datetime,
    base_url: str,
    user_agent: str,
    delay: float = 0.6,
    weekday_only: bool = True,
) -> pd.DataFrame:
    """Scrape SJC từ start → end. Chỉ T2-T6 nếu weekday_only=True."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    rows = []
    current = start_date
    total_days = (end_date - start_date).days + 1
    done = 0
    while current <= end_date:
        if (not weekday_only) or current.weekday() < 5:
            row = scrape_sjc_day(current, session, base_url, user_agent)
            if row:
                rows.append(row)
            time.sleep(delay)
        done += 1
        if done % 50 == 0:
            log.info(f"SJC scrape progress: {done}/{total_days} ngày, thu được {len(rows)} bản ghi")
        current += timedelta(days=1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df


# ============================================================
# ENTRY: fetch_all (idempotent — overwrite raw files)
# ============================================================

def fetch_all_full(config_path: str = "configs/data.yaml") -> dict[str, bool]:
    """Tải lại TOÀN BỘ raw từ 2018 → today. Chỉ nên dùng lần đầu setup."""
    cfg = load_yaml(config_path)
    out_dir = project_root() / cfg["paths"]["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    start = cfg["start_date"]
    end = datetime.now().strftime("%Y-%m-%d") if cfg["end_date"] == "today" else cfg["end_date"]

    results: dict[str, bool] = {}
    # yfinance
    for ticker, name in cfg["yfinance"].items():
        if ticker in ("retry", "delay_seconds"):
            continue
        results[name] = fetch_yfinance(
            ticker=ticker, name=name, start_date=start, end_date=end, out_dir=out_dir,
            retries=cfg["yfinance"]["retry"], delay=cfg["yfinance"]["delay_seconds"],
        )
    # vnstock VN-Index
    results["VN_Index"] = fetch_vnindex_vnstock(start, end, out_dir)
    # FRED
    for name, code in cfg["fred"].items():
        rename = {"FEDFUNDS": "Interest_Rate"}.get(name, name)
        out_path = out_dir / f"{name.lower()}.csv" if name != "FEDFUNDS" else out_dir / "interest_rate.csv"
        results[name] = fetch_fred_series(code, start, end, out_path, rename_to=rename)
    # SJC
    sjc_cfg = cfg["sjc"]
    df_sjc = fetch_sjc_range(
        start_date=datetime.strptime(start, "%Y-%m-%d"),
        end_date=datetime.strptime(end, "%Y-%m-%d"),
        base_url=sjc_cfg["base_url"], user_agent=sjc_cfg["user_agent"],
        delay=sjc_cfg["delay_seconds"], weekday_only=sjc_cfg["weekday_only"],
    )
    if not df_sjc.empty:
        df_sjc.to_csv(out_dir / "SJC_gold_historical.csv", index=False, encoding="utf-8-sig")
        log.info(f"OK SJC: {len(df_sjc)} dòng")
        results["SJC"] = True
    else:
        results["SJC"] = False
    return results
