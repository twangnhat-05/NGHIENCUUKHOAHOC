# -*- coding: utf-8 -*-
"""
NCKH - Dự báo giá vàng Việt Nam
Script thu thập dữ liệu: yfinance (OHLCV) + web scraping (giá vàng SJC)
Lưu dữ liệu vào các file CSV riêng biệt.
"""

import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from io import StringIO

# Sửa lỗi SSL curl (77) khi đường dẫn có ký tự Unicode (vd: "Trung Kiên")
# Copy cacert.pem đến thư mục không có ký tự đặc biệt
def _fix_ssl_cert_path():
    if sys.platform != "win32":
        return
    try:
        import certifi
        import ctypes
        cert_src = certifi.where()
        # Thử dùng đường dẫn 8.3 (short path) tránh lỗi Unicode
        buf = ctypes.create_unicode_buffer(512)
        if ctypes.windll.kernel32.GetShortPathNameW(cert_src, buf, 512):
            short_path = buf.value
            if short_path and os.path.exists(short_path):
                os.environ["SSL_CERT_FILE"] = short_path
                os.environ["REQUESTS_CA_BUNDLE"] = short_path
                return
        # Fallback: copy sang ProgramData (đường dẫn ASCII)
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

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup


# ============== Cấu hình ==============
DATA_DIR = "data"
START_YEAR = 2018

# Các mã yfinance: Gold futures, USD Index, VN-Index, Giá dầu WTI
# VN-Index: Yahoo đã ngừng ^VNINDEX/VNI.HM; dùng VNM (VanEck Vietnam ETF) làm proxy nếu cần
YFINANCE_TICKERS = {
    "GC=F": "Gold_Futures",
    "DX-Y.NYB": "USD_Index",
    "VNM": "VN_Index",  # VanEck Vietnam ETF - proxy cho VN-Index khi Yahoo không có chỉ số gốc
    "CL=F": "Oil_WTI",
}

# Nguồn scrape giá vàng SJC (webgia.com có URL theo ngày)
SJC_BASE_URL = "https://webgia.com/gia-vang/sjc"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def ensure_data_dir():
    """Tạo thư mục data nếu chưa có."""
    os.makedirs(DATA_DIR, exist_ok=True)


# VN-Index: ưu tiên vnstock (VNINDEX); fallback yfinance (VNM, ^VNINDEX, ...)
VN_INDEX_TICKERS = ["VNM", "^VNINDEX", "VNI.HM", "^VNI"]

# Các mã lãi suất FRED: FEDFUNDS (Fed), có thể thêm DFF, SOFR...
FRED_INTEREST_CODES = ["FEDFUNDS"]  # Lãi suất Fed Funds (Mỹ) - proxy lãi suất toàn cầu


# ============== 0. VN-Index bằng vnstock (VCI) ==============
def download_vnindex_vnstock(
    start_year: int = START_YEAR,
    end_date: str | None = None,
) -> bool:
    """
    Tải lịch sử VN-Index (chỉ số HOSE) từ vnstock, nguồn VCI.
    Symbol: VNINDEX. Lưu vào data/VN_Index_ohlcv.csv (cùng format OHLCV).
    Trả về True nếu thành công, False nếu lỗi (vd: chưa cài vnstock, API lỗi).
    """
    ensure_data_dir()
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = f"{start_year}-01-01"
    out_path = os.path.join(DATA_DIR, "VN_Index_ohlcv.csv")
    try:
        from vnstock import Quote

        quote = Quote(symbol="VNINDEX", source="VCI")
        df = quote.history(
            start=start_date,
            end=end_date,
            interval="1D",
            show_log=False,
        )
        if df is None or df.empty or len(df) < 10:
            return False
        # vnstock trả về: time, open, high, low, close, volume (có thể viết thường)
        rename_map = {
            "time": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Đã lưu (vnstock VCI): {out_path} ({len(df)} dòng)")
        return True
    except Exception as e:
        print(f"  vnstock VNINDEX: {e}")
        return False


# ============== 1. Tải dữ liệu OHLCV bằng yfinance ==============
def _download_single_ticker(
    ticker: str, name: str, start_date: str, end_date: str
) -> bool:
    """Tải một ticker, trả về True nếu thành công."""
    try:
        obj = yf.Ticker(ticker)
        df = obj.history(start=start_date, end=end_date, auto_adjust=True)
        if df.empty or len(df) < 10:
            return False
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        out_path = os.path.join(DATA_DIR, f"{name}_ohlcv.csv")
        df.to_csv(out_path, encoding="utf-8-sig")
        print(f"  Đã lưu: {out_path} ({len(df)} dòng)")
        return True
    except Exception as e:
        print(f"  Lỗi {ticker}: {e}")
        return False


def download_yfinance_ohlcv(start_year: int = START_YEAR) -> None:
    """
    Tải dữ liệu lịch sử OHLCV: Gold, USD Index, Giá dầu WTI từ yfinance;
    VN-Index ưu tiên từ vnstock (VNINDEX/VCI), fallback yfinance (VNM).
    """
    ensure_data_dir()
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    for ticker, name in YFINANCE_TICKERS.items():
        print(f"Đang tải {ticker} ({name}) từ {start_date} đến {end_date}...")
        success = False
        if name == "VN_Index":
            # Ưu tiên vnstock (chỉ số VNINDEX gốc từ VCI)
            if download_vnindex_vnstock(start_year=start_year, end_date=end_date):
                success = True
        if not success:
            tickers_to_try = VN_INDEX_TICKERS if name == "VN_Index" else [ticker]
            for t in tickers_to_try:
                if _download_single_ticker(t, name, start_date, end_date):
                    success = True
                    if name == "VN_Index" and t != ticker:
                        print(f"  (Dùng yfinance proxy: {t})")
                    break
                if name == "VN_Index":
                    print(f"  Thử ticker khác...")
        if not success:
            print(f"  Cảnh báo: Không tải được dữ liệu cho {name}")
    print("Hoàn tất tải dữ liệu yfinance.\n")


# ============== 2. Scraper giá vàng SJC (BeautifulSoup) ==============
def _parse_vn_number(text: str) -> float | None:
    """
    Chuyển chuỗi số kiểu VN thành số.
    Trang lịch sử webgia (triệu đồng/lượng): '74.000' = 74.0, '76.800 (-200)' = 76.8.
    """
    if not text or not isinstance(text, str):
        return None
    # Bỏ phần trong ngoặc như (-200), (+300)
    text = re.sub(r"\s*\([^)]*\)\s*", "", text).strip()
    # Thử parse trực tiếp (74.000 -> 74.0 trên trang triệu đồng)
    try:
        return float(text.replace(",", "."))
    except ValueError:
        pass
    # Fallback: bỏ dấu chấm phân cách hàng nghìn (ví dụ 17.860.000)
    text_clean = text.replace(".", "").replace(",", ".")
    try:
        return float(text_clean)
    except ValueError:
        return None


def scrape_sjc_day(date: datetime, session: requests.Session) -> dict | None:
    """
    Lấy giá vàng SJC một ngày từ webgia.com.
    URL: https://webgia.com/gia-vang/sjc/DD-MM-YYYY.html
    Trả về dict {date, mua_vao, ban_ra} hoặc None nếu không có dữ liệu.
    """
    url = f"{SJC_BASE_URL}/{date.strftime('%d-%m-%Y')}.html"
    try:
        r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"
    except Exception as e:
        print(f"    Request lỗi {url}: {e}")
        return None

    soup = BeautifulSoup(r.text, "lxml")
    # Bảng lịch sử: cột Lần, Thời gian cập nhật, Mua vào, Bán ra
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        if "mua vào" not in str(header) or "bán ra" not in str(header):
            continue
        # Lấy dòng dữ liệu (bỏ header); dùng giá lần cập nhật cuối trong ngày
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
            # Trang webgia lịch sử theo ngày: đơn vị triệu đồng/lượng
            return {
                "date": date.strftime("%Y-%m-%d"),
                "mua_vao": mua_vals[-1],
                "ban_ra": ban_vals[-1],
            }
    return None


def scrape_sjc_historical(
    start_year: int = START_YEAR,
    end_date: datetime | None = None,
    start_date: datetime | None = None,
    delay_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Scrape lịch sử giá vàng SJC từ webgia.com từ start_date (hoặc start_year) đến end_date.
    Mỗi ngày gửi 1 request; delay giữa các request để tránh quá tải server.
    """
    ensure_data_dir()
    end_date = end_date or datetime.now()
    start_d = start_date or datetime(start_year, 1, 1)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows = []
    current = start_d
    total_days = (end_date - start_d).days + 1
    done = 0

    while current <= end_date:
        # Bỏ cuối tuần (thị trường vàng SJC thường không cập nhật T7, CN)
        if current.weekday() < 5:
            row = scrape_sjc_day(current, session)
            if row:
                rows.append(row)
            time.sleep(delay_seconds)
        done += 1
        if done % 100 == 0:
            print(f"  Đã quét {done}/{total_days} ngày, thu thập {len(rows)} bản ghi...")
        current += timedelta(days=1)

    if not rows:
        print("Không thu thập được bản ghi giá SJC nào.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df


# ============== 3. Lãi suất: FRED (requests, không cần pandas_datareader) ==============
def download_interest_rate_fred(
    start_year: int = START_YEAR,
    codes: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Lấy lãi suất từ FRED qua CSV download (không cần API key).
    Mặc định: FEDFUNDS (lãi suất Fed Funds - Mỹ).
    """
    codes = codes or FRED_INTEREST_CODES
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={codes[0]}"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        text = r.text.lstrip("\ufeff")  # BOM
        df = pd.read_csv(StringIO(text))
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        # FRED CSV: cột đầu thường là DATE, cột 2 là giá trị
        date_col = None
        for c in ["DATE", "Date", "observation_date", "date"]:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        other_cols = [c for c in df.columns if c != "Date"]
        rate_col = codes[0] if codes[0] in df.columns else (other_cols[0] if other_cols else None)
        if rate_col is None:
            raise ValueError("Không tìm thấy cột lãi suất trong dữ liệu FRED")
        df = df.rename(columns={rate_col: "Interest_Rate"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.dropna(subset=["Date", "Interest_Rate"])
        df = df[(df["Date"] >= pd.to_datetime(start_date).date()) & (df["Date"] <= pd.to_datetime(end_date).date())]
        out_path = os.path.join(DATA_DIR, "interest_rate.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Đã lưu: {out_path} ({len(df)} dòng)")
        return df
    except Exception as e:
        print(f"  Lỗi lấy dữ liệu FRED: {e}")
        print("  Có thể tạo data/interest_rate_vn.csv (đổi tên từ interest_rate_vn.csv.example)")
        return None


def load_interest_rate_csv(
    path: str | None = None,
    date_col: str = "Date",
    rate_col: str = "rate",
) -> pd.DataFrame | None:
    """
    Load lãi suất từ file CSV (vd: lãi suất tiền gửi VN từ SBV).
    CSV cần có: cột ngày (YYYY-MM-DD) và cột lãi suất (%).
    Đặt file tại data/interest_rate_vn.csv hoặc truyền path.
    Ví dụ CSV: Date,rate hoặc ngay,lai_suat
    """
    default_path = os.path.join(DATA_DIR, "interest_rate_vn.csv")
    path = path or default_path
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    date_found = date_col in df.columns
    if not date_found:
        for c in ["date", "ngay", "Ngày", "DATE"]:
            if c in df.columns:
                df = df.rename(columns={c: "Date"})
                date_found = True
                break
    rate_found = rate_col in df.columns
    if not rate_found:
        for c in ["rate", "lai_suat", "Lãi suất", "interest_rate", "value", "FEDFUNDS"]:
            if c in df.columns:
                df = df.rename(columns={c: "Interest_Rate"})
                rate_found = True
                break
    if not date_found or not rate_found:
        print(f"  CSV cần cột Date và rate. Cột hiện có: {list(df.columns)}")
        return None
    if "Interest_Rate" not in df.columns and rate_col != "Interest_Rate":
        df = df.rename(columns={rate_col: "Interest_Rate"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df[["Date", "Interest_Rate"]].dropna()
    out_path = os.path.join(DATA_DIR, "interest_rate.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def download_interest_rate(start_year: int = START_YEAR) -> None:
    """
    Tải lãi suất: ưu tiên FRED (FEDFUNDS). Nếu có file interest_rate_vn.csv
    thì dùng thay thế (lãi suất VN phù hợp hơn cho thị trường vàng VN).
    """
    ensure_data_dir()
    vn_path = os.path.join(DATA_DIR, "interest_rate_vn.csv")
    if os.path.exists(vn_path):
        print("Đang load lãi suất từ CSV (interest_rate_vn.csv)...")
        load_interest_rate_csv(path=vn_path)
    else:
        print("Đang tải lãi suất từ FRED (FEDFUNDS)...")
        download_interest_rate_fred(start_year=start_year)


def download_sjc_to_csv(
    start_year: int = START_YEAR,
    delay_seconds: float = 0.6,
) -> None:
    """
    Scrape giá vàng SJC từ webgia.com (nguồn tin cậy, dữ liệu từ SJC)
    từ start_year đến nay và lưu vào CSV.
    """
    print("Đang scrape giá vàng SJC từ webgia.com (BeautifulSoup)...")
    print("(Chỉ lấy ngày giao dịch, mỗi request cách nhau vài giây để tôn trọng server.)")
    df = scrape_sjc_historical(
        start_year=start_year,
        delay_seconds=delay_seconds,
    )
    if df.empty:
        return
    out_path = os.path.join(DATA_DIR, "SJC_gold_historical.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {out_path} ({len(df)} dòng)\n")


# ============== Main ==============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu NCKH dự báo giá vàng")
    parser.add_argument(
        "--sjc-only",
        action="store_true",
        help="Chỉ chạy scraper giá vàng SJC (bỏ qua yfinance)",
    )
    parser.add_argument(
        "--yfinance-only",
        action="store_true",
        help="Chỉ tải dữ liệu yfinance (bỏ qua SJC)",
    )
    parser.add_argument(
        "--sjc-days",
        type=int,
        default=None,
        help="Giới hạn số ngày scrape SJC (mặc định: từ 2018 đến nay). VD: 30 để test nhanh.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"Năm bắt đầu (mặc định: {START_YEAR})",
    )
    parser.add_argument(
        "--interest-rate",
        action="store_true",
        help="Tải thêm lãi suất từ FRED (hoặc load CSV interest_rate_vn.csv nếu có)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NCKH - Thu thập dữ liệu dự báo giá vàng Việt Nam")
    print("=" * 60)

    if not args.sjc_only:
        download_yfinance_ohlcv(start_year=args.start_year)
        if args.interest_rate:
            download_interest_rate(start_year=args.start_year)

    if not args.yfinance_only:
        if args.sjc_days:
            end_d = datetime.now()
            start_d = end_d - timedelta(days=args.sjc_days)
            ensure_data_dir()
            print(f"Scrape SJC {args.sjc_days} ngày gần nhất...")
            df = scrape_sjc_historical(
                end_date=end_d,
                start_date=start_d,
                delay_seconds=0.6,
            )
            if not df.empty:
                out_path = os.path.join(DATA_DIR, "SJC_gold_historical.csv")
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"Đã lưu: {out_path} ({len(df)} dòng)")
        else:
            download_sjc_to_csv(start_year=args.start_year, delay_seconds=0.6)

    print("Kết thúc. Các file CSV nằm trong thư mục:", os.path.abspath(DATA_DIR))
