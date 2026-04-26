import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

# ============== Cấu hình ==============
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FINAL_CLEANED_CSV = OUTPUT_DIR / "final_data_cleaned.csv"
MODEL_READY_CSV = OUTPUT_DIR / "model_ready_data.csv"

# Tham số kỹ thuật
RSI_PERIOD = 14
SMA_SHORT = 10
SMA_LONG = 30


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI (Relative Strength Index) theo chu kỳ period.
    RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    # Tránh chia cho 0: nếu avg_loss = 0 thì RSI = 100
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100.0)
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng: Lags (1,3,5) cho SJC và Vàng thế giới,
    Returns, SMA(10), SMA(30), RSI(14) cho SJC, Target (giá SJC ngày t+1).
    Sau đó xóa dòng NaN do Lag/SMA/RSI.
    """
    out = df.copy()
    if "SJC_ban_ra" not in out.columns:
        return out

    # ----- Lags: trễ 1, 3, 5 ngày cho SJC và Vàng thế giới -----
    out["SJC_ban_ra_lag1"] = out["SJC_ban_ra"].shift(1)
    out["SJC_ban_ra_lag3"] = out["SJC_ban_ra"].shift(3)
    out["SJC_ban_ra_lag5"] = out["SJC_ban_ra"].shift(5)
    if "Gold_Close" in out.columns:
        out["Gold_Close_lag1"] = out["Gold_Close"].shift(1)
        out["Gold_Close_lag3"] = out["Gold_Close"].shift(3)
        out["Gold_Close_lag5"] = out["Gold_Close"].shift(5)

    # ----- Returns: % thay đổi hàng ngày -----
    out["SJC_return"] = out["SJC_ban_ra"].pct_change()
    if "Gold_Close" in out.columns:
        out["Gold_return"] = out["Gold_Close"].pct_change()

    # ----- Technical Indicators cho giá SJC -----
    out["SJC_ban_ra_sma10"] = out["SJC_ban_ra"].rolling(SMA_SHORT).mean()
    out["SJC_ban_ra_sma30"] = out["SJC_ban_ra"].rolling(SMA_LONG).mean()
    out["SJC_ban_ra_rsi14"] = compute_rsi(out["SJC_ban_ra"], period=RSI_PERIOD)

    # ----- Target: giá SJC_ban_ra ngày t+1 (dùng cho dự báo) -----
    out["target"] = out["SJC_ban_ra"].shift(-1)

    # Xóa dòng NaN phát sinh do Lag/SMA/RSI (và target ở dòng cuối)
    out = out.dropna()
    return out


def main():
    if not FINAL_CLEANED_CSV.exists():
        print(f"Không tìm thấy {FINAL_CLEANED_CSV}. Chạy eda_merge_analysis.py trước.")
        return

    df = pd.read_csv(FINAL_CLEANED_CSV, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    print(f"Đã đọc: {FINAL_CLEANED_CSV} ({len(df)} dòng)")

    model_ready = build_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(MODEL_READY_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {MODEL_READY_CSV} ({len(model_ready)} dòng)")
    print(f"Phạm vi: {model_ready['Date'].min()} -> {model_ready['Date'].max()}")
    print("\n--- 5 dòng đầu model_ready_data ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(model_ready.head().to_string())
    print("---\nHoàn tất.")


if __name__ == "__main__":
    main()
