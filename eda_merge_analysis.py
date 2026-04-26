import os
import sys
import warnings
from pathlib import Path

# Tránh UnicodeEncodeError khi in tiếng Việt trên Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ============== Cấu hình ==============
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MERGED_CSV = OUTPUT_DIR / "merged_data.csv"
FINAL_CLEANED_CSV = OUTPUT_DIR / "final_data_cleaned.csv"

# Phạm vi nghiên cứu: từ năm 2018
START_DATE = "2018-01-01"

# Cột giá dùng cho merge và phân tích
SJC_PRICE_COL = "SJC_ban_ra"  # Giá bán ra SJC (triệu VND/lượng)
GOLD_CLOSE = "Gold_Close"
USD_CLOSE = "USD_Index_Close"


def ensure_output_dir():
    """Tạo thư mục output nếu chưa có."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============== 1. Đọc và chuẩn hóa Date ==============
def _normalize_date(series: pd.Series) -> pd.Series:
    """
    Chuẩn hóa cột ngày: bỏ timezone, chỉ lấy date (YYYY-MM-DD).
    Xử lý lệch múi giờ: chuyển về date thuần để merge theo ngày giao dịch.
    """
    s = pd.to_datetime(series, utc=True)
    return pd.to_datetime(s.dt.date)


def load_sjc(path: Path) -> pd.DataFrame:
    """Đọc SJC: date, mua_vao, ban_ra -> Date, SJC_mua_vao, SJC_ban_ra."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"date": "Date"})
    df["Date"] = _normalize_date(pd.to_datetime(df["Date"], utc=True))
    df = df.rename(columns={"mua_vao": "SJC_mua_vao", "ban_ra": "SJC_ban_ra"})
    return df[["Date", "SJC_mua_vao", "SJC_ban_ra"]]


def load_ohlcv(path: Path, close_suffix: str) -> pd.DataFrame:
    """Đọc OHLCV từ yfinance/vnstock: chuẩn hóa Date (bỏ timezone), lấy Close."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = _normalize_date(pd.to_datetime(df["Date"], utc=True))
    close_col = [c for c in df.columns if c == "Close"][0]
    df = df[["Date", close_col]].rename(columns={close_col: close_suffix})
    return df


# ============== 2. Merge theo Date + Forward Fill ==============
def load_all_csvs() -> dict[str, pd.DataFrame]:
    """Đọc tất cả CSV có trong data/ (theo tên file quen thuộc)."""
    data = {}
    if (DATA_DIR / "SJC_gold_historical.csv").exists():
        data["sjc"] = load_sjc(DATA_DIR / "SJC_gold_historical.csv")
    if (DATA_DIR / "Gold_Futures_ohlcv.csv").exists():
        data["gold"] = load_ohlcv(DATA_DIR / "Gold_Futures_ohlcv.csv", GOLD_CLOSE)
    if (DATA_DIR / "USD_Index_ohlcv.csv").exists():
        data["usd"] = load_ohlcv(DATA_DIR / "USD_Index_ohlcv.csv", USD_CLOSE)
    if (DATA_DIR / "VN_Index_ohlcv.csv").exists():
        data["vn"] = load_ohlcv(DATA_DIR / "VN_Index_ohlcv.csv", "VN_Index_Close")
    if (DATA_DIR / "Oil_WTI_ohlcv.csv").exists():
        data["oil"] = load_ohlcv(DATA_DIR / "Oil_WTI_ohlcv.csv", "Oil_Close")
    if (DATA_DIR / "interest_rate.csv").exists():
        df_ir = pd.read_csv(DATA_DIR / "interest_rate.csv", encoding="utf-8-sig")
        df_ir["Date"] = _normalize_date(pd.to_datetime(df_ir["Date"], utc=True))
        rate_col = "Interest_Rate" if "Interest_Rate" in df_ir.columns else "FEDFUNDS"
        df_ir = df_ir[["Date", rate_col]].rename(columns={rate_col: "Interest_Rate"})
        df_ir = df_ir.drop_duplicates(subset=["Date"])
        data["interest"] = df_ir
    return data


def merge_on_date(loaders: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge các bộ dữ liệu theo cột Date (outer join).
    Xử lý lệch múi giờ: tất cả đã chuẩn hóa về date thuần.
    Ngày nghỉ: ngày không có giao dịch sẽ có NaN, xử lý bằng Forward Fill sau.
    """
    merged = None
    for name, df in loaders.items():
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="Date", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def forward_fill_missing(df: pd.DataFrame, use_bfill: bool = False) -> pd.DataFrame:
    """
    Xử lý giá trị thiếu: ffill (kéo giá trước) - không gây look-ahead bias.
    Nếu use_bfill=True: thêm bfill cho đầu chuỗi (CÓ look-ahead bias - không khuyến nghị cho mô hình dự báo).
    Mặc định chỉ ffill rồi dropna để tránh dùng thông tin tương lai.
    """
    df = df.ffill()
    if use_bfill:
        df = df.bfill()
    # Loại dòng còn NaN (đầu chuỗi hoặc toàn bộ trống)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df = df.dropna(subset=numeric_cols, how="all")
    return df


# ============== 3. Xử lý outlier - Winsorization ==============
def handle_outliers(
    df: pd.DataFrame,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Winsorization: giới hạn giá trị cực đoan ở mức percentile (1% và 99%).
    Thay vì xóa, gán giá trị tại ngưỡng - phù hợp dữ liệu tài chính.
    exclude_cols: các cột không xử lý (vd: Date).
    """
    exclude_cols = exclude_cols or []
    result = df.copy()
    numeric_cols = [
        c
        for c in result.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    for col in numeric_cols:
        low = result[col].quantile(lower_percentile / 100)
        high = result[col].quantile(upper_percentile / 100)
        result[col] = result[col].clip(lower=low, upper=high)
    return result


# ============== 4. EDA - Correlation Heatmap ==============
def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """
    Vẽ Correlation Heatmap giữa giá vàng SJC và các chỉ số kinh tế
    (USD Index, giá vàng thế giới Gold; nếu có VN_Index hoặc dầu sẽ tự thêm).
    """
    # Chọn cột số cho correlation (bỏ Date)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        print("Không đủ cột số để vẽ correlation heatmap.")
        return
    corr_df = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Giá vàng SJC và các chỉ số kinh tế")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu biểu đồ: {save_path}")


# ============== 5. Kiểm định ADF (Augmented Dickey-Fuller) ==============
def adf_test(series: pd.Series, name: str) -> dict:
    """
    Kiểm định tính dừng của chuỗi thời gian bằng ADF.
    Trả về dict với ADF stat, p-value, kết luận (dừng nếu p < 0.05).
    """
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return {"name": name, "adf_stat": None, "pvalue": None, "stationary": None}
    result = adfuller(series_clean, autolag="AIC")
    adf_stat, pvalue = result[0], result[1]
    stationary = pvalue < 0.05
    return {
        "name": name,
        "adf_stat": adf_stat,
        "pvalue": pvalue,
        "stationary": stationary,
        "critical_values": result[4],
    }


def run_adf_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chạy ADF cho các cột số trong df, trả về bảng kết quả."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        r = adf_test(df[col], col)
        rows.append(r)
    result_df = pd.DataFrame(rows)
    return result_df


def main():
    ensure_output_dir()

    # 1. Đọc và merge
    loaders = load_all_csvs()
    if not loaders:
        print("Không tìm thấy file CSV trong", DATA_DIR)
        return

    merged = merge_on_date(loaders)
    if merged.empty:
        print("Merge không tạo ra dữ liệu.")
        return

    # 2. Lọc thời gian: chỉ giữ từ 2018-01-01 (phạm vi nghiên cứu)
    merged["Date"] = pd.to_datetime(merged["Date"], utc=True)
    merged = merged[merged["Date"] >= START_DATE].copy()
    merged["Date"] = pd.to_datetime(merged["Date"].dt.date)
    print(f"Đã lọc từ {START_DATE}: còn {len(merged)} dòng")

    # 3. Xử lý missing values (chỉ ffill + dropna để tránh look-ahead bias; không dùng bfill)
    merged = forward_fill_missing(merged, use_bfill=False)

    # 4. Xóa dòng trống SJC_ban_ra (đảm bảo mô hình không lỗi)
    if "SJC_ban_ra" in merged.columns:
        before = len(merged)
        merged = merged.dropna(subset=["SJC_ban_ra"])
        if before > len(merged):
            print(f"Đã xóa {before - len(merged)} dòng có SJC_ban_ra trống. Còn {len(merged)} dòng.")

    # 4b. Loại bỏ mọi dòng còn NaN (vd: 01/01/2018 thiếu Gold_Close, Oil_Close do ngày lễ)
    before_dropna = len(merged)
    merged = merged.dropna()
    if before_dropna > len(merged):
        print(f"Đã xóa {before_dropna - len(merged)} dòng còn NaN (đảm bảo dữ liệu liên tục từ ngày đủ thông tin). Còn {len(merged)} dòng.")

    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã merge và xử lý missing: {MERGED_CSV} ({len(merged)} dòng)")

    # 5. Xử lý outlier (Winsorization 1% - 99%)
    cleaned = handle_outliers(merged, lower_percentile=1.0, upper_percentile=99.0)
    cleaned.to_csv(FINAL_CLEANED_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã xử lý outlier và lưu: {FINAL_CLEANED_CSV} ({len(cleaned)} dòng)")

    # Kiểm tra: in ngày bắt đầu và ngày kết thúc
    min_date = cleaned["Date"].min()
    max_date = cleaned["Date"].max()
    print(f"\n--- Phạm vi dữ liệu final_data_cleaned ---")
    print(f"  Ngày bắt đầu (min Date): {min_date}")
    print(f"  Ngày kết thúc (max Date): {max_date}")

    # 6. EDA - Correlation Heatmap (dùng dữ liệu đã cleaned)
    plot_correlation_heatmap(cleaned, OUTPUT_DIR / "correlation_heatmap.png")

    # 7. ADF test
    adf_results = run_adf_tests(cleaned)
    adf_path = OUTPUT_DIR / "adf_stationarity_results.csv"
    # Lưu chỉ cột có thể serialize (bỏ critical_values dict)
    adf_results[["name", "adf_stat", "pvalue", "stationary"]].to_csv(
        adf_path, index=False, encoding="utf-8-sig"
    )
    print(f"Đã lưu kết quả ADF: {adf_path}")

    # In tóm tắt ADF ra console
    print("\n--- Kiểm định tính dừng (ADF) ---")
    for _, row in adf_results.iterrows():
        if row["pvalue"] is not None:
            conclusion = "Dừng" if row["stationary"] else "Không dừng"
            print(f"  {row['name']}: ADF stat = {row['adf_stat']:.4f}, p-value = {row['pvalue']:.4f} -> {conclusion}")

    print("\nHoàn tất. Output nằm trong:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
