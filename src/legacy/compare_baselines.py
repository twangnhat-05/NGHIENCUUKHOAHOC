# -*- coding: utf-8 -*-
"""
NCKH - So sánh mô hình baseline cho báo cáo
So sánh: Naive (ŷ_{t+1}=y_t) vs Linear Regression vs XGBoost
trên cùng tập Test. In bảng RMSE, MAE, MAPE và % cải thiện so với Naive.
"""

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ============== Cấu hình (trùng train_xgboost) ==============
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_READY_CSV = OUTPUT_DIR / "model_ready_data.csv"
TRAIN_RATIO = 0.8
EARLY_STOPPING_ROUNDS = 50
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror",
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}


def load_and_split():
    """Đọc model_ready_data.csv, chia Train (80%) / Test (20%) theo thời gian."""
    df = pd.read_csv(MODEL_READY_CSV, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    feature_cols = [c for c in df.columns if c not in ("Date", "target")]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test_price = test_df["target"].values
    test_sjc = test_df["SJC_ban_ra"].values
    return X_train, train_df, X_test, test_df, y_test_price, test_sjc, feature_cols


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%), tránh chia cho 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) >= 1e-10
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def _create_xgb():
    try:
        return xgb.XGBRegressor(**XGB_PARAMS)
    except TypeError:
        params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        return xgb.XGBRegressor(**params)


def main():
    if not MODEL_READY_CSV.exists():
        print(f"Không tìm thấy {MODEL_READY_CSV}. Chạy feature_engineering.py trước.")
        return

    X_train, train_df, X_test, test_df, y_test_price, test_sjc, feature_cols = load_and_split()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_return = (train_df["target"] / train_df["SJC_ban_ra"]).values - 1.0
    y_test_return = (test_df["target"] / test_df["SJC_ban_ra"]).values - 1.0

    # ----- 1. Naive: ŷ_{t+1} = y_t -----
    y_pred_naive = test_sjc
    rmse_naive = np.sqrt(mean_squared_error(y_test_price, y_pred_naive))
    mae_naive = mean_absolute_error(y_test_price, y_pred_naive)
    mape_naive = mape(y_test_price, y_pred_naive)

    # ----- 2. Linear Regression (dự báo giá trực tiếp) -----
    lr = LinearRegression()
    lr.fit(X_train_scaled, train_df["target"].values)
    y_pred_lr = lr.predict(X_test_scaled)
    rmse_lr = np.sqrt(mean_squared_error(y_test_price, y_pred_lr))
    mae_lr = mean_absolute_error(y_test_price, y_pred_lr)
    mape_lr = mape(y_test_price, y_pred_lr)

    # ----- 3. XGBoost (dự báo return rồi quy đổi sang giá) -----
    model_xgb = _create_xgb()
    model_xgb.fit(
        X_train_scaled,
        y_train_return,
        eval_set=[(X_test_scaled, y_test_return)],
        verbose=False,
    )
    y_pred_return_xgb = model_xgb.predict(X_test_scaled)
    y_pred_xgb = (y_pred_return_xgb + 1.0) * test_sjc
    rmse_xgb = np.sqrt(mean_squared_error(y_test_price, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test_price, y_pred_xgb)
    mape_xgb = mape(y_test_price, y_pred_xgb)

    # ----- Bảng so sánh -----
    print("=" * 70)
    print("SO SÁNH MÔ HÌNH: Naive vs Linear Regression vs XGBoost (cùng tập Test)")
    print("=" * 70)
    print(f"  Số mẫu Test: {len(y_test_price)}")
    print()
    print("-" * 70)
    print(f"{'Chỉ số':<12} {'Naive':>18} {'Linear Regression':>18} {'XGBoost':>18}")
    print("-" * 70)
    print(f"{'RMSE':<12} {rmse_naive:>18.4f} {rmse_lr:>18.4f} {rmse_xgb:>18.4f}")
    print(f"{'MAE':<12} {mae_naive:>18.4f} {mae_lr:>18.4f} {mae_xgb:>18.4f}")
    print(f"{'MAPE (%)':<12} {mape_naive:>17.2f}% {mape_lr:>17.2f}% {mape_xgb:>17.2f}%")
    print("-" * 70)

    # ----- Kết luận: % cải thiện so với Naive -----
    print()
    if mape_xgb < mape_naive:
        improvement = (mape_naive - mape_xgb) / mape_naive * 100.0
        print(f"  >>> Mô hình AI (XGBoost) cải thiện độ chính xác so với mô hình cơ sở (Naive) là {improvement:.1f}%")
    else:
        print("  >>> XGBoost không tốt hơn Naive trên tập Test này.")
    if mape_xgb < mape_lr:
        improvement_lr = (mape_lr - mape_xgb) / mape_lr * 100.0
        print(f"  >>> XGBoost cải thiện so với Linear Regression là {improvement_lr:.1f}%")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
