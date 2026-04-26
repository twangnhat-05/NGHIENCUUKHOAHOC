# -*- coding: utf-8 -*-
"""
NCKH - Dự báo giá vàng Việt Nam
Huấn luyện Random Forest: dự báo Return (% thay đổi), StandardScaler,
đánh giá RMSE/MAE/MAPE trên giá (triệu VND), so sánh Naive vs XGBoost vs RF.
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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============== Cấu hình ==============
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_READY_CSV = OUTPUT_DIR / "model_ready_data.csv"
TRAIN_RATIO = 0.8  # 80% train, 20% test (theo thời gian)
TOP_N_IMPORTANCE = 10

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
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
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_test_price = test_df["target"].values
    test_dates = test_df["Date"].values
    test_sjc = test_df["SJC_ban_ra"].values

    return (
        X_train,
        train_df,
        X_test,
        test_df,
        y_test_price,
        test_dates,
        test_sjc,
        feature_cols,
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%), tránh chia cho 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) >= 1e-10
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def main():
    if not MODEL_READY_CSV.exists():
        print(f"Không tìm thấy {MODEL_READY_CSV}. Chạy feature_engineering.py trước.")
        return

    print("=" * 60)
    print("Random Forest - Dự báo Return rồi quy đổi sang giá (triệu VND)")
    print("=" * 60)

    # 1. Dữ liệu: Train/Test theo thời gian (80/20, không shuffle)
    (
        X_train,
        train_df,
        X_test,
        test_df,
        y_test_price,
        test_dates,
        test_sjc,
        feature_cols,
    ) = load_and_split()

    y_train_return = (train_df["target"] / train_df["SJC_ban_ra"]).values - 1.0
    y_test_return = (test_df["target"] / test_df["SJC_ban_ra"]).values - 1.0

    print(f"\n1. Dữ liệu: Train {len(X_train)} mẫu, Test {len(X_test)} mẫu (theo thời gian)")
    print(f"   Target = Return (%% thay đổi). Đánh giá trên đơn vị triệu VND sau khi quy đổi.")
    print(f"   Số biến: {len(feature_cols)}")

    # 2. Chuẩn hóa: StandardScaler fit trên Train, transform Train & Test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"\n2. StandardScaler: fit chỉ trên Train, transform Train và Test")

    # 3. Huấn luyện Random Forest
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train_s, y_train_return)
    print(f"\n3. Đã huấn luyện Random Forest: n_estimators={RF_PARAMS['n_estimators']}, max_depth={RF_PARAMS['max_depth']}, random_state={RF_PARAMS['random_state']}")

    # 4. Dự báo Return -> quy đổi sang giá; đánh giá RMSE, MAE, MAPE
    y_pred_return = model.predict(X_test_s)
    predicted_price_rf = (y_pred_return + 1.0) * test_sjc
    rmse_rf = np.sqrt(mean_squared_error(y_test_price, predicted_price_rf))
    mae_rf = mean_absolute_error(y_test_price, predicted_price_rf)
    mape_rf = mape(y_test_price, predicted_price_rf)

    print(f"\n4. Đánh giá Random Forest trên tập Test (triệu VND/lượng):")
    print(f"   RMSE = {rmse_rf:.4f}")
    print(f"   MAE  = {mae_rf:.4f}")
    print(f"   MAPE = {mape_rf:.2f}%")

    # 5. Naive và XGBoost trên CÙNG tập Test để so sánh
    naive_pred = test_sjc  # ŷ_{t+1} = giá hôm nay (SJC_ban_ra)
    rmse_naive = np.sqrt(mean_squared_error(y_test_price, naive_pred))
    mae_naive = mean_absolute_error(y_test_price, naive_pred)
    mape_naive = mape(y_test_price, naive_pred)

    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    xgb_model.fit(
        X_train_s,
        y_train_return,
        eval_set=[(X_test_s, y_test_return)],
        verbose=False,
    )
    xgb_return = xgb_model.predict(X_test_s)
    xgb_price = (xgb_return + 1.0) * test_sjc
    rmse_xgb = np.sqrt(mean_squared_error(y_test_price, xgb_price))
    mae_xgb = mean_absolute_error(y_test_price, xgb_price)
    mape_xgb = mape(y_test_price, xgb_price)

    # 6. Bảng so sánh: Naive vs XGBoost vs Random Forest (cùng tập Test)
    print("\n" + "-" * 60)
    print("BẢNG SO SÁNH (cùng tập Test): Naive vs XGBoost vs Random Forest")
    print("-" * 60)
    print(f"{'Chỉ số':<20} {'Naive':>12} {'XGBoost':>12} {'Random Forest':>14}")
    print("-" * 60)
    print(f"{'RMSE':<20} {rmse_naive:>12.4f} {rmse_xgb:>12.4f} {rmse_rf:>14.4f}")
    print(f"{'MAE':<20} {mae_naive:>12.4f} {mae_xgb:>12.4f} {mae_rf:>14.4f}")
    print(f"{'MAPE (%)':<20} {mape_naive:>11.2f}% {mape_xgb:>11.2f}% {mape_rf:>13.2f}%")
    print("-" * 60)
    best = min([("Naive", mape_naive), ("XGBoost", mape_xgb), ("Random Forest", mape_rf)], key=lambda x: x[1])
    print(f"   >>> Mô hình có MAPE thấp nhất: {best[0]} (MAPE = {best[1]:.2f}%)")

    # 7. Biểu đồ Top 10 Feature Importance
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False).head(TOP_N_IMPORTANCE)
    fi_plot = fi_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(fi_plot) * 0.35)))
    ax.barh(fi_plot["feature"], fi_plot["importance"], color="forestgreen", alpha=0.8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest: Top 10 biến quan trọng nhất (Vàng thế giới, Dầu, Lãi suất...)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plot_fi_path = OUTPUT_DIR / "rf_feature_importance.png"
    plt.savefig(plot_fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n7. Đã lưu biểu đồ Feature Importance: {plot_fi_path}")

    # 8. Biểu đồ Giá thực tế vs Giá dự báo (RF)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_dates, y_test_price, label="Giá thực tế", alpha=0.8)
    ax.plot(test_dates, predicted_price_rf, label="Giá dự báo (RF)", alpha=0.8)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá SJC (triệu VND/lượng)")
    ax.set_title("Random Forest: Giá thực tế vs Giá dự báo (tập Test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_actual_path = OUTPUT_DIR / "rf_actual_vs_predicted.png"
    plt.savefig(plot_actual_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"8. Đã lưu biểu đồ Actual vs Predicted: {plot_actual_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
