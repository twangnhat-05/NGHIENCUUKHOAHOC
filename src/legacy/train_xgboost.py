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
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ============== Cấu hình ==============
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_READY_CSV = OUTPUT_DIR / "model_ready_data.csv"
TRAIN_RATIO = 0.8  # 80% train, 20% test (theo thời gian)
EARLY_STOPPING_ROUNDS = 50
TOP_N_IMPORTANCE = 10

# Tham số XGBoost (early_stopping_rounds trong constructor; fit() chỉ dùng eval_set)
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror",
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}


def _create_model():
    """Tạo XGBRegressor; nếu phiên bản cũ không hỗ trợ early_stopping_rounds trong constructor thì bỏ qua."""
    try:
        return xgb.XGBRegressor(**XGB_PARAMS)
    except TypeError:
        params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        return xgb.XGBRegressor(**params)


def load_and_split():
    """Đọc model_ready_data.csv, chia Train (80%) / Test (20%) theo thời gian. Trả thêm train_df, test_df."""
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
    y_test_price = test_df["target"].values  # Giá thực tế ngày t+1 (để tính MAPE triệu đồng)
    test_dates = test_df["Date"]
    test_sjc = test_df["SJC_ban_ra"].values  # Giá hôm nay để đổi return -> giá

    return X_train, train_df, X_test, test_df, y_test_price, test_dates, test_sjc, feature_cols


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
    print("XGBoost - Dự báo % thay đổi (Return) rồi quy đổi sang giá (triệu VND)")
    print("=" * 60)

    # 1. Chuẩn bị dữ liệu: Train/Test theo thời gian
    X_train, train_df, X_test, test_df, y_test_price, test_dates, test_sjc, feature_cols = load_and_split()
    # Target = Return: (target / SJC_ban_ra) - 1 (tránh mô hình "ám ảnh" bởi độ lớn 80–90 triệu)
    y_train_return = (train_df["target"] / train_df["SJC_ban_ra"]).values - 1.0
    y_test_return = (test_df["target"] / test_df["SJC_ban_ra"]).values - 1.0
    print(f"\n1. Dữ liệu: Train {len(X_train)} mẫu, Test {len(X_test)} mẫu (theo thời gian)")
    print(f"   Target = Return (%% thay đổi). Đánh giá MAPE trên đơn vị triệu đồng sau khi nhân ngược.")
    print(f"   Số biến độc lập: {len(feature_cols)}")

    # 2. Chuẩn hóa: CHỈ fit trên Train, transform Train & Test (tránh data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\n2. StandardScaler: fit chỉ trên Train, transform Train và Test (không dùng Test khi fit)")

    # 3. Huấn luyện: early_stopping_rounds trong constructor (nếu có), eval_set trong fit()
    model = _create_model()
    model.fit(
        X_train_scaled,
        y_train_return,
        eval_set=[(X_test_scaled, y_test_return)],
        verbose=False,
    )
    print(f"\n3. Đã huấn luyện XGBoost (n_estimators={XGB_PARAMS['n_estimators']}, lr={XGB_PARAMS['learning_rate']}, max_depth={XGB_PARAMS['max_depth']}, subsample={XGB_PARAMS['subsample']})")
    print(f"   Early stopping: early_stopping_rounds={EARLY_STOPPING_ROUNDS} (trong constructor), eval_set=Test")

    # 4. Dự báo Return rồi quy đổi sang giá (triệu VND); đánh giá RMSE/MAE/MAPE trên giá
    y_pred_return = model.predict(X_test_scaled)
    predicted_price = (y_pred_return + 1.0) * test_sjc
    rmse = np.sqrt(mean_squared_error(y_test_price, predicted_price))
    mae_val = mean_absolute_error(y_test_price, predicted_price)
    mape_val = mape(y_test_price, predicted_price)
    print(f"\n4. Đánh giá trên tập Test (đơn vị triệu VND/lượng):")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   MAE  = {mae_val:.4f}")
    print(f"   MAPE (XGBoost) = {mape_val:.2f}%")

    # 4b. Mô hình Naive (baseline): giá ngày mai = giá hôm nay (ŷ_{t+1} = y_t)
    y_pred_naive = test_sjc  # giá hôm nay = dự báo giá ngày mai
    mape_naive = mape(y_test_price, y_pred_naive)
    print(f"   MAPE (Naive baseline, ŷ_{{t+1}}=y_t) = {mape_naive:.2f}%")
    if mape_val < mape_naive:
        improvement_pct = (mape_naive - mape_val) / mape_naive * 100.0
        print(f"\n   >>> Mô hình AI cải thiện độ chính xác so với mô hình cơ sở là {improvement_pct:.1f}%")
    else:
        print(f"\n   >>> Mô hình Naive có MAPE thấp hơn hoặc bằng XGBoost trên tập Test này.")

    # 5. Biểu đồ Actual vs Predicted (giá)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_dates.values, y_test_price, label="Giá thực tế", alpha=0.8)
    ax.plot(test_dates.values, predicted_price, label="Giá dự báo", alpha=0.8)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá SJC (triệu VND/lượng)")
    ax.set_title("XGBoost: Giá thực tế vs Giá dự báo (tập Test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_actual_path = OUTPUT_DIR / "xgboost_actual_vs_predicted.png"
    plt.savefig(plot_actual_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n5. Đã lưu biểu đồ so sánh: {plot_actual_path}")

    # 6. Feature Importance: SHAP nếu có, không thì top-10 feature_importances_
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    use_shap = False
    try:
        import shap
        sample_size = min(500, len(X_train_scaled))
        explainer = shap.TreeExplainer(model, X_train_scaled[:sample_size])
        shap_vals = explainer.shap_values(X_test_scaled[: min(200, len(X_test_scaled))])
        if hasattr(shap_vals, "shape") and shap_vals.ndim >= 1:
            shap_mean = np.abs(np.asarray(shap_vals)).mean(axis=0)
            if len(shap_mean) == len(feature_cols):
                fi_df["shap"] = shap_mean
                use_shap = True
    except Exception:
        pass
    fi_df = fi_df.sort_values("shap" if use_shap else "importance", ascending=False).head(TOP_N_IMPORTANCE)
    fi_plot = fi_df.sort_values("shap" if use_shap else "importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, len(fi_plot) * 0.3)))
    col = "shap" if use_shap else "importance"
    ax.barh(fi_plot["feature"], fi_plot[col], color="steelblue", alpha=0.8)
    ax.set_xlabel("|SHAP| (impact)" if use_shap else "Feature Importance")
    ax.set_title("XGBoost: Biến nào ảnh hưởng lớn nhất đến giá vàng SJC? (Top 10)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plot_fi_path = OUTPUT_DIR / "xgboost_feature_importance.png"
    plt.savefig(plot_fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"6. {'SHAP' if use_shap else 'feature_importances_'}; lưu top {TOP_N_IMPORTANCE}: {plot_fi_path}")

    print(f"\n--- Top {TOP_N_IMPORTANCE} Feature Importance (cao -> thấp) ---")
    for _, row in fi_df.iterrows():
        val = row.get("shap", row["importance"])
        print(f"   {row['feature']}: {val:.4f}")

    # 7. Dự báo thực tế: giá vàng SJC cho phiên tiếp theo (ngày mai)
    last_date = test_dates.iloc[-1]
    last_sjc = test_df["SJC_ban_ra"].iloc[-1]
    X_last = X_test.iloc[[-1]]
    X_last_scaled = scaler.transform(X_last)
    pred_return_tomorrow = model.predict(X_last_scaled)[0]
    pred_tomorrow_price = (pred_return_tomorrow + 1.0) * last_sjc
    if hasattr(last_date, "strftime"):
        last_str = pd.Timestamp(last_date).strftime("%d/%m/%Y")
    else:
        last_str = str(pd.Timestamp(last_date).date())
    print(f"\n--- Dự báo thực tế ---")
    print(f"   Dữ liệu đến ngày: {last_str}")
    print(f"   Dự báo giá vàng SJC cho phiên tiếp theo (ngày mai): {pred_tomorrow_price:.2f} triệu VND/lượng")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
