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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_READY_CSV = OUTPUT_DIR / "model_ready_data.csv"
TRAIN_RATIO = 0.8
WINDOW_SIZE = 30  
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 10


def load_data():
    """Đọc model_ready_data.csv, sort theo Date."""
    df = pd.read_csv(MODEL_READY_CSV, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%), tránh chia cho 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) >= 1e-10
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def build_sequences(scaled_data: np.ndarray, return_t: np.ndarray, window: int):
    """
    Tạo cấu trúc 3D cho LSTM: (samples, timesteps, features).

    LSTM cần input shape (batch, timesteps, features):
    - timesteps = 30 (30 ngày liên tiếp);
    - features = số cột đặc trưng.

    Cách xử lý shape:
    - scaled_data có shape (n_rows, n_features) = 2D.
    - Với mỗi chỉ số i (0 .. n_rows-window): lấy cửa sổ 30 ngày [i : i+30]
      -> một mẫu có shape (30, n_features) = 1 sample x 30 timesteps x n_features.
    - Gộp n_samples cửa sổ -> X có shape (n_samples, 30, n_features) = 3D.
    - y[i] = return tại thời điểm (i+30-1), tức return từ ngày 30 sang ngày 31.

    Trả về:
      X: (n_samples, window, n_features), n_samples = n_rows - window
      y: (n_samples,)
    """
    n = len(scaled_data)
    n_samples = n - window
    n_features = scaled_data.shape[1]
    X = np.zeros((n_samples, window, n_features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    for i in range(n_samples):
        X[i] = scaled_data[i : i + window]  # 30 ngày
        # Return từ ngày (i+window-1) sang ngày (i+window) = return_t tại hàng i+window-1
        y[i] = return_t[i + window - 1]
    return X, y


def main():
    if not MODEL_READY_CSV.exists():
        print(f"Không tìm thấy {MODEL_READY_CSV}. Chạy feature_engineering.py trước.")
        return

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, callbacks
    except ImportError:
        print("Cần cài TensorFlow: pip install tensorflow")
        return

    print("=" * 60)
    print("LSTM - Dự báo % thay đổi (Return) giá vàng SJC")
    print("=" * 60)

    df = load_data()
    n = len(df)
    feature_cols = [c for c in df.columns if c not in ("Date", "target")]
    n_features = len(feature_cols)

    # Return: (target / SJC_ban_ra) - 1
    return_t = (df["target"] / df["SJC_ban_ra"]).values - 1.0

    # Chia Train/Test theo thời gian (80/20)
    split_idx = int(n * TRAIN_RATIO)
    train_data = df[feature_cols].iloc[:split_idx].values
    test_data = df[feature_cols].iloc[split_idx:].values

    # MinMaxScaler: CHỈ fit trên Train, transform toàn bộ (tránh leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    scaled_full = scaler.transform(df[feature_cols].values)

    # Tạo dữ liệu 3D (samples, timesteps=30, features)
    X_all, y_all = build_sequences(scaled_full, return_t, WINDOW_SIZE)
    n_seq = len(X_all)

    # Chia train/test theo thời gian (train: các sequence nằm trọn trong train period)
    # Train: sequence i với i+WINDOW_SIZE-1 <= split_idx-1  => i <= split_idx - WINDOW_SIZE
    # Test:  sequence i với i >= split_idx
    train_end = split_idx - WINDOW_SIZE
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_test, y_test = X_all[split_idx:], y_all[split_idx:]

    # Giá thực tế và SJC (ngày cuối của window) cho Test — khớp độ dài với y_test (n - split_idx - WINDOW_SIZE)
    n_test = n - split_idx - WINDOW_SIZE
    test_sjc_last = df["SJC_ban_ra"].values[split_idx + WINDOW_SIZE - 1 : split_idx + WINDOW_SIZE - 1 + n_test]
    test_y_price = df["target"].values[split_idx + WINDOW_SIZE : split_idx + WINDOW_SIZE + n_test]
    test_dates = df["Date"].values[split_idx + WINDOW_SIZE : split_idx + WINDOW_SIZE + n_test]

    print(f"\n1. Dữ liệu 3D (shape) - giải thích:")
    print(f"   Input LSTM cần (batch, timesteps, features).")
    print(f"   X_train: {X_train.shape}  -> (samples, timesteps={WINDOW_SIZE}, features={n_features})")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_test:  {y_test.shape}")
    print(f"   MinMaxScaler: fit chỉ trên Train, transform [0,1] (tránh leakage).")

    # Xử lý shape 3D: LSTM nhận input (batch, timesteps, features) = (None, 30, n_features)
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, n_features)),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.Dropout(DROPOUT_RATE),
        layers.LSTM(LSTM_UNITS, return_sequences=False),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Dự báo return, quy đổi sang giá (triệu VND)
    y_pred_return = model.predict(X_test, verbose=0).flatten()
    predicted_price_lstm = (y_pred_return + 1.0) * test_sjc_last

    rmse_lstm = np.sqrt(mean_squared_error(test_y_price, predicted_price_lstm))
    mae_lstm = mean_absolute_error(test_y_price, predicted_price_lstm)
    mape_lstm = mape(test_y_price, predicted_price_lstm)

    print(f"\n2. Đánh giá LSTM trên tập Test (đơn vị triệu VND/lượng):")
    print(f"   RMSE = {rmse_lstm:.4f}")
    print(f"   MAE  = {mae_lstm:.4f}")
    print(f"   MAPE = {mape_lstm:.2f}%")

    # Naive và XGBoost trên CÙNG period (LSTM test period) để so sánh công bằng
    # Naive: ŷ(ngày k) = SJC_ban_ra(ngày k-1) = test_sjc_last (cùng độ dài n_test)
    naive_pred = test_sjc_last
    mape_naive = mape(test_y_price, naive_pred)

    # XGBoost: chạy nhanh trên full test rồi lấy slice trùng LSTM test
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler as StdScaler
    X_train_xgb = df[feature_cols].iloc[:split_idx].values
    X_test_xgb = df[feature_cols].iloc[split_idx:].values
    scaler_xgb = StdScaler()
    X_train_xgb_s = scaler_xgb.fit_transform(X_train_xgb)
    X_test_xgb_s = scaler_xgb.transform(X_test_xgb)
    y_train_return = (df["target"] / df["SJC_ban_ra"]).iloc[:split_idx].values - 1.0
    test_sjc_full = df["SJC_ban_ra"].iloc[split_idx:].values
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.8, random_state=42)
    xgb_model.fit(X_train_xgb_s, y_train_return, eval_set=[(X_test_xgb_s, (df["target"]/df["SJC_ban_ra"]).iloc[split_idx:].values - 1.0)], verbose=False)
    xgb_return = xgb_model.predict(X_test_xgb_s)
    xgb_price_full = (xgb_return + 1.0) * test_sjc_full
    # LSTM test = từ ngày split_idx+30 đến n-1 -> XGBoost test index 30 đến cuối (độ dài n_test)
    xgb_price_aligned = xgb_price_full[WINDOW_SIZE :]
    mape_xgb = mape(test_y_price, xgb_price_aligned)

    # Bảng tổng kết: Naive vs XGBoost vs LSTM (cùng period)
    print("\n" + "-" * 60)
    print("BẢNG TỔNG KẾT (cùng tập Test - period LSTM): Naive vs XGBoost vs LSTM")
    print("-" * 60)
    print(f"{'Chỉ số':<12} {'Naive':>14} {'XGBoost':>14} {'LSTM':>14}")
    print("-" * 60)
    rmse_naive = np.sqrt(mean_squared_error(test_y_price, naive_pred))
    rmse_xgb = np.sqrt(mean_squared_error(test_y_price, xgb_price_aligned))
    mae_naive = mean_absolute_error(test_y_price, naive_pred)
    mae_xgb = mean_absolute_error(test_y_price, xgb_price_aligned)
    print(f"{'RMSE':<12} {rmse_naive:>14.4f} {rmse_xgb:>14.4f} {rmse_lstm:>14.4f}")
    print(f"{'MAE':<12} {mae_naive:>14.4f} {mae_xgb:>14.4f} {mae_lstm:>14.4f}")
    print(f"{'MAPE (%)':<12} {mape_naive:>13.2f}% {mape_xgb:>13.2f}% {mape_lstm:>13.2f}%")
    print("-" * 60)
    best = min([("Naive", mape_naive), ("XGBoost", mape_xgb), ("LSTM", mape_lstm)], key=lambda x: x[1])
    print(f"   >>> Mô hình có MAPE thấp nhất: {best[0]} (MAPE = {best[1]:.2f}%)")
    if mape_lstm < mape_naive:
        improvement = (mape_naive - mape_lstm) / mape_naive * 100.0
        print(f"   >>> LSTM cải thiện độ chính xác so với Naive là {improvement:.1f}%")

    # Biểu đồ: Thực tế vs XGBoost vs LSTM
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_dates, test_y_price, label="Thực tế", alpha=0.9)
    ax.plot(test_dates, xgb_price_aligned, label="XGBoost", alpha=0.8)
    ax.plot(test_dates, predicted_price_lstm, label="LSTM", alpha=0.8)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá SJC (triệu VND/lượng)")
    ax.set_title("So sánh: Thực tế vs XGBoost vs LSTM (tập Test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "lstm_actual_vs_xgboost_vs_lstm.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n3. Đã lưu biểu đồ: {plot_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
