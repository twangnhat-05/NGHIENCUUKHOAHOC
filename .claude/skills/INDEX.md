# 🧩 Skill Registry

Repository skills tự sinh bởi Claude Architect, dùng để chuẩn hóa context cho các sub-task lặp lại trong dự án dự đoán giá vàng.

## Active Skills

| Name | Description | Used | Tokens saved (est) | Last updated |
|------|-------------|------|--------------------|--------------|
| _(chưa có — sẽ auto-tạo khi pattern lặp ≥ 2 lần)_ | | | | |

## Total Impact

- Skills created: **0**
- Total uses: **0**
- Estimated tokens saved: **~0**

## Planned Skills (sẽ tạo khi gặp use case)

| Skill | Mục đích | Token saved/use (est) |
|-------|----------|------------------------|
| `data-fetcher` | Fetch yfinance/FRED/SJC với cache, retry, schema validation | ~800 |
| `time-series-cv` | Walk-forward / expanding window CV, anti data leakage | ~600 |
| `gold-feature-engineering` | Lags, returns, RSI, MACD, Bollinger, volatility, log-returns đặc thù vàng | ~1200 |
| `model-trainer-template` | Training loop với early stopping, checkpoint, MLflow logging | ~1000 |
| `model-evaluator` | RMSE/MAE/MAPE/sMAPE/DA + Diebold-Mariano + plots | ~900 |
| `experiment-logger` | Format chuẩn ghi experiment vào MLflow + leaderboard CSV | ~400 |
| `eda-notebook-template` | Notebook structure: stats, plots, decomposition, ACF/PACF | ~700 |
| `paper-section-writer` | Template viết Introduction/Methods/Results/Discussion (TDTU format) | ~1500 |
| `safety-validator` | Chạy validation gates sau mỗi code change | ~400 |
| `dispatch-prompt-builder` | Tạo prompt template chuẩn cho từng loại sub-model task | ~500 |

## Deprecated Skills

| Name | Reason | Date |
|------|--------|------|
| _(none)_ | | |
