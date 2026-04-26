# Gold Price Forecasting — TDTU NCKH 2025-2026

> **Đề tài**: Ứng dụng trí tuệ nhân tạo trong dự đoán giá vàng và phân tích biến động thị trường Việt Nam
> **Title (EN)**: Applied Artificial Intelligence to Gold Price Forecasting and Market Volatility in Vietnam
> **Tổ chức**: Đại học Tôn Đức Thắng (TDTU) — Nghiên cứu Khoa học Sinh viên
> **Năm học**: 2025-2026
> **License**: MIT

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

## 🎯 Mục tiêu

Xây dựng hệ thống dự báo giá vàng SJC (Việt Nam) bằng AI/ML, tích hợp dữ liệu đa nguồn (giá vàng quốc tế, USD Index, dầu, lãi suất, sentiment tin tức), so sánh hệ thống các mô hình từ baseline cổ điển đến SOTA 2024-2026 (foundation models zero-shot), và cung cấp dashboard trực quan cho nhà đầu tư.

**Target proposal**: MAPE 4-5% trên test out-of-sample.

## 🏗️ Kiến trúc

Xem [`ARCHITECTURE.md`](ARCHITECTURE.md) để biết chi tiết.

```
NCKH/
├── data/
│   ├── raw/         # CSV gốc, immutable
│   ├── interim/     # merged + ffill
│   ├── processed/   # features V2 + splits
│   └── external/    # news, sentiment cache
├── src/
│   ├── data/        # fetch, refresh, schema
│   ├── features/    # technical, macro, sentiment, calendar
│   ├── models/      # classical, ml, dl, foundation, ensemble
│   ├── training/    # cv (walk-forward), tune (optuna), trainer
│   ├── evaluation/  # metrics, stat_tests (DM, Friedman), conformal
│   ├── xai/         # SHAP, attention, TimeSHAP
│   ├── utils/       # logging, seeds, io
│   └── legacy/      # FROZEN — code gốc trước Claude refactor
├── notebooks/       # 00..99 — EDA → models → ensemble → reproduce
├── tests/           # pytest, no-leakage gates
├── configs/         # YAML configs (no hardcoded constants)
├── app/             # streamlit + fastapi
├── reports/         # figures, leaderboard, paper (TDTU + IEEE)
└── scripts/         # bash helpers
```

## ⚡ Quickstart

### 1. Setup environment
```bash
git clone https://github.com/twangnhat-05/NGHIENCUUKHOAHOC.git
cd NGHIENCUUKHOAHOC
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Refresh data (≤ 2 phút)
```bash
python -m src.data.refresh --start-year 2018 --end-date today
```

### 3. Build features
```bash
python -m src.features.build
```

### 4. Train baselines
```bash
python -m src.models.classical       # AutoARIMA, Prophet, ETS
python -m src.models.ml --model xgboost --tune
```

### 5. Run dashboard
```bash
streamlit run app/streamlit_app.py
```

### 6. Reproduce all (≤ 2 giờ Colab free)
Mở `notebooks/99_reproduce_all.ipynb`.

## 📊 Model Lineup (5 tiers, 30+ models)

| Tier | Models |
|---|---|
| 0 — Trivial | Naive, Seasonal Naive, Drift, SMA |
| 1 — Classical | AutoARIMA, AutoETS, Theta, Prophet, NeuralProphet |
| 2 — ML | Ridge, SVR, RF, XGBoost, LightGBM, CatBoost, Stacking |
| 3 — Deep Learning | LSTM, GRU, TCN, N-HiTS, PatchTST, iTransformer, TFT, TimeMixer, TSMixer |
| 4 — Foundation (zero-shot) | Chronos-Bolt, TTM (IBM), TimesFM, Lag-Llama, Moirai-MoE |
| 5 — Ensemble | Inverse-RMSE weighted, Stacking, Conformal calibration |

## 📏 Evaluation

- **Walk-forward CV** 5 folds (expanding window) — no leakage
- **Final test**: 2025-10-01 → 2026-04-25 (~7 tháng OOS)
- **Metrics**: MAE, RMSE, MAPE, sMAPE, MASE, R², Directional Accuracy, CRPS
- **Statistical tests**: Diebold-Mariano + Friedman + Nemenyi
- **Prediction intervals**: Conformal (Adaptive Conformal Inference, 80/95%)

## 📚 Dữ liệu (free tier only)

| Nguồn | Frequency | Range | Source |
|---|---|---|---|
| Giá vàng SJC mua/bán | Daily | 2018+ | webgia.com (scrape) |
| Gold Futures (GC=F) | Daily | 2018+ | yfinance |
| USD Index (DXY) | Daily | 2018+ | yfinance + FRED |
| VN-Index | Daily | 2018+ | vnstock (VCI) |
| Oil WTI (CL=F) | Daily | 2018+ | yfinance |
| FED funds rate | Monthly | 2018+ | FRED |
| USD/VND | Daily | 2018+ | yfinance (VND=X) |
| News sentiment | Daily | 2020+ | CafeF + PhoBERT (W2) |

## 🛠️ Development

```bash
pip install -r requirements-dev.txt
pre-commit install

# Lint + format
ruff check src/ tests/
black src/ tests/

# Tests
pytest -v
pytest -m no_leakage    # critical anti-leakage gates
pytest --cov=src --cov-report=html
```

## 📖 Tài liệu tham khảo chính

Xem [`reports/paper/ieee_en/bib.bib`](reports/paper/ieee_en/bib.bib).

## 🙏 Acknowledgments

- Open-source community: Nixtla (statsforecast/neuralforecast/mlforecast), Amazon Chronos, Google TimesFM, IBM Granite TTM, Salesforce Moirai
- TDTU advisor (TBD)
- Co-architected with Claude Opus 4.7 (Anthropic)

## 📄 Citation

Nếu sử dụng code này, vui lòng cite (xem [`CITATION.cff`](CITATION.cff)):
```bibtex
@misc{wangnhat2026goldforecast,
  author = {WangNhat},
  title  = {Vietnamese Gold Price Forecasting with Foundation Models and Conformal Prediction},
  year   = {2026},
  url    = {https://github.com/twangnhat-05/NGHIENCUUKHOAHOC}
}
```
