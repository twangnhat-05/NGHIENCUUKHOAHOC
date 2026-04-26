# 🏛️ MASTER ARCHITECTURE — Gold Price Prediction (TDTU NCKH 2025-2026)

> **Version**: v0.1 (Gate 3 draft — chờ user APPROVE)
> **Author**: Claude Architect (Opus 4.7)
> **Date**: 2026-04-26
> **Project owner**: dev2@wolffungame.com (TDTU sinh viên NCKH)
> **Deadline**: 2026-05 (cấp khoa) — **~5 tuần** từ hôm nay.

---

## 1. Mục tiêu & Phạm vi (chốt theo user choices)

| Item | Quyết định |
|---|---|
| **Q1 Target** | Multi-horizon: **h=1, h=5, h=20** ngày + **Directional Accuracy (DA)**. Target chính: SJC `ban_ra` (triệu VND/lượng). |
| **Q2 Features** | Macro VN mở rộng: **SBV policy rate, USD/VND, CPI VN, GLD, BTC, DXY official, news sentiment FinBERT/PhoBERT** + technical (RSI/MACD/Bollinger/ATR/volatility). |
| **Q3 Models** | Đủ proposal (RF, XGB, SVM, LSTM, Transformer) + classical (ARIMA, Prophet) + **SOTA 2024-2026: PatchTST, iTransformer, N-HiTS, TFT, TimeMixer + Foundation models zero-shot (Chronos-Bolt, TTM, TimesFM, Lag-Llama)**. |
| **Q4 Output** | Báo cáo TDTU (Word VN) + bản English IEEE/Springer-ready + repo public + reproducibility notebook + dataset HF Hub (optional). |
| **Q5 Demo** | **Streamlit Cloud** (free) + tùy chọn FastAPI sidecar nếu kịp. |

### Novelty angles cho paper (đã research)
1. **Zero-shot foundation models trên emerging market**: bench Chronos-Bolt vs TimesFM vs Moirai-MoE vs TTM trên SJC với regime analysis (COVID, gold rally 2024-2025).
2. **Conformal coverage trên volatile asset**: ACI (Adaptive Conformal Inference) cho SJC có premium spread vs world gold; sentiment exog từ VN news.
3. **Vietnamese-specific**: PhoBERT-finetuned sentiment + macro VN exog, hiếm có nghiên cứu trước đây.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Raw sources (data/raw/):                                            │
│   • SJC (webgia scraper, 2018-2026)                                  │
│   • Gold Futures GC=F (yfinance)                                     │
│   • USD Index DX-Y.NYB (yfinance) + DXY official (FRED DTWEXBGS)     │
│   • VN-Index (vnstock VCI native)                                    │
│   • Oil WTI CL=F (yfinance)                                          │
│   • FED funds (FRED FEDFUNDS) + SBV policy rate (CSV manual)         │
│   • USD/VND (yfinance VND=X) + CPI VN (FRED CPALTT01VNQ657N)         │
│   • GLD ETF, BTC-USD (yfinance)                                      │
│   • News headlines (CafeF/VnExpress gold tag) — scrape + cache       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ src/data/fetch.py  (robust, retry, cache, schema validation) │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Interim (data/interim/merged.parquet)                              │
│   • Outer join trên Date, ffill (no bfill)                          │
│   • Outlier flag (NOT clip — clip sẽ được fit-on-train-only)        │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  src/features/build.py + technical.py + macro.py + sentiment.py     │
│   • Lags: 1, 3, 5, 7, 14, 30 cho SJC + Gold + USD + Sentiment      │
│   • Returns: simple + log returns, multi-period                      │
│   • Technical: SMA(10/30/60), EMA, RSI(14), MACD, Bollinger,        │
│     ATR, Stochastic, OBV, realized volatility (rolling std)          │
│   • Macro: yield spread, real interest, CPI YoY, USD/VND change      │
│   • Sentiment: PhoBERT-finetuned daily score (mean, count, polarity)│
│   • Calendar: dow, dom, month, quarter, holidays VN                 │
│                                                                      │
│  Output: data/processed/features_v2.parquet                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    SPLIT & CV LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  src/training/cv.py (walk-forward)                                   │
│   • Final test: 2025-10-01 → 2026-04-25 (out-of-sample)             │
│   • Walk-forward CV: 5 folds expanding window trên train             │
│   • Mỗi fold: refit scaler, refit model, refit clip thresholds      │
│   • Anti-leakage tests (tests/test_no_leakage.py)                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER (5 tiers)                          │
├─────────────────────────────────────────────────────────────────────┤
│ Tier 0 Trivial:    Naive, Seasonal Naive, Drift, SMA                │
│ Tier 1 Classical:  AutoARIMA, AutoETS, Theta, Prophet, NeuralProphet│
│ Tier 2 ML:         Linear, SVR, RF, XGB, LightGBM, CatBoost, Stack │
│ Tier 3 DL:         LSTM, GRU, TCN, N-HiTS, PatchTST, iTransformer, │
│                    TFT, TimeMixer, TSMixer                          │
│ Tier 4 Foundation: Chronos-Bolt, TTM (IBM), TimesFM, Lag-Llama,    │
│                    Moirai-MoE  — zero-shot + fine-tune              │
│ Tier 5 Ensemble:   Inverse-RMSE weighted, Stacking, Conformal calib │
│                                                                      │
│  Tuning: Optuna (TPE, 50-100 trials/model).                         │
│  Tracking: MLflow (mlruns/).                                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  EVALUATION & XAI LAYER                              │
├─────────────────────────────────────────────────────────────────────┤
│ Metrics:  MAE, RMSE, MAPE, sMAPE, MASE, R², DA, Hit Rate, CRPS     │
│ Tests:    Diebold-Mariano (pairwise), Friedman + Nemenyi (group)    │
│ PI:       Nixtla ConformalIntervals + MAPIE EnbPI/ACI               │
│ XAI:      SHAP (TreeExplainer for ML), Captum IG (DL),              │
│           TFT attention weights, TimeSHAP                            │
│ Output:   reports/leaderboard/{horizon}/results.csv + figures        │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DELIVERY LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  app/streamlit_app.py     (Streamlit Cloud — public demo)           │
│  app/api/main.py          (FastAPI optional — Render free)          │
│  notebooks/99_reproduce_all.ipynb (one-click reproducibility)        │
│  reports/paper/tdtu_vi/   (Word, VN)                                │
│  reports/paper/ieee_en/   (LaTeX, English, conference-ready)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Folder Structure

```
NCKH/
├── .claude/
│   ├── settings.local.json (gitignored)
│   └── skills/                 # Skill factory
├── .github/workflows/          # (optional) CI: lint + test
├── app/
│   ├── streamlit_app.py
│   └── api/main.py             # FastAPI
├── configs/                    # YAML configs (no hardcoded constants)
│   ├── data.yaml
│   ├── features.yaml
│   ├── cv.yaml
│   └── models/
│       ├── xgboost.yaml
│       ├── lightgbm.yaml
│       ├── lstm.yaml
│       ├── patchtst.yaml
│       ├── tft.yaml
│       ├── chronos.yaml
│       └── ttm.yaml
├── data/
│   ├── raw/                    # immutable, append-only
│   ├── interim/                # merged + ffill
│   ├── processed/              # features + splits
│   └── external/               # news, sentiment cache
├── notebooks/
│   ├── 00_eda_v2.ipynb
│   ├── 01_features_v2.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_dl_models.ipynb
│   ├── 05_foundation_models.ipynb
│   ├── 06_ensembling_conformal.ipynb
│   ├── 07_xai.ipynb
│   └── 99_reproduce_all.ipynb
├── src/
│   ├── data/{fetch,refresh,schema}.py
│   ├── features/{build,technical,macro,sentiment,calendar}.py
│   ├── models/{classical,ml,dl,foundation,ensemble}.py
│   ├── training/{cv,tune,trainer}.py
│   ├── evaluation/{metrics,stat_tests,conformal,leaderboard}.py
│   ├── xai/{shap_utils,attention}.py
│   ├── utils/{logging,seeds,io}.py
│   └── legacy/                  # current scripts moved here, kept frozen
├── tests/
│   ├── test_features.py
│   ├── test_cv_no_leakage.py
│   ├── test_metrics.py
│   └── test_data_schema.py
├── reports/
│   ├── figures/
│   ├── leaderboard/
│   └── paper/
│       ├── tdtu_vi/report.docx
│       └── ieee_en/{main.tex, bib.bib, figures/}
├── scripts/
│   ├── refresh_data.sh
│   ├── train_all.sh
│   └── reproduce.sh
├── output/                     # legacy outputs (kept for traceability)
├── mlruns/                     # MLflow tracking (gitignored)
├── pyproject.toml              # ruff/black/pytest config
├── requirements.txt            # pinned ==
├── requirements-dev.txt
├── environment.yml             # conda alt
├── README.md
├── CHANGELOG.md
├── ARCHITECTURE.md             # this file
├── CITATION.cff
├── LICENSE                     # MIT
├── MONITORING.md
├── CLAUDE_EXECUTION_LOG.md
└── BM02_decuong.pdf
```

---

## 4. Tech Stack (pinned)

```python
# Core
python==3.11
pandas==2.2.3
numpy==1.26.4
scipy==1.14.1
pyarrow==17.0.0   # parquet
pyyaml==6.0.2
tqdm==4.66.5

# Data fetch
yfinance==0.2.50
vnstock==3.2.5
beautifulsoup4==4.12.3
requests==2.32.3
lxml==5.3.0
pandas-datareader==0.10.0

# Classical TS
statsforecast==1.7.8       # AutoARIMA, AutoETS, Theta, MSTL
prophet==1.1.6
neuralprophet==0.9.0

# ML
scikit-learn==1.5.2
xgboost==2.1.2
lightgbm==4.5.0
catboost==1.2.7
optuna==4.0.0
mlforecast==0.13.4

# DL stack
torch==2.4.1+cpu
neuralforecast==1.7.5      # PatchTST, iTransformer, NHITS, TFT, TimeMixer, TSMixer
darts==0.30.0              # backup, also has TFT/PatchTST

# Foundation models
chronos-forecasting==1.4.1
uni2ts==1.2.0               # Moirai
timesfm==1.2.6
# lag-llama: clone from github
transformers==4.45.2        # for FinBERT/PhoBERT/mDeBERTa

# Conformal
mapie==0.9.2

# XAI
shap==0.46.0
captum==0.7.0
# timeshap — optional

# Tracking & UX
mlflow==2.16.2
streamlit==1.40.1
fastapi==0.115.4
uvicorn==0.32.0

# Plotting
matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1

# Quality
pytest==8.3.3
ruff==0.7.1
black==24.10.0
pre-commit==4.0.1
```

---

## 5. Model Lineup chi tiết

### Tier 0 — Trivial (sanity baselines)
| Model | Lib | Config | Why |
|---|---|---|---|
| Naive | inhouse | y_t+1 = y_t | Threshold dưới |
| Seasonal Naive | statsforecast | y_t+1 = y_{t-5} | Tuần hóa |
| Drift / Random Walk | statsforecast | linear extrap | |
| SMA(10) | inhouse | rolling mean | |

### Tier 1 — Classical
| Model | Lib | Notes |
|---|---|---|
| AutoARIMA | statsforecast | Numba-accelerated |
| AutoETS | statsforecast | |
| Theta / MSTL | statsforecast | |
| Prophet | prophet | with VN holidays |
| NeuralProphet | neuralprophet | with exog (USD, sentiment) |

### Tier 2 — Tabular ML (with engineered features)
| Model | Lib | Tuning |
|---|---|---|
| Ridge / Lasso / ElasticNet | sklearn | Optuna |
| SVR (linear + RBF) | sklearn | Optuna |
| Random Forest | sklearn | Optuna |
| XGBoost | xgboost | Optuna 100 trials |
| LightGBM | lightgbm | Optuna 100 trials |
| CatBoost | catboost | Optuna 100 trials |
| Stacking (XGB+LGBM+Cat → Ridge) | sklearn | |

### Tier 3 — Deep Learning trainable
| Model | Lib | Multivar+Exog | Hardware |
|---|---|---|---|
| LSTM (refactored) | torch | yes | CPU |
| GRU | torch | yes | CPU |
| TCN | darts | yes | CPU |
| N-HiTS | neuralforecast | exog yes | CPU |
| PatchTST | neuralforecast | univar (channel-indep) | CPU |
| **iTransformer** | neuralforecast | multivar native | T4 free Colab |
| **TFT** | neuralforecast | static + future-known + past exog | T4 free Colab |
| TimeMixer | neuralforecast | multivar | CPU |
| TSMixer | neuralforecast | multivar | CPU |

### Tier 4 — Foundation models
| Model | HF | Size | License | Mode |
|---|---|---|---|---|
| **Chronos-Bolt-Small** | `amazon/chronos-bolt-small` | 48M | Apache-2.0 | Zero-shot + fine-tune |
| **TTM (Tiny Time Mixer)** ⭐ | `ibm-granite/granite-timeseries-ttm-r2` | ~1M | Apache-2.0 | Native exog, fastest CPU |
| **TimesFM 2.0** | `google/timesfm-2.0-500m-pytorch` | 500M | Apache-2.0 | Zero-shot |
| **Lag-Llama** | `time-series-foundation-models/Lag-Llama` | 2.4M | Apache-2.0 | Probabilistic |
| **Moirai-MoE-Small** | `Salesforce/moirai-moe-1.0-R-small` | ~117M | Apache-2.0 | Native multivar+exog |

> **⚠️ Lưu ý license**: Moirai-1.1-R (small/base) là CC-BY-NC; chỉ dùng `moirai-moe` (Apache-2.0) cho safety. Cần verify HF model card trước khi dùng.

### Tier 5 — Ensembling
- Simple average / median trên top-3 model mỗi horizon
- Weighted by inverse RMSE (validation)
- Stacking với Ridge meta-learner trên out-of-fold predictions
- Conformal calibration on top of best ensemble

---

## 6. Evaluation Protocol

### 6.1 Splits
- **Final test set** (out-of-sample, không touch trong CV): **2025-10-01 → 2026-04-25** (~7 tháng)
- **Train+Val (CV)**: 2018-01-01 → 2025-09-30
- **Walk-forward CV**: 5 folds expanding window
  - Fold 1: train 2018-2021, val 2022-Q1
  - Fold 2: train 2018-2022Q1, val 2022Q2-Q3
  - Fold 3: train 2018-2022Q3, val 2022Q4-2023Q1
  - Fold 4: train 2018-2023Q1, val 2023Q2-Q3
  - Fold 5: train 2018-2023Q3, val 2023Q4-2024
- Refit scaler + clip thresholds + model PER fold (no leakage).

### 6.2 Metrics (mỗi horizon h ∈ {1, 5, 20})
| Metric | Why |
|---|---|
| **MAE** (triệu VND) | Đơn vị thực tế |
| **RMSE** | Penalize outliers |
| **MAPE** (%) | So sánh proposal target 4-5% |
| **sMAPE** | Đối xứng |
| **MASE** | So với seasonal naive — chuẩn academic |
| **R²** | Explained variance |
| **Directional Accuracy (DA)** | % đoán đúng hướng |
| **Hit Rate** | DA chia theo strict threshold |
| **CRPS** | Probabilistic (cho models có distribution output) |
| **Coverage @ 80/95%** | Cho Conformal PI |

### 6.3 Statistical tests
- **Diebold-Mariano** pairwise (model A vs B trên test errors)
- **Friedman test + Nemenyi post-hoc** (so sánh nhiều model trên multiple folds, chuẩn ICML/NeurIPS)
- **Bonferroni correction** khi multiple comparisons

### 6.4 Reproducibility
- Seeds fixed (`numpy`, `torch`, `random`, `sklearn`, env `PYTHONHASHSEED`)
- Run mỗi DL model 3 seeds → report mean ± std
- All hyperparams logged trong MLflow
- `notebooks/99_reproduce_all.ipynb` chạy end-to-end < 2 giờ trên Colab free

---

## 7. Roadmap (5 tuần — đến 2026-05-31)

| Tuần | Dates | Milestone | Deliverable | Tag |
|---|---|---|---|---|
| **W1** | Apr 27 – May 3 | M1: Refactor + Data refresh + CV framework | Project structure, data tới Apr 25, walk-forward CV no-leakage tests pass | `milestone-1-foundation` |
| **W2** | May 4 – May 10 | M2: Features V2 + Sentiment + Classical baselines | features_v2.parquet, FinBERT/PhoBERT pipeline, AutoARIMA/Prophet/NeuralProphet/MLForecast leaderboard | `milestone-2-baselines` |
| **W3** | May 11 – May 17 | M3: ML models + DL models | XGB/LGBM/Cat/SVR + LSTM/GRU/N-HiTS/PatchTST/iTransformer/TFT/TimeMixer trên CV; Optuna tuning + MLflow | `milestone-3-models` |
| **W4** | May 18 – May 24 | M4: Foundation models + Ensemble + Conformal + XAI | Chronos-Bolt + TTM + TimesFM + Lag-Llama bench; ensemble; ACI conformal; SHAP/Captum/attention | `milestone-4-frontier` |
| **W5** | May 25 – May 31 | M5: Dashboard + API + Paper + Final QC | Streamlit Cloud live, FastAPI optional, TDTU report.docx, IEEE paper draft, reproducibility nb pass | `milestone-5-delivery` |

**Buffer**: nếu chậm 1 tuần → cắt foundation models xuống 2 (chỉ Chronos-Bolt + TTM) và FastAPI optional → chỉ Streamlit.

---

## 8. Skill Catalog (sẽ tự tạo khi gặp pattern lặp)

Theo INDEX.md, sẽ tạo skills sau khi pattern lặp ≥ 2 lần:
1. `data-fetcher` — wrap yfinance + FRED + vnstock + cache
2. `time-series-cv` — walk-forward + no-leakage validators
3. `gold-feature-engineering` — kit lags + technical + macro + sentiment
4. `model-trainer-template` — neuralforecast + MLflow + Optuna
5. `model-evaluator` — metrics + DM + plots + leaderboard
6. `experiment-logger` — MLflow + leaderboard CSV
7. `eda-notebook-template` — stats + decomposition + ACF/PACF
8. `paper-section-writer` — TDTU VN + IEEE EN templates
9. `safety-validator` — pre-commit gates
10. `dispatch-prompt-builder` — chuẩn prompt cho sub-agents

---

## 9. Risks & Mitigation

| Risk | Severity | Mitigation |
|---|---|---|
| **Free Colab GPU quota cạn** | Medium | TTM + Chronos-Bolt chạy CPU OK; chỉ TFT/iTransformer cần GPU; user chạy lệnh export trên Kaggle (30h/tuần) nếu cần |
| **HF model name/license thay đổi tháng 4/2026** | Medium | Code parametrize tên model qua config YAML, dễ swap; verify HF card trước khi pin version |
| **Sentiment data: scrape CafeF lỗi** | Low | Fallback: skip sentiment, vẫn đủ proposal nếu chỉ macro |
| **NeuralForecast version bug với foundation models** | Low | Mỗi foundation model có thể chạy stand-alone với HF transformers nếu neuralforecast bug |
| **Deadline trượt** | Medium | Buffer 1 tuần; cut foundation models nếu cần |
| **Data SJC scrape rate-limit khi refresh** | Low | Chỉ refresh delta (90 ngày), delay 0.6s/request → ~1 phút |
| **Disk space cho MLflow + foundation model weights** | Low | TimesFM 500M = 2GB; cleanup mlruns định kỳ; user có ≥10GB free |

---

## 10. Acceptance Criteria (closure cho dự án)

- [ ] Walk-forward CV với 5 folds, no leakage (test pass)
- [ ] ≥ 5 baselines (Naive, SNaive, AutoARIMA, Prophet, NeuralProphet)
- [ ] ≥ 6 ML models (Ridge, SVR, RF, XGB, LGBM, Cat, Stacking)
- [ ] ≥ 5 DL models (LSTM, N-HiTS, PatchTST, iTransformer, TFT)
- [ ] ≥ 3 foundation models zero-shot + 1 fine-tuned (Chronos-Bolt, TTM, TimesFM)
- [ ] DM tests + Friedman + Nemenyi báo cáo statistical significance
- [ ] Conformal PI với coverage report 80/95%
- [ ] SHAP + attention interpretability
- [ ] Streamlit dashboard live
- [ ] TDTU report Word + IEEE LaTeX paper draft
- [ ] Reproducibility notebook chạy < 2 giờ trên Colab free
- [ ] requirements.txt pinned, README chuẩn, MIT license
- [ ] MAPE < 5% trên test cho horizon h=1 (target proposal) cho ít nhất 1 model
