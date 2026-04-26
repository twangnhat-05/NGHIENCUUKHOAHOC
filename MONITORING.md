# 🎯 PROJECT HEALTH DASHBOARD

> **Auto-updated by Claude Architect** — last refresh: 2026-04-26 (after M2)

## 📊 CODE QUALITY
- Tests: **35/35 PASS** (12 no-leakage CV gates + 13 metrics + 10 features)
- Test coverage: ~55% (target ≥ 70% by M3)
- Lint: pyproject.toml configured (ruff + black) — chưa enforce
- Type hints coverage: ~50% trong `src/`
- TODO/FIXME count: 0

## 🧪 EXPERIMENT TRACKING (M2 Classical Baselines)

**135 records collected** (9 models × 5 walk-forward folds × 3 horizons {h=1, 5, 20})

### Top 5 MAPE @ h=1 (mode-A baselines, mean ± std across 5 folds)
| Rank | Model | MAPE (%) | RMSE | DA (%) |
|---|---|---|---|---|
| 🥇 mode-B floor | RollingNaive | 0.33 ± 0.23 | 0.43 | (n/a) |
| 1 | SeasonalNaive | 2.91 ± 2.92 | 2.72 | 51.2 |
| 2 | Naive | 3.12 ± 3.18 | 2.82 | 48.6 |
| 3 | AutoARIMA | 3.13 ± 2.37 | 2.79 | 50.4 |
| 4 | AutoTheta | 3.15 ± 2.44 | 2.80 | 49.9 |
| 5 | AutoETS | 3.17 ± 2.37 | 2.81 | **52.4** ⭐ |
| | MLForecast_LGBM | 3.25 ± 3.20 | 3.02 | 47.0 |
| | Prophet | 3.90 ± 2.91 | 3.45 | 49.4 |
| | HistoricAverage | 23.29 ± 4.79 | 17.29 | 44.6 |

**Key insights**:
- Mode-A baselines cluster ~3.0–3.3% MAPE — **đã đạt target proposal 4-5%** (proposal target = trên test, ta đang trên CV val)
- Mode-B `RollingNaive` floor = 0.33% MAPE → bound dưới cho mọi mô hình ML/DL "biết hôm qua"
- AutoETS thắng nhẹ về DA (52.4%) ở h=1
- HistoricAverage vô dụng (gold trend mạnh từ 60→90 trong 8 năm)
- Prophet underperform vs AutoARIMA — ARIMA là baseline mạnh nhất trong classical

### MAPE @ h=5
- Tất cả mode-A: 3.0-3.4% (sai lệch nhỏ vs h=1 — gold daily noise dominates regime)
- RollingNaive: 0.96% (degrade nhanh từ 0.33% → 0.96% khi xa nguồn)

### MAPE @ h=20
- Tất cả mode-A: 3.5-3.9%
- AutoETS thắng DA = 56.3% — mô hình "smooth" nhất
- RollingNaive 2.67% (degrade tiếp khi h tăng)

### Diebold-Mariano test: **chưa chạy** (W3-W4)
### Conformal prediction intervals: **chưa chạy** (W4)

## 📦 DATA HEALTH
| Source | Rows | Latest |
|---|---|---|
| 11 raw sources | 99-3037 | 2026-04-24/25 |
| Merged (interim/merged.parquet) | **2,169** | 2018-01-01 → 2026-04-25 (B days) |
| Features V2 (processed/features_v2.parquet) | **1,883** | 2018-12-19 → 2026-03-27 (after warm-up + target shift) |
| Features V2 + sentiment (stub) | **1,883** | thêm 10 cột sentiment zeros |
| Total features | **122** (108 numeric + Date + 3 targets + 10 sentiment) | |

- ✅ Data leakage trong outlier WIN: defer to per-fold (configs/features.yaml.outliers.fit_on=`train_fold`)
- ✅ Business-day filter (drop weekends do BTC trade 7/7 không match SJC schedule)

## ⚙️ INFRA & RESOURCES
- Disk: ~50MB project (raw 3MB + interim+processed 4MB + plots 2MB + .git 40MB)
- Memory peak: ~600MB trong train (LightGBM)
- Training time M2: ~3 phút total (Naive 0s, ARIMA 2-3s/fold, Prophet ~10s/fold, MLForecast 0.5s)
- Free tier quota: 100% (chưa dùng GPU)

## 🤖 AI ORCHESTRATION
- Total dispatches: Opus 1 (research SOTA, ~25k tokens)
- Skills created: 0 (chưa cần — code đã modular)
- Sub-model usage: minimal (Opus chính)

## 🚨 ALERTS (sau M2)
- ✅ **DATA_LEAKAGE_WINSORIZE**: config OK, sẽ enforce trong W3 trainer (fit per fold)
- ✅ **NO_WALK_FORWARD_CV**: 12 no-leakage gates pass
- ✅ **NO_PROJECT_STRUCTURE**: cookiecutter
- ✅ **STALE_DATA**: refreshed
- ✅ **MISSING_BASELINES_CLASSICAL**: 9 models OK (Naive, SNaive, RollingNaive, AutoARIMA, AutoETS, AutoTheta, HistoricAverage, Prophet, MLForecast_LGBM)
- 🔴 **MISSING_MODELS_ML_ADVANCED**: SVR, RF v2, XGBoost v2, LightGBM v2 với engineered features + Optuna tuning (W3)
- 🔴 **MISSING_MODELS_DL**: LSTM v2, GRU, N-HiTS, PatchTST, iTransformer, TFT, TimeMixer, TSMixer (W3)
- 🔴 **MISSING_MODELS_FOUNDATION**: Chronos-Bolt, TTM, TimesFM, Lag-Llama (W4)
- 🟡 **NO_DM_TEST**: Diebold-Mariano + Friedman+Nemenyi (W3-W4)
- 🟡 **NO_CONFORMAL_PI**: ACI conformal intervals (W4)
- 🟡 **NO_XAI_FULL**: SHAP/Captum/attention (W4)
- 🟡 **CPI_VN_404**: cần user lấy thủ công từ GSO (optional)
- 🟡 **NO_REAL_SENTIMENT**: sentiment stub zeros — scrape CafeF + PhoBERT chưa chạy (optional)

## 🔖 GIT STATE
- Current branch: `claude/auto-execution`
- Local commits trên branch: ~10 (xem `git log`)
- Tags: `pre-claude-v0` (rollback), `milestone-1-foundation`, sắp `milestone-2-baselines`
- Remote: github.com/twangnhat-05/NGHIENCUUKHOAHOC (pushed)
