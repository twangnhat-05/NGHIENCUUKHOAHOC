# Changelog

Tất cả thay đổi đáng chú ý của dự án sẽ được ghi tại đây.

Format dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning theo [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — branch `claude/auto-execution`

(Đang chuẩn bị W4 — Foundation models + Conformal PI + XAI)

---

## [0.3.0-w3] — `milestone-3-models` (2026-04-27)

### Added (M3: ML + DL models, statistical tests, combined leaderboard)

#### Tier-2 Tabular ML (7 models)
- `src/models/ml.py`: _MLBaseWrapper với mode-B per-row prediction
  - Ridge, ElasticNet, SVR_RBF, RandomForest, XGBoost, LightGBM, CatBoost
  - StackingRegressor (XGB + LGBM + Cat → Ridge meta)
  - Horizon baked vào constructor; ffill/bfill/fillna handle NaN
- `scripts/run_ml_baselines.py`: CLI entry
- `src/training/tune.py`: Optuna helpers (XGB + LGBM, TPE 30 trials)
- `src/training/trainer.py`: thêm `evaluate_ml_one_fold` cho mode-B path

#### Tier-3 Deep Learning (7 models)
- `src/models/dl_simple.py`: PyTorch sequence wrappers (CPU)
  - LSTM v2 + GRU; sliding window 30 days × 108 features
  - Anti-leakage: scaler fit on train, train_tail bridge cho val predict
  - Early stopping (patience=5), batch=64, Adam lr=1e-3
- `src/models/dl_neuralforecast.py`: 7 NeuralForecast wrappers
  - N-HiTS, N-BEATS, PatchTST, TimeMixer, TSMixer, iTransformer, TFT
  - --fast mode bỏ iTransformer + TFT (CPU slow ~60-120s/run)
- `scripts/run_dl_baselines.py`: CLI entry

#### Statistical tests
- `src/evaluation/stat_tests.py`:
  - Diebold-Mariano (HAC variance, Harvey-Leybourne small-sample correction)
  - Friedman + Nemenyi post-hoc (mean ranks + critical difference table)
  - dm_pairwise_table cho mọi cặp model
- `scripts/combine_leaderboards.py`: merge classical+ml+dl, run Friedman per horizon

#### Tests (47 total, +12 new)
- `tests/test_ml_models.py`: 7 tests (ML scaler no-leakage, target exclusion, fit/predict)
- `tests/test_stat_tests.py`: 5 tests (DM identical=p=1, perfect-vs-naive p<0.001, Friedman ranks)

### Results — Combined leaderboard (23 models × 5 folds × 3 horizons = 345 records)

| Horizon | Top 3 by MAPE | Family |
|---|---|---|
| h=1 | RollingNaive 0.33% / Ridge 0.63% / ElasticNet 0.67% | mode-B / ML / ML |
| h=5 | RollingNaive 0.96% / ElasticNet 1.41% / Ridge 1.67% | mode-B / ML / ML |
| h=20 | RollingNaive 2.67% / ElasticNet 3.06% / LightGBM 3.20% | mode-B / ML / ML |

Friedman test h=20: stat=49.35, p=0.000718 → reject H0 (models differ significantly).

### Key insights cho paper
1. **Linear regularized (Ridge/ElasticNet) bứt phá** với 108 engineered features — beat trees + DL ngắn hạn.
2. **TSMixer là DL tốt nhất** (3.0% MAPE) — MLP-mixing > Transformer (PatchTST, TimeMixer) trên dataset nhỏ.
3. **2024 gold rally regime** (fold 3-4) đẩy MAPE 5-10x — cần regime-aware forecasting (W4 conformal).

### Bug fixes
- ML wrapper Ridge fail NaN → fillna trong fit
- yfinance API empty (W1 carryover) — fix lasting
- MLForecast freq B mismatch VN holidays — integer index workaround

### Defer to W4 / W5
- Optuna tuning per fold (linear models đã good enough)
- iTransformer + TFT (need GPU) — user chạy Colab nếu muốn
- DM pairwise raw-predictions (cần update trainer save preds)
- Sentiment scraping (PhoBERT pipeline ready, news data missing)

---

---

## [0.2.0-w2] — `milestone-2-baselines` (2026-04-26)

### Added (M2: Features V2 + Classical Baselines)

#### Data layer
- `src/data/merge.py` — outer-join 11 raw sources, ffill (no bfill), filter business days, output `data/interim/merged.parquet` (2,169 rows × 16 cols)

#### Features V2 (108 features)
- `src/features/technical.py` — SMA/EMA/RSI/MACD/Bollinger/ATR/Stochastic/realized vol/momentum/ROC
- `src/features/calendar.py` — VN holidays (Tết âm 2018-2027), dow/month/quarter cyclical encoding, days_to_tet
- `src/features/macro.py` — yield spread, USD z-gap, USD/VND change, realized vol, SJC/Gold ratio
- `src/features/build.py` — pipeline: lags + returns + technical + calendar + macro + targets (h=1,5,20)
- `src/features/sentiment.py` — STUB pipeline (PhoBERT/mDeBERTa zero-shot ready, scrape later)
- Output: `data/processed/features_v2.parquet` (1,883 × 112), `features_v2_with_sentiment.parquet` (1,883 × 122)

#### Models (9 classical)
- `src/models/base.py` — `BaseForecaster` abstract interface
- `src/models/classical.py`:
  - **Tier 0 trivial**: NaiveForecaster (mode-A constant), SeasonalNaiveForecaster (cycle), RollingNaiveForecaster (mode-B shift)
  - **Tier 1 statistical** (StatsForecast): AutoARIMA, AutoETS, AutoTheta, HistoricAverage
  - **Prophet** (Meta) wrapper với cmdstanpy backend
  - **MLForecastLGBM** (Nixtla + LightGBM) với auto lag features

#### Training & evaluation
- `src/training/trainer.py` — `evaluate_one_fold` mode-A protocol (single fit train, multi-step forecast, slice align cho horizon)
- `src/evaluation/leaderboard.py` — aggregate per-model/horizon/fold metrics, save CSV + barplot
- `scripts/run_classical_baselines.py` — entry CLI

#### Tests (35 total, +10 new)
- `tests/test_features.py`: 10 sanity tests (no leakage on technical/lag/return/target/calendar/macro)
- VN holidays detection verified (Tết 2024 + 30/4 + 9/2 + Tết Dương)

### Results — Classical leaderboard (5 walk-forward folds)
| Horizon | Best mode-A MAPE | Mode-B floor | Best DA |
|---|---|---|---|
| h=1 | SeasonalNaive 2.91% | RollingNaive 0.33% | AutoETS 52.4% |
| h=5 | SeasonalNaive 3.03% | RollingNaive 0.96% | AutoETS 50.2% |
| h=20 | SeasonalNaive 3.49% | RollingNaive 2.67% | AutoETS 56.3% |

→ **Đã đạt target proposal 4-5% MAPE** ngay từ classical baselines.

### Fixed
- MLForecast freq mismatch (B vs VN holidays) → chuyển sang integer index
- SJC scraper webgia working sau refresh

### Known limitations
- Multi-horizon mode-A: y_pred[i] là (i+1)-step-ahead chứ không phải fixed h-step → "horizon" mang ý nghĩa "min h" — sẽ chuyển sang rolling-origin trong W3 cho ML models
- Sentiment vẫn là STUB zeros (chưa scrape news)
- CPI VN data missing (FRED code 404)

---

---

## [0.1.0-w1] — `milestone-1-foundation` (2026-04-26)

### Added (M1: Foundation)
- `ARCHITECTURE.md` — Master architecture v0.1
- Cookiecutter folder structure: `src/{data,features,models,training,evaluation,xai,utils}`, `configs/`, `tests/`, `notebooks/`, `app/`, `reports/`, `scripts/`, `data/{raw,interim,processed,external}`
- `requirements.txt` + `requirements-dev.txt` pinned theo Architecture v0.1
- `pyproject.toml`: ruff + black + pytest + mypy config
- `README.md`, `LICENSE` (MIT), `CITATION.cff`, `MONITORING.md`, `CLAUDE_EXECUTION_LOG.md`, `CHANGELOG.md`
- `.claude/skills/INDEX.md` — skill registry
- Config files: `configs/data.yaml`, `configs/cv.yaml`, `configs/features.yaml` (no hardcoded constants)
- `src/utils/{io,logging,seeds}.py` — foundation helpers
- `src/data/{schema,fetch,refresh}.py` — config-driven data layer với delta refresh CLI
- `src/training/cv.py` — `WalkForwardCV` (expanding/rolling), `Fold` dataclass, `split_test_holdout`, `build_cv_from_config`
- `src/evaluation/metrics.py` — MAE/RMSE/MAPE/sMAPE/MASE/R²/Directional Accuracy/Hit Rate/CRPS Gaussian
- `tests/test_cv_no_leakage.py` — 12 anti-leakage gates
- `tests/test_metrics.py` — 13 metrics unit tests
- 3 nguồn dữ liệu MỚI vào `data/raw/`: USD/VND, GLD ETF, BTC-USD, USD broad index FRED, 10y treasury

### Changed
- Migrate raw data từ `data/*.csv` → `data/raw/*.csv` (immutable layer)
- Move legacy scripts → `src/legacy/` + README mapping (frozen — KHÔNG sửa)
- Refresh tất cả nguồn yfinance/FRED/vnstock/SJC tới **2026-04-24/25**

### Fixed
- Bug data.yaml.yfinance dict key/value swap → loop iterate ngược
- yfinance API: chuyển từ `Ticker.history()` → `yf.download()` + exponential backoff để tránh rate-limit
- FRED CPI_VN code 404 → tạm bỏ, sẽ lấy thủ công từ GSO (W2)

### Tests
- 25/25 pytest pass (12 no-leakage gates + 13 metrics)

---

## [0.0.0] — `pre-claude-v0` tag (2026-04-26)

Snapshot baseline trước khi Claude Architect refactor.
- 5 legacy scripts: download_data, eda_merge, feature_engineering, compare_baselines, train_xgboost, train_random_forest, train_lstm
- 6 raw CSVs: SJC + 5 yfinance/FRED indicators
- Output legacy: 6 plot PNGs, 4 CSVs

---

## [0.0.0] — `pre-claude-v0` tag (2026-04-26)

Snapshot baseline trước khi Claude Architect refactor.
- 5 legacy scripts: download_data, eda_merge, feature_engineering, compare_baselines, train_xgboost, train_random_forest, train_lstm
- 6 raw CSVs: SJC + 5 yfinance/FRED indicators
- Output legacy: 6 plot PNGs, 4 CSVs
