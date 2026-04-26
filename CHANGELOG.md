# Changelog

Tất cả thay đổi đáng chú ý của dự án sẽ được ghi tại đây.

Format dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning theo [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — branch `claude/auto-execution`

(Đang chuẩn bị W3 — ML/DL models)

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
