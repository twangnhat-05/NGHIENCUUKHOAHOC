# Changelog

Tất cả thay đổi đáng chú ý của dự án sẽ được ghi tại đây.

Format dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning theo [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — branch `claude/auto-execution`

(Đang ở W2)

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
