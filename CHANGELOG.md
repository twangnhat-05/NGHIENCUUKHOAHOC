# Changelog

Tất cả thay đổi đáng chú ý của dự án sẽ được ghi tại đây.

Format dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning theo [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — branch `claude/auto-execution`

### Added (W1: Foundation)
- `ARCHITECTURE.md` — Master architecture v0.1
- Folder structure: `src/{data,features,models,training,evaluation,xai,utils}`, `configs/`, `tests/`, `notebooks/`, `app/`, `reports/`, `scripts/`
- `requirements.txt` (pinned), `requirements-dev.txt`, `pyproject.toml`
- `README.md`, `LICENSE` (MIT), `CITATION.cff`, `MONITORING.md`, `CLAUDE_EXECUTION_LOG.md`
- `.claude/skills/INDEX.md` — skill registry
- Config files: `configs/data.yaml`, `configs/cv.yaml`, `configs/features.yaml`

### Changed
- Migrate raw data từ `data/*.csv` → `data/raw/*.csv` (immutable layer)
- Move legacy scripts → `src/legacy/` (frozen — KHÔNG sửa)

### Fixed
- _(W1 sắp fix data leakage trong outlier winsorization — config đã set `fit_on: train_fold`)_

---

## [0.0.0] — `pre-claude-v0` tag (2026-04-26)

Snapshot baseline trước khi Claude Architect refactor.
- 5 legacy scripts: download_data, eda_merge, feature_engineering, compare_baselines, train_xgboost, train_random_forest, train_lstm
- 6 raw CSVs: SJC + 5 yfinance/FRED indicators
- Output legacy: 6 plot PNGs, 4 CSVs
