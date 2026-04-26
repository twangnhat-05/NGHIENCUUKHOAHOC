# 📜 CLAUDE EXECUTION LOG

> Audit trail cho mọi action của Claude Architect trên dự án.

## HOW TO ROLLBACK

```bash
# Rollback toàn bộ về trước khi Claude bắt đầu:
cd "D:/WangNhat/Study/NCKH"
git checkout main
git branch -D claude/auto-execution   # nếu muốn xóa hẳn nhánh làm việc
git reset --hard pre-claude-v0

# Rollback về milestone N (sau khi đã có milestone-N tag):
git reset --hard milestone-N
```

---

## 2026-04-26

### 19:45 — STEP 6/7: Backup + Infra init
- ✅ `git init -b main`
- ✅ Tạo `.gitignore`, `MONITORING.md`, `CLAUDE_EXECUTION_LOG.md`, `.claude/skills/INDEX.md`
- ✅ Baseline commit `b9a059e`
- ✅ Cleanup commit `3a63995` (untrack settings.local.json)

### 19:55 — STEP 6.5: Remote setup
- ✅ Set remote `origin` → `https://github.com/twangnhat-05/NGHIENCUUKHOAHOC.git`
  (initial chọn `Twangnhatt/NCKH` bị 403; user chuyển sang account `twangnhat-05`)
- ✅ Rebase `--allow-unrelated-histories` để giữ README.md từ remote
- ✅ Push `main` (fast-forward) — local ahead by 2 commits
- ✅ Tạo + push tag `pre-claude-v0` (= commit `3a63995`) → ROLLBACK POINT

### 20:00 — Branch isolation
- ✅ Tạo + switch sang branch `claude/auto-execution`
- Mọi thay đổi tiếp theo commit lên branch này, KHÔNG đụng `main` cho tới khi user merge.

### Permissions granted
- User trả lời "GRANT ALL" → full autonomy theo permission template (destructive ops vẫn cần CONFIRM riêng).

### 20:15 — Gate 2 Clarification
- 5 câu hỏi gửi user; user trả lời "ALL ⭐" + research thêm Q3 (SOTA models 2025-2026)
- Spawn research agent (general-purpose, background) → trả về SOTA stack: Chronos-Bolt, TTM, TimesFM, Lag-Llama, PatchTST, iTransformer, N-HiTS, TFT, TimeMixer
- Note: WebSearch denied trong sub-agent → research dựa trên training cutoff (Jan 2026); user cần verify HF model cards trước W4

### 20:25 — Gate 3 Master Architecture
- `ARCHITECTURE.md` v0.1 commit `da9d5db` — 5-tier model lineup, walk-forward CV, 5-week roadmap
- User confirm: APPROVE + có account Colab cho TFT/iTransformer

### 20:30 — M1 (W1) Foundation execution
- Cookiecutter folder structure (src/{...}, configs/, tests/, notebooks/, app/, reports/)
- Migrate raw → data/raw/, legacy → src/legacy/ (frozen)
- Project meta: requirements.txt + dev pinned, pyproject.toml (ruff/black/pytest), README, LICENSE, CITATION
- Configs: data.yaml, cv.yaml, features.yaml (no hardcoded constants; outliers.fit_on=train_fold)
- src skeleton: utils, data.{schema,fetch,refresh}, training.cv (WalkForwardCV), evaluation.metrics
- Install: pyarrow, pytest, vnstock, pandas-datareader (others đã có sẵn)
- Bug fix: data.yaml.yfinance dict key/value swap → loop iterate ngược
- Bug fix: yfinance API → chuyển sang `yf.download()` + backoff
- Refresh tất cả nguồn → 2026-04-24/25 (8 nguồn fresh, +3 nguồn mới: USD/VND, GLD, BTC, FRED DTWEXBGS, FRED DGS10)
- Tests: 25/25 pass (12 no_leakage + 13 metrics)
- Skill candidates đã nhận diện (sẽ tạo W2 khi pattern lặp): `data-fetcher` đã ngầm tồn tại trong fetch.py

### 20:50 — M1 closure
- Update MONITORING.md, CHANGELOG.md
- Commit + tag `milestone-1-foundation` + push remote ✅

### 21:00 — M2 (W2) start: Features V2 + Sentiment + Classical
- Install: statsforecast, prophet, mlforecast, lightgbm, neuralprophet (background ~3 phút)
  - Prophet downgraded numpy 2.1 → 1.26.4 (compatibility với cmdstanpy)
- W2.2 src/data/merge.py: 11 raw sources → 2169 business-day rows
  - Bug found: BTC trade 7/7 → weekend pollute → filter to business days
- W2.3 src/features/{technical,calendar,macro,build}.py: 108 features từ 16 raw cols
  - Calendar: VN holidays (cố định + Tết âm lookup table 2018-2027)
  - Macro: yield spread, USD z-gap, USD/VND change, realized vol
- W2.4 src/features/sentiment.py: stub pipeline (PhoBERT-ready, scrape defer)
- W2.5 src/models/{base,classical}.py: 9 models (3 trivial + 4 statsforecast + Prophet + MLForecast_LGBM)
- W2.6 src/training/trainer.py + src/evaluation/leaderboard.py:
  - Mode-A evaluation: single fit train, n_val-step forecast, align cho horizon
  - Bug fix: Naive shifted y_observed → wrong; sửa thành mode-A constant
- W2.7 Run baselines: 135 records collected (9 × 5 folds × 3 horizons)
  - Output: reports/leaderboard/classical_full_{long,summary}.csv + 12 plots PNG
- Tests: 35/35 PASS (12 CV no-leakage + 13 metrics + 10 features)

### 22:35 — M2 closure
- Sắp commit + tag `milestone-2-baselines` + push



