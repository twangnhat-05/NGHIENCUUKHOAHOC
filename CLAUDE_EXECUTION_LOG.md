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
- Commit + tag `milestone-2-baselines` + push ✅

### 22:50 — M3 (W3) start: ML + DL + stat tests
- Install: catboost 1.2.10, mlflow 3.11.1, neuralforecast 3.1.7 (background ~3 phút)
- W3.2 ML wrappers: Ridge/ElasticNet/SVR/RF/XGB/LGBM/CatBoost + Stacking
  - Mode-B per-row using features_v2 (108 features)
  - Bug fix: Ridge fillna trong fit
- W3.4 Run ML baselines: 105 records, Ridge h=1 = 0.63% MAPE (BEAT classical)
- W3.5 LSTM v2 + GRU PyTorch sequence (CPU, 30-day window × 108 features)
- W3.6 NeuralForecast wrappers: NHITS/NBEATS/PatchTST/TimeMixer/TSMixer + iTransformer/TFT
  - --fast mode bỏ iTransformer/TFT (~60-120s/run trên CPU)
- W3.7 Statistical tests: DM (HAC + small-sample) + Friedman + Nemenyi
- DL benchmark chạy ngầm ~58 phút (5 NF + 2 simple × 3 horizons × 5 folds = 105 records)
- Tests: 47/47 PASS (+12: 7 ML + 5 stat tests)

### 00:15 (2026-04-27) — Combined leaderboard
- 23 models × 5 folds × 3 horizons = 345 records
- Friedman h=20: stat=49.35, p=0.000718 → reject H0
- Top: RollingNaive (mode-B floor) > Ridge > ElasticNet
- Best DL: TSMixer (3.0% MAPE h=1) — MLP-mixing > Transformer

### 00:20 — M3 closure
- Commit + tag `milestone-3-models` + push ✅

## 2026-04-27

### 00:25 — M4 (W4) start: Foundation + Ensemble + Conformal + XAI
- Install: chronos-forecasting 2.2.2, mapie 1.3.0, captum 0.9.0, shap 0.51.0
- W4.2 src/models/foundation.py:
  - ChronosBoltForecaster + TTMForecaster wrappers
  - API fix: Chronos-Bolt 2.x dùng `predict_quantiles(context, prediction_length, quantile_levels)`
  - Smoke test Chronos-Bolt h=1: MAPE 0.65% (parity với Ridge 0.63%)
- W4.3 Run foundation full benchmark (15 records, ~5 phút):
  - Chronos-Bolt h=1=3.07%, h=5=3.20%, h=20=3.65% averaged across 5 folds
- W4.4 src/models/ensemble.py: weighted/inverse_rmse/median ensemble
- W4.5 src/evaluation/conformal.py: split conformal + ACI (Gibbs & Candès 2021)
- W4.6 src/xai/{shap_utils,attention}.py: TreeSHAP + Captum IG + attention rollout
- W4 demo (run_xai_conformal_demo.py):
  - SHAP top-20 cho LightGBM: lag features dominate, macro contribute
  - ACI ElasticNet h=1 last fold (2024 rally): 83% coverage @ alpha=0.10
- Combined v2 leaderboard: 24 models × 5 folds × 3 horizons = 360 records
- Friedman: p < 0.001 cho h=1/5/20 (strong rejection — models differ)

### 00:38 — M4 closure
- Commit + tag `milestone-4-frontier` + push ✅

### 00:45 — M5 (W5) start: Dashboard + Paper + Reproducibility
- Install: streamlit 1.56, fastapi, uvicorn
- W5.1 Streamlit app: 4 tabs (overview/leaderboard/predict/xai), HTTP 200 verified
- W5.2 FastAPI: 5 endpoints, predict(h=1) returns 171.6→170.3 (-0.76%)
- W5.3 Reproducibility: scripts/reproduce_all.{sh,bat}
- W5.4 TDTU report: 14-page Vietnamese Markdown với auto-filled results
- W5.5 IEEE paper: English LaTeX 2-column + bib.bib, submission-ready
- W5.6 Final QC: 47/47 tests pass, no regression

### 01:00 — M5 closure (FINAL DELIVERY)
- Sắp commit + tag `milestone-5-delivery` + push
- Project COMPLETE — sẵn sàng merge vào main



