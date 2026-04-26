# 🎯 PROJECT HEALTH DASHBOARD

> **Auto-updated by Claude Architect** — last refresh: 2026-04-26 (init)

## 📊 CODE QUALITY
- Test coverage: **0%** (target ≥ 70%)
- Lint score: **n/a** (ruff/black chưa cài)
- Type hints coverage: **~5%**
- TODO/FIXME count: **0**
- Cyclomatic complexity: **n/a**

## 🧪 EXPERIMENT TRACKING
- Models trained (legacy, before Claude): **5** — Naive, Linear Regression, Random Forest, XGBoost, LSTM
- Best MAPE so far (legacy, single 80/20 split): **chưa được audit lại**
- Best directional accuracy: **chưa đo**
- Statistical tests: **❌ chưa có Diebold-Mariano**
- Latest leaderboard: `output/` (legacy plots, chưa systematic)

## 📦 DATA HEALTH
- Last data refresh: ~2026-01-31 (≈ 3 tháng trước — **cần refresh**)
- Data drift detected: **chưa đo**
- Missing values: **0%** sau ffill
- Outliers: winsorized 1%-99% (⚠️ leakage)
- Schema match: ✅ stable

## ⚙️ INFRA & RESOURCES
- Disk usage (project): ~3MB code + ~2MB data + ~2MB output
- Memory peak: n/a
- Total training time: n/a
- Free tier quota left: 100% (chưa dùng)

## 🤖 AI ORCHESTRATION
- Total dispatches: Opus 0 | Sonnet 0 | Haiku 0
- Token consumption (estimated): 0
- Skills created: **0** (xem `.claude/skills/INDEX.md`)
- Skills reused: 0
- Failed dispatches: 0

## 🚨 ALERTS (active)
- 🔴 **DATA_LEAKAGE_WINSORIZE**: `eda_merge_analysis.handle_outliers()` tính percentile trên toàn bộ dataset (bao gồm test) — phải fix
- 🔴 **NO_WALK_FORWARD_CV**: chỉ one-shot 80/20 — kết quả không robust
- 🔴 **MISSING_MODELS**: thiếu ARIMA/SARIMA, Prophet, SVM, Transformer (proposal yêu cầu)
- 🟡 **NO_DM_TEST**: thiếu Diebold-Mariano để so sánh statistical significance
- 🟡 **NO_PROJECT_STRUCTURE**: scripts flat, không có src/, tests/, configs/
- 🟡 **STALE_DATA**: SJC dữ liệu tới 2026-01, cần refresh tới 04-2026
- 🟡 **NO_XAI_FULL**: chỉ SHAP cho XGB, thiếu LIME, thiếu RF/LSTM

## 🔖 GIT STATE
- Current branch: `main` (về sau sẽ work trên `claude/auto-execution`)
- Latest commit: (sắp baseline)
- Latest tag: (sắp `pre-claude-v0`)
- Dirty files: tất cả (chưa commit lần đầu)
