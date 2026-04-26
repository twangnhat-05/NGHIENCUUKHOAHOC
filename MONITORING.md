# 🎯 PROJECT HEALTH DASHBOARD

> **Auto-updated by Claude Architect** — last refresh: 2026-04-26 (after M1)

## 📊 CODE QUALITY
- Test coverage: **48% / 25 tests** (no_leakage gates: 12/12 ✅; metrics: 13/13 ✅) — target ≥ 70% by M3
- Lint: pyproject.toml configured (ruff + black) — chưa chạy lint scan đầu
- Type hints coverage: **~50%** trong `src/` (strict trên public APIs)
- TODO/FIXME count: **0**
- Cyclomatic complexity: nhỏ (≤ 8 per function)

## 🧪 EXPERIMENT TRACKING
- Models trained (legacy, before refactor): **5** (frozen ở `src/legacy/`)
- Models trained (new pipeline): **0** (sẽ bắt đầu W2)
- Best MAPE: chưa có (W2 baselines first)
- Best directional accuracy: chưa đo
- Statistical tests: chưa có (W4)
- Latest leaderboard: chưa có

## 📦 DATA HEALTH (sau refresh 2026-04-26)
| Source | Rows | Latest | Status |
|---|---|---|---|
| SJC gold | 2,015 | 2026-04-24 | ✅ Fresh (51 rows new) |
| Gold Futures (GC=F) | 2,090 | 2026-04-24 | ✅ |
| USD Index (DX-Y.NYB) | 2,091 | 2026-04-24 | ✅ |
| Oil WTI (CL=F) | 2,091 | 2026-04-24 | ✅ |
| USD/VND (VND=X) | 2,164 | 2026-04-24 | ✅ NEW |
| GLD ETF | 2,089 | 2026-04-24 | ✅ NEW |
| BTC-USD | 3,037 | 2026-04-25 | ✅ NEW |
| VN-Index (vnstock VCI) | 2,172 | 2026-04-24 | ✅ |
| FED Funds (FRED) | 99 | 2026-03-01 | ✅ Monthly |
| USD Broad Index (FRED DTWEXBGS) | 2,069 | 2026-04-17 | ✅ NEW |
| 10Y Treasury (FRED DGS10) | 2,077 | 2026-04-23 | ✅ NEW |
| ❌ CPI VN (FRED CPALTT01VNQ657N) | — | — | 404 — code đã bị FRED gỡ; phải lấy từ GSO |

- Data drift detected: **chưa đo** (W2)
- Missing values: 0% (sau ffill)
- Outliers: **đã thiết kế fit-on-train-fold** trong `configs/features.yaml` (sẽ dùng W2)

## ⚙️ INFRA & RESOURCES
- Disk usage (project): ~5MB code + ~3MB data raw
- Memory peak: <500MB (data load)
- Total training time: 0 phút
- Free tier quota left: 100%

## 🤖 AI ORCHESTRATION
- Total dispatches: Opus 1 (research SOTA) | Sonnet 0 | Haiku 0
- Token consumption (estimated): ~25k tokens
- Skills created: **0** (planning to auto-create khi pattern lặp ≥ 2x)
- Skills reused: 0

## 🚨 ALERTS (sau M1)
- ✅ **DATA_LEAKAGE_WINSORIZE**: configs/features.yaml.outliers.fit_on=`train_fold`; sẽ enforce trong W2 build_features
- ✅ **NO_WALK_FORWARD_CV**: `src/training/cv.WalkForwardCV` đã build + 12 no-leakage tests pass
- 🔴 **MISSING_MODELS**: thiếu ARIMA/SARIMA, Prophet, SVM, Transformer, foundation models (sẽ làm W2-W4)
- 🟡 **NO_DM_TEST**: chưa có (W3-W4)
- ✅ **NO_PROJECT_STRUCTURE**: cookiecutter structure hoàn tất, configs YAML đã có
- ✅ **STALE_DATA**: refresh tới 2026-04-24/25
- 🟡 **NO_XAI_FULL**: W4
- 🟡 **CPI_VN_404**: cần user thêm tay từ gso.gov.vn → data/external/cpi_vn.csv (W2)

## 🔖 GIT STATE
- Current branch: `claude/auto-execution`
- Local commits trên branch: ~5 (xem `git log`)
- Latest tag: `pre-claude-v0` (rollback point) → sắp tag `milestone-1-foundation`
- Remote `origin` = github.com/twangnhat-05/NGHIENCUUKHOAHOC
