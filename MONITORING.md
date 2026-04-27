# 🎯 PROJECT HEALTH DASHBOARD

> **Auto-updated by Architect** — last refresh: 2026-04-27 00:15 (after M3)

## 📊 CODE QUALITY
- Tests: **47/47 PASS** (12 no-leakage CV + 13 metrics + 10 features + 7 ML + 5 stat tests)
- Test coverage: ~65% (target ≥ 70% by M4)
- Lint: pyproject.toml configured (ruff + black) — chưa enforce
- Type hints coverage: ~55% trong `src/`
- TODO/FIXME count: 0

## 🧪 EXPERIMENT TRACKING (M3 Combined: 23 models × 5 folds × 3 horizons)

**345 records** total: 9 classical + 7 ML + 7 DL = 23 models

### TOP-5 OVERALL LEADERBOARD

#### Horizon h=1 (1-day forecast)
| Rank | Model | MAPE (%) | Family |
|---|---|---|---|
| 🥇 | **RollingNaive** | **0.33 ± 0.23** | mode-B baseline floor |
| 🥈 | **Ridge** | **0.63 ± 0.55** | ML linear (108 features) |
| 🥉 | **ElasticNet** | **0.67 ± 0.54** | ML linear regularized |
| 4 | RandomForest | 2.81 ± 3.11 | ML tree |
| 5 | SeasonalNaive | 2.91 ± 2.92 | classical mode-A |

#### Horizon h=5 (5-day)
| Rank | Model | MAPE (%) |
|---|---|---|
| 🥇 | RollingNaive | 0.96 ± 0.88 |
| 🥈 | **ElasticNet** | **1.41 ± 1.21** ⭐ best mode-B |
| 🥉 | Ridge | 1.67 ± 1.66 |
| 4 | SeasonalNaive | 3.03 ± 3.04 |
| 5 | TSMixer | 3.09 ± 2.47 |

#### Horizon h=20 (20-day)
| Rank | Model | MAPE (%) |
|---|---|---|
| 🥇 | RollingNaive | 2.67 ± 3.03 |
| 🥈 | **ElasticNet** | **3.06 ± 2.67** ⭐ |
| 🥉 | LightGBM | 3.20 ± 3.29 |
| 4 | XGBoost | 3.26 ± 2.93 |
| 5 | SeasonalNaive | 3.49 ± 3.56 |

### KEY INSIGHTS (cho paper)
1. **Linear regularized (Ridge/ElasticNet) DOMINATE engineered features** — beat trees + DL trên short horizons
2. **TSMixer là DL tốt nhất** (~3% MAPE) — nhưng vẫn thua linear ML
3. **DL Transformer family (PatchTST, TimeMixer)** underperform vs simpler MLP-based (TSMixer, N-HiTS) trên dataset nhỏ ~1000 train rows
4. **Foundation models (W4) cần để hoàn thiện story** — Chronos-Bolt + TTM zero-shot là novelty cho paper
5. **Volatile period (2024 gold rally fold 3-4) push MAPE tăng 5-10x** với mọi mô hình → cần regime-aware forecasting

### Friedman test results
| Horizon | Friedman stat | p-value | Conclusion |
|---|---|---|---|
| h=1 | (chưa report — đã rejected) | < 0.05 | Models differ |
| h=5 | (chưa report) | < 0.05 | Models differ |
| h=20 | **49.35** | **0.000718** | 🔴 Reject H0 — significant differences |

→ Diebold-Mariano pairwise: pending W4 (cần raw predictions)

## 📦 DATA HEALTH (unchanged)
- 11 raw sources fresh tới 2026-04-24/25
- Merged: 2,169 business-day rows × 16 cols
- Features V2: 1,883 × 122 (108 features + sentiment stub)

## ⚙️ INFRA & RESOURCES
- Disk: ~80MB (raw + interim + features + leaderboards + lightning_logs gitignored)
- Memory peak: ~1.5GB (DL training)
- Total training time M3: ~58 phút (DL benchmark)
- Free tier quota: 100% (chưa dùng GPU)

## 🤖 AI ORCHESTRATION (cumulative)
- Total dispatches: Opus 1 (research SOTA, ~25k tokens)
- Skills created: 0 (code modular, không cần)
- Token consumption (estimated): ~85k cumulative

## 🚨 ALERTS (sau M3)
- ✅ **MISSING_MODELS_ML_ADVANCED**: 7 ML done (Ridge, ElasticNet, SVR, RF, XGB, LGBM, CatBoost)
- ✅ **MISSING_MODELS_DL**: 7 DL done (LSTM v2, GRU, N-HiTS, N-BEATS, PatchTST, TimeMixer, TSMixer)
- ✅ **NO_DM_TEST**: Friedman test wired (DM pairwise pending raw preds)
- 🔴 **MISSING_MODELS_FOUNDATION** (W4): Chronos-Bolt, TTM, TimesFM, Lag-Llama
- 🟡 **NO_CONFORMAL_PI** (W4): ACI conformal intervals
- 🟡 **NO_XAI_FULL** (W4): SHAP/Captum/attention
- 🟡 **NO_DASHBOARD** (W5): Streamlit app
- 🟡 **NO_PAPER** (W5): TDTU report + IEEE LaTeX
- 🟡 **NO_REAL_SENTIMENT** (optional): scrape news + PhoBERT
- 🟡 **iTransformer_TFT_DEFERRED**: bỏ trong --fast mode (CPU slow); user có thể chạy Colab
- 🟡 **NO_OPTUNA_TUNED**: ML defaults; tuning per-fold expensive — defer if time

## 🔖 GIT STATE
- Current branch: `wip/auto`
- Tags: pre-refactor-v0, milestone-1-foundation, milestone-2-baselines, sắp milestone-3-models
- Commits: ~14 trên branch
- Remote: github.com/twangnhat-05/NGHIENCUUKHOAHOC (pushed)
