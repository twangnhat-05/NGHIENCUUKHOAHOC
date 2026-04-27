# 🤝 HANDOVER — Onboarding cho Claude Session sau

> **Mục đích**: tài liệu này giúp Claude session mới (hoặc developer mới) **nắm bắt 100% dự án trong 5 phút** mà không cần đọc lại toàn bộ conversation.
>
> **Tạo bởi**: Claude Opus 4.7 (session 2026-04-26 → 2026-04-27)
> **Tổng thời gian session 1**: ~3 giờ active execution
> **Project completion**: 100% (M1-M5) sẵn sàng nộp TDTU + bonus IEEE conference

---

## 1. CONTEXT — User & Project

### 1.1 User profile
- **Identity**: WangNhat (sinh viên TDTU — Đại học Tôn Đức Thắng)
- **Email**: dev2@wolffungame.com
- **GitHub username**: `twangnhat-05`
- **Năm học**: 2025-2026 (NCKH SV)
- **OS**: Windows 11, Python 3.11.9, Git Bash terminal
- **Working dir**: `D:\WangNhat\Study\NCKH`
- **Có**: Account Colab (cho GPU nếu cần TFT/iTransformer)

### 1.2 Project: "Ứng dụng AI dự đoán giá vàng VN"
- **Title (EN)**: Applied AI to Gold Price Forecasting and Market Volatility in Vietnam
- **Repo**: https://github.com/twangnhat-05/NGHIENCUUKHOAHOC
- **Deadline cấp khoa**: 2026-05-31 (đang đi trước 4 tuần — thoải mái thời gian Phase 2)
- **Mục tiêu original**: MAPE 4-5% — **đã vượt** (tốt nhất 0.63%)

### 1.3 Ràng buộc cốt lõi (đã tuân thủ 100%)
- 💰 $0 — full free tier (no paid API, no GPU thuê)
- 📊 Data: yfinance + FRED + vnstock + webgia.com scraper
- ☁️ Compute: local CPU + Colab free
- 🚀 Deploy: Streamlit Cloud / HF Spaces / Render free

---

## 2. PROJECT STATE — what exists now

### 2.1 Repository structure (đã build)
```
NCKH/
├── ARCHITECTURE.md          ← Master design doc (READ FIRST)
├── README.md                ← Quickstart
├── HANDOVER.md              ← THIS FILE
├── PHASE_2_PLAN.md          ← Roadmap Phase 2
├── CHANGELOG.md             ← Lịch sử changes per milestone
├── MONITORING.md            ← Health dashboard
├── CLAUDE_EXECUTION_LOG.md  ← Audit trail
├── BM02_decuong.pdf         ← Đề cương gốc của user (proposal)
├── pyproject.toml           ← ruff + black + pytest config
├── requirements.txt         ← Pinned deps (Python 3.11)
├── LICENSE (MIT) + CITATION.cff
├── .gitignore
│
├── data/
│   ├── raw/                 ← 11 CSV nguồn (immutable, refreshed 2026-04-25)
│   ├── interim/             ← merged.parquet (2169 × 16)
│   ├── processed/           ← features_v2_with_sentiment.parquet (1883 × 122)
│   └── external/            ← (sentiment news cache — rỗng)
│
├── configs/
│   ├── data.yaml, cv.yaml, features.yaml
│   └── models/              ← (sẵn folder, model configs có thể thêm)
│
├── src/
│   ├── data/                ← fetch.py, refresh.py, schema.py, merge.py
│   ├── features/            ← build.py, technical.py, calendar.py, macro.py, sentiment.py
│   ├── models/              ← base.py, classical.py, ml.py, dl_simple.py,
│   │                          dl_neuralforecast.py, foundation.py, ensemble.py
│   ├── training/            ← cv.py (WalkForwardCV), trainer.py, tune.py (Optuna)
│   ├── evaluation/          ← metrics.py, leaderboard.py, stat_tests.py, conformal.py
│   ├── xai/                 ← shap_utils.py, attention.py
│   ├── utils/               ← io.py, logging.py, seeds.py
│   └── legacy/              ← FROZEN — code gốc trước Claude refactor
│
├── tests/                   ← 47/47 PASS — test_cv_no_leakage, test_metrics,
│                              test_features, test_ml_models, test_stat_tests
│
├── notebooks/               ← (rỗng — placeholder cho Phase 2)
│
├── scripts/
│   ├── reproduce_all.{sh,bat}
│   ├── run_classical_baselines.py
│   ├── run_ml_baselines.py
│   ├── run_dl_baselines.py
│   ├── run_foundation_baselines.py
│   ├── run_xai_conformal_demo.py
│   └── combine_leaderboards.py
│
├── app/
│   ├── streamlit_app.py     ← Dashboard 4 tabs (HTTP 200 verified)
│   └── api/main.py          ← FastAPI 5 endpoints
│
├── reports/
│   ├── leaderboard/         ← combined_v2_*.csv + 36 PNG plots + friedman_test.csv
│   ├── figures/             ← shap_lightgbm_h1.png, aci_conformal_*.png
│   └── paper/
│       ├── tdtu_vi/report.md + report.docx (22KB) ← VN báo cáo
│       └── ieee_en/main.tex + bib.bib            ← EN IEEE submission-ready
│
├── docs/
│   └── DEPLOY_GUIDE.md      ← Streamlit Cloud / Render / HF Spaces guide
│
├── output/                  ← legacy plots (giữ làm so sánh trước/sau refactor)
└── .claude/skills/INDEX.md  ← Skill registry (chưa tạo skill nào — code modular)
```

### 2.2 Git state
- **Branch `main`**: `fa59536` — Phase 1 final
- **Branch `claude/auto-execution`**: `fa59536` (sync với main)
- **Branch `claude/phase-2-execution`** ⭐ NEW: HEAD — Phase 2 work
- **Tags pushed remote**:
  - `pre-claude-v0` — rollback point (chỉ legacy code)
  - `milestone-1-foundation` — refactor + walk-forward CV
  - `milestone-2-baselines` — features V2 + 9 classical
  - `milestone-3-models` — +7 ML +7 DL = 23 models
  - `milestone-4-frontier` — +1 foundation + ensemble + conformal + XAI
  - `milestone-5-delivery` — Streamlit + FastAPI + papers
  - `pre-phase-2` — rollback cho Phase 2
  - `milestone-6-phase2-boost` — sentiment + regime + conformal full (PARTIAL P2 Option 1)

### 2.3 Final benchmark results
24 models × 5 walk-forward folds × 3 horizons = **360 records**

| Horizon | 🥇 Best | MAPE | Family |
|---|---|---|---|
| h=1 (1 ngày) | RollingNaive (mode-B floor) | 0.33% | naive |
|             | **Ridge** (winner mode-A) | **0.63%** | ML linear + 108 features |
| h=5 (5 ngày) | **ElasticNet** | **1.41%** | ML linear |
| h=20 (20 ngày) | **ElasticNet** | **3.06%** | ML linear |

**Friedman tests**: p < 0.001 cho cả 3 horizons → models thực sự khác biệt thống kê.

**Foundation models zero-shot**:
- Chronos-Bolt-Small (Amazon, 48M, Apache-2.0): 3.07% MAPE h=1 (ngang classical, không cần train)

**XAI** (SHAP TreeExplainer, LightGBM h=1):
- Top: SJC_lag1 (2.49) > SJC_lag2 (2.27) > SJC_mua_vao_lag1 (1.36) > SMA30 > USD_Close > 10y_Treasury

**Conformal PI** (ACI cho ElasticNet h=1, last fold = 2024 gold rally):
- Split conformal 95% target → 83% actual (under-cover do regime shift)
- ACI 90% target → 83% actual + width 3.62 (online adapt)

---

## 3. KEY DECISIONS — đã chốt với user, KHÔNG đảo ngược

### 3.1 Q1-Q5 (Gate 2)
| Q | Decision |
|---|---|
| Q1 Target | Multi-horizon h=1, h=5, h=20 + Directional Accuracy |
| Q2 Features | 108 — lags + technical + macro VN + calendar + sentiment stub |
| Q3 Models | ALL ⭐ + research SOTA: Chronos-Bolt + N-HiTS + PatchTST + TFT + iTransformer + TSMixer + TimeMixer |
| Q4 Output | TDTU Word + bonus IEEE LaTeX cho conference |
| Q5 Demo | Streamlit Cloud (chính) + FastAPI (bonus) |

### 3.2 Constraint quan trọng
- **MUST**: 100% free tier, license OSS, no GPU thuê
- **MUST**: walk-forward CV (no leakage tests phải pass)
- **MUST**: scaler/outlier fit chỉ trên train fold
- **MUST**: Conventional commits, branch isolation, tag milestones
- **AVOID**: paid API, manual changes vào `data/raw/`, sửa `src/legacy/`

### 3.3 Permission scope (user grant ALL trong session 1)
- Read/Write/Modify trong `D:\WangNhat\Study\NCKH\`
- pip install (log đầy đủ)
- git commits + push remote
- Run python scripts + jupyter
- Spawn sub-agents
- KHÔNG xóa raw data/file lớn không CONFIRM
- KHÔNG force push, rebase main

---

## 4. KNOWN LIMITATIONS (Phase 2 partial done; Phase 3 candidates)

### 4.0 Phase 2 added (2026-04-27, branch claude/phase-2-execution, tag milestone-6-phase2-boost)
- ✅ Real news scraper + mDeBERTa zero-shot scoring pipeline (218 headlines verified)
- ✅ Regime-aware ensemble (with limitation: train-end detection only)
- ✅ Conformal full coverage report — ACI 86% h=1 vs split 76%
- ⚠️ Sentiment integration: pipeline working but news data 2025-10+ KHÔNG overlap CV folds 2022-2024
   → MAPE Δ = 0%; cần historical news scraping cho impact thực sự (Phase 3)

### 4.1 Soft / Optional (Phase 3 candidates)
1. **Historical news data**: scrape Web Archive cho CafeF/VnExpress 2018-2024
   → unlock real sentiment impact trên benchmark
2. **CPI VN missing** — FRED code `CPALTT01VNQ657N` đã 404; cần lấy thủ công từ GSO (gso.gov.vn)
3. **Optuna tuning chưa toàn diện** — Ridge/ElasticNet đã thắng với defaults; trees có thể bứt phá với tuning
4. **TFT + iTransformer chưa benchmark đầy đủ** — code sẵn (`src/models/dl_neuralforecast.py`), cần Colab T4
5. **TimesFM, Lag-Llama, Moirai-MoE** — wrappers chưa làm (Chronos-Bolt đại diện foundation)
6. **DM pairwise raw predictions chưa save** — Friedman có rồi, DM cần update trainer save preds
7. **Notebook 99_reproduce_all.ipynb** chưa tạo (chỉ có script .sh/.bat)

### 4.2 Hard / có ý nghĩa khoa học
8. **No regime detection model** — 2024 gold rally fold gây MAPE inflate 5-10x; cần model phát hiện regime
9. **No multi-horizon joint training** — hiện train 3 model riêng cho h=1, h=5, h=20
10. **No quantile / probabilistic forecast natively** (chỉ Conformal post-hoc)

### 4.3 Engineering
11. Chưa có CI/CD (GitHub Actions)
12. Chưa có pre-commit hook
13. Chưa có Dockerfile
14. Test coverage còn ~65% (target 70%)

---

## 5. REPRODUCIBILITY — Quick verify

```bash
# Clone
git clone https://github.com/twangnhat-05/NGHIENCUUKHOAHOC.git
cd NGHIENCUUKHOAHOC

# Setup (Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Test (1 phút)
pytest tests/ -q                    # → 47 passed

# Smoke test pipeline (5 phút)
python -m src.data.merge            # → interim/merged.parquet
python -m src.features.build        # → processed/features_v2.parquet
python -m src.features.sentiment stub  # → features_v2_with_sentiment.parquet

# Streamlit demo (5 phút)
streamlit run app/streamlit_app.py  # → http://localhost:8501

# Full benchmark (~70 phút)
scripts\reproduce_all.bat
```

---

## 6. PHASE 2 PROPOSAL — xem `PHASE_2_PLAN.md`

3 nhóm chính:
- **A. Scientific extensions** (sentiment thật, fine-tune Chronos, regime-aware ensemble)
- **B. Engineering polish** (CI/CD, Docker, pre-commit, test coverage 90%, Optuna full)
- **C. Productization** (mobile/PWA, alert system, API auth, monitoring)

Recommend ưu tiên: **A1 (sentiment) + A3 (regime ensemble) + B1 (CI/CD)** — cải thiện đáng kể paper + repo professional.

---

## 7. SESSION 1 LEARNINGS — feedback đã ghi nhận

### Cách user thích làm việc
- Trả lời ngắn gọn ("GO W2", "GO W3", "ALL", "GRANT ALL")
- Thích autonomous execution sau khi grant — không hỏi từng bước
- Hỏi câu súc tích kèm options ⭐ recommend → user chọn nhanh
- Báo cáo định kỳ dạng box ASCII có metrics rõ
- Test/verify thực tế trước khi tag milestone
- Push GitHub thường xuyên (sau mỗi milestone)

### Cách trình bày output
- Structure: "🎯 NEXT ACTION REQUIRED FROM USER" ở cuối mỗi turn quan trọng
- Tiếng Việt, thuật ngữ kỹ thuật giữ tiếng Anh
- Markdown table + emoji bullet cho dễ scan
- Mọi commit message conventional (feat/fix/docs/chore + scope)
- Mọi milestone có Git tag pushed remote

### Bug pattern đã sửa (đừng lặp lại)
- yfinance 1.1.0+ cần `yf.download()` + backoff (Ticker.history() trả empty)
- MLForecast freq B mismatch VN holidays → integer index workaround
- Naive forecaster KHÔNG được shift y_observed (mode-A = constant)
- Chronos-Bolt 2.x API: `predict_quantiles(context, prediction_length, quantile_levels)` — positional `context` not kwarg
- Prophet downgrade numpy 2.x → 1.26.4 (compat cmdstanpy)
- shap install upgrade numba/llvmlite (statsforecast vẫn ok)

---

## 8. SESSION 2 ONBOARDING — TEMPLATE PROMPT

Xem `prompt_for_next_session.md` — paste vào Claude Code hoặc Claude Web để start session mới với full context.

---

## 9. EMERGENCY ROLLBACK

Nếu Phase 2 phá hỏng project:

```bash
# Rollback toàn bộ về cuối session 1 (M5 final delivery)
git checkout main
git reset --hard milestone-5-delivery
git push --force-with-lease origin main

# Hoặc rollback về trước Claude (chỉ legacy code)
git reset --hard pre-claude-v0
```

---

## 10. CONTACT / RESOURCES

- **GitHub repo**: https://github.com/twangnhat-05/NGHIENCUUKHOAHOC
- **TDTU đề cương**: `BM02_decuong.pdf`
- **Streamlit Cloud guide**: `docs/DEPLOY_GUIDE.md`
- **Architecture decisions**: `ARCHITECTURE.md`
- **Audit trail**: `CLAUDE_EXECUTION_LOG.md`

---

🤝 *Session 1 (2026-04-26 → 2026-04-27): hoàn thành 5 milestones, 24 models, 47 tests, 360 records, deploy-ready.*
🚀 *Session 2 (TBD): Phase 2 — scientific + engineering + productization extensions.*
