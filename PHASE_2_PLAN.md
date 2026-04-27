# 🚀 PHASE 2 PLAN — Gold Price Forecasting Project

> **Pre-condition**: Phase 1 (M1-M5) completed. Repo ở `13b9ebd` trên main.
> **Owner**: WangNhat (TDTU NCKH 2025-2026)
> **Estimated**: 2-4 tuần (tùy scope user chọn)

---

## A. Scientific extensions (cải thiện kết quả + paper)

### A1. Real Sentiment Pipeline — PhoBERT finetuned ⭐ HIGH IMPACT
**Mô tả**: Implement scrape news từ CafeF/VnExpress + zero-shot mDeBERTa hoặc finetune PhoBERT trên VN financial sentiment.

**Effort**: 2-3 ngày
**Impact**:
- Boost MAPE thêm 0.05-0.2% (modest nhưng consistent)
- Strong novelty cho paper: "first VN-language sentiment for gold forecasting"
- Hoàn thành đề cương proposal

**Steps**:
1. Implement `src/features/sentiment_scrape.py` — Selenium/Playwright cho CafeF (anti-bot)
2. Cache headlines vào `data/external/news_headlines.parquet` (date, title, body, source)
3. Score bằng `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` zero-shot 3-class (positive/neutral/negative) — CPU OK
4. (Optional) Finetune `vinai/phobert-base-v2` trên 500-1000 labeled headlines (manual labeling 4-6 giờ)
5. Re-run features build → leaderboard với sentiment thật
6. So sánh MAPE before/after — báo cáo trong paper

**Pitfalls**:
- CafeF có anti-bot (Cloudflare) → cần rotating proxy hoặc dùng Web Archive
- Sentiment lag effect: news có thể leak future nếu publish time wrong → kiểm tra timezone

---

### A2. Fine-tune Chronos-Bolt trên SJC ⭐ NOVELTY
**Mô tả**: Fine-tune pretrained Chronos-Bolt-Small trên train fold của SJC để boost zero-shot 3.07% → có thể 1-2%.

**Effort**: 2 ngày
**Impact**:
- Strong paper contribution: "First fine-tuning of foundation model on emerging market gold"
- Có thể vượt Ridge nếu tune đúng (ngang ML linear)

**Steps**:
1. Cài `chronos-forecasting[training]`
2. Prepare train data trong format Chronos (univariate ts long)
3. Fine-tune script với `transformers.Trainer`:
   - Model: `amazon/chronos-bolt-small`
   - LR: 1e-5
   - Epochs: 3-5 (tránh overfit)
   - Loss: WQL (Weighted Quantile Loss — Chronos default)
4. Walk-forward CV cho fine-tuned variant
5. Add vào leaderboard

**Pitfalls**:
- Fine-tune cần ≥ 8GB RAM hoặc Colab T4
- Có thể overfit — monitor val loss

---

### A3. Regime-aware Ensemble ⭐ HIGH IMPACT (paper headline)
**Mô tả**: 2024 gold rally fold gây MAPE inflate 5-10x. Build regime detector (rolling vol > threshold → "rally regime") + dynamic ensemble weights.

**Effort**: 3-4 ngày
**Impact**:
- Cải thiện đáng kể MAPE trong volatile period
- Paper headline mạnh: "Regime-aware ensemble cho emerging market gold"

**Steps**:
1. Implement `src/models/regime_detector.py`:
   - Feature: rolling 20-day std of log returns
   - 2 regime: "stable" (vol < 0.5%) vs "volatile" (vol >= 0.5%)
   - HMM hoặc threshold rule
2. Train 2 ensemble:
   - Stable regime: ElasticNet + Ridge + AutoARIMA (top 3 mode-A)
   - Volatile regime: heavy weight RollingNaive + ElasticNet (closer to actual movement)
3. At test time: detect regime → use appropriate ensemble
4. Compare với single ensemble baseline

**Pitfalls**:
- Look-ahead bias trong regime detection — fit threshold chỉ trên train fold
- Need ≥ 200 obs per regime để stable

---

### A4. Multi-horizon Joint Training (TFT, NeuralForecast)
**Mô tả**: Train 1 model output 3 horizons cùng lúc (joint loss) thay vì 3 model riêng.

**Effort**: 1-2 ngày
**Impact**: Modest improvement, tiết kiệm training time

**Steps**:
1. NeuralForecast TFT support multi-horizon natively → set `h=20`, output 20 step
2. So sánh per-horizon MAPE vs separate models

---

### A5. Conformal calibration cho top ensemble ⭐
**Mô tả**: Hiện ACI demo trên 1 model 1 fold. Mở rộng cho top-3 ensemble + report coverage trên ALL folds.

**Effort**: 1 ngày
**Impact**: Strong empirical contribution

---

### A6. CRPS + Quantile loss optimization
**Mô tả**: Tối ưu CRPS thay vì MSE/MAE → better probabilistic forecast.

**Effort**: 2 ngày (cần custom loss cho LightGBM/PyTorch)

---

### A7. Add TimesFM, Lag-Llama, Moirai-MoE
**Mô tả**: Bổ sung 3 foundation models để bench đầy đủ "foundation family".

**Effort**: 1-2 ngày mỗi model

---

## B. Engineering polish

### B1. CI/CD với GitHub Actions ⭐
**Effort**: 1 ngày
**Impact**: Repo professional, auto-test mỗi push

**Setup**:
- `.github/workflows/test.yml`: pytest + ruff + black check
- Badge trong README
- Nightly data refresh job (optional)

### B2. Pre-commit hooks
**Effort**: 0.5 ngày
- ruff format + check
- mypy (loose)
- pytest fast tests

### B3. Dockerfile + docker-compose
**Effort**: 1 ngày
- Image cho training (có torch + neuralforecast)
- Image cho serving (chỉ Streamlit + FastAPI deps)
- Multi-stage build < 1GB

### B4. Test coverage 90%+
**Effort**: 2 ngày
- Add integration tests cho fetch + merge + features pipeline
- Property-based tests (hypothesis) cho metrics

### B5. Optuna full tuning
**Effort**: 1-2 ngày
- 200 trials per model per horizon
- MLflow tracking
- Pareto frontier analysis (MAPE vs runtime)

### B6. Type hints 100%
**Effort**: 1 ngày — `mypy --strict`

---

## C. Productization

### C1. Mobile PWA ⭐ (cho user thực tế dùng)
**Effort**: 3-5 ngày
**Stack**: Streamlit Cloud + Streamlit-PWA add-on, hoặc rebuild với Next.js + FastAPI backend

**Features**:
- Daily push notification giá vàng dự báo
- Alert khi giá vượt ngưỡng
- Lịch sử dự báo của user

### C2. Telegram bot
**Effort**: 2 ngày
**Stack**: python-telegram-bot, deploy trên Render free
- Command `/predict 1` → response prediction
- Daily auto-post vào group/channel

### C3. API auth + rate limiting
**Effort**: 1 ngày
- API key system (FastAPI + slowapi)
- Cho commercial users

### C4. Monitoring (Grafana + Prometheus)
**Effort**: 2 ngày
- Track API latency, error rate
- Track prediction drift

### C5. Auto-retrain weekly
**Effort**: 2 ngày
- Cronjob refresh data + retrain top 3 models
- Email alert nếu MAPE deteriorate > 20%

---

## D. Academic outputs (nếu user muốn publish)

### D1. Submit IEEE conference
- **RIVF 2026** (https://rivf.tdtu.edu.vn): VN top conference, deadline ~tháng 6 hàng năm
- **SoICT** (https://soict.org): VN, deadline ~tháng 8
- **KSE 2026**: VN/international hybrid
- **ICONIP 2026**: international (cần English)

**Effort**: 1-2 tuần (revise paper draft + reviewer responses)

### D2. Submit Scopus journal Q3-Q4
- **Resources Policy** (Elsevier, Q1)
- **Asian Economic Review** (Q3)
- **Heliyon** (Q1, broad scope)
- Effort: 1-2 tháng (extensive revision)

### D3. Open dataset on Hugging Face
- Upload SJC + macro features parquet → HF Datasets
- Citation: paper + dataset DOI

---

## E. RECOMMENDED PHASE 2 ROADMAP

### Option 1 — "Boost Paper" (2 tuần)
1. A1 Real sentiment (3 ngày)
2. A3 Regime-aware ensemble (3 ngày)
3. A5 Conformal full (1 ngày)
4. D1 Submit RIVF (1 tuần)

→ Paper với 3 contributions mới, có thể vào conference Q1.

### Option 2 — "Production Ready" (2 tuần)
1. B1 CI/CD (1 ngày)
2. B3 Docker (1 ngày)
3. C2 Telegram bot (2 ngày)
4. C5 Auto-retrain (2 ngày)
5. C1 Mobile PWA (1 tuần)

→ Hệ thống deployed thực tế.

### Option 3 — "Both" (4 tuần) ⭐ recommend nếu user có thời gian
- Tuần 1-2: Option 1 (paper boost)
- Tuần 3-4: Option 2 (production)

→ Hoàn thiện cả academic + practical.

---

## F. PRE-PHASE-2 CHECKLIST

Trước khi bắt đầu Phase 2:

- [ ] Verify Phase 1 still works:
  ```bash
  cd D:\WangNhat\Study\NCKH
  pytest tests/ -q                           # → 47 passed
  streamlit run app/streamlit_app.py          # → HTTP 200
  python -m src.data.refresh                  # data fresh
  ```
- [ ] User confirm Phase 2 option (1, 2, hoặc 3)
- [ ] User grant any new permissions (e.g. C2 cần Telegram bot token)
- [ ] Branch: `git checkout -b claude/phase-2-execution`
- [ ] Tag rollback: `git tag pre-phase-2 milestone-5-delivery` (đã có sẵn M5)

---

## G. RISK MATRIX cho Phase 2

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CafeF anti-bot block scraping | High | Medium | Use mDeBERTa zero-shot trên ít headlines manual paste |
| Chronos fine-tune cần Colab > 1 phiên | Medium | Low | Save checkpoint mỗi epoch; resume nếu disconnect |
| Regime detector overfit vì ít regime | Medium | High | Use simple threshold rule trước HMM |
| GitHub Actions free quota cạn | Low | Low | 2000 phút/tháng — đủ cho personal repo |
| Foundation model weights > 5GB | Low | Medium | Ưu tiên TimesFM 200M trước Moirai-MoE |
| Paper rejection ở RIVF | Medium | Low | Resubmit SoICT hoặc journal |

---

## H. SUCCESS METRICS PHASE 2

Cho biết Phase 2 thành công:

### Scientific
- [ ] MAPE h=1 < 0.50% (cải thiện từ 0.63%)
- [ ] DA > 55% (current ~52%)
- [ ] Conformal coverage 90% target → actual > 88%
- [ ] Paper accepted ít nhất 1 conference VN

### Engineering
- [ ] Test coverage > 90%
- [ ] CI/CD green badge trên README
- [ ] Docker image < 1GB
- [ ] Streamlit Cloud deploy uptime > 99%

### Product
- [ ] Telegram bot có ≥ 50 subscribers
- [ ] Daily forecast email cho ≥ 10 users
- [ ] PWA install rate > 30% từ visitor

---

🤖 *Generated by Claude Opus 4.7 — handover document for next session.*
