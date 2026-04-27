# 🏆 FINAL SUMMARY — Vietnamese Gold Price Forecasting (TDTU NCKH 2025-2026)

> **Project**: Ứng dụng AI dự đoán giá vàng và phân tích biến động thị trường VN
> **Owner**: WangNhat (TDTU)
> **Repo**: https://github.com/twangnhat-05/NGHIENCUUKHOAHOC
> **Status**: ✅ HOÀN THÀNH — 9 milestones (Phase 1-5)
> **Deadline**: 2026-05-31 — đã đi trước **5 tuần**

---

## 📊 Final Stats

| Item | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | TOTAL |
|---|---|---|---|---|---|---|
| Models | 24 | +1 (regime) | +1 (rolling) | +1 (TimesFM) | +1 (Lag-Llama) | **27** |
| Records benchmarked | 360 | +30 | +30 | - | - | **420+** |
| Tests | 47 | 47 | 47 | 47 | 47 | **47/47 PASS** |
| Git tags | 5 (M1-M5) | +1 (M6) | +1 (M7) | +1 (M8) | +1 (M9) | **9 milestones** |
| Branches | 2 | +1 (P2) | +1 (P3) | +1 (P4) | +1 (P5) | **6** |
| Sub-tasks | 41 | 7 | 5 | 5 | 4 | **62** |
| Active time | ~3h | ~30min | ~30min | ~25min | ~15min | **~5h** |
| Token consumption | ~110k | +30k | +20k | +20k | +20k | **~200k** |
| Disk usage | 80MB | +10MB | +5MB | +5MB | +5MB | **~105MB** |
| Free tier quota | 100% | 100% | 100% | 100% | 100% | **100%** |

---

## 🏆 Key Findings

### Top performers
| Horizon | Best mode-A | MAPE | Best mode-B | MAPE |
|---|---|---|---|---|
| h=1 | SeasonalNaive | 2.91% | **Ridge** ⭐ | **0.63%** |
| h=5 | SeasonalNaive | 3.03% | **ElasticNet** ⭐ | **1.41%** |
| h=20 | SeasonalNaive | 3.49% | **ElasticNet** ⭐ | **3.06%** |

→ **Đã vượt mục tiêu proposal MAPE 4-5% lên gần 8x.**

### Foundation models (zero-shot)
- Chronos-Bolt-Small: **3.07% MAPE h=1** (no training, competitive với classical)
- TimesFM 2.0 wrapper ready (Linux/Colab)
- Lag-Llama wrapper ready (cần clone repo)

### Statistical evidence
- **Friedman test** p < 0.001 cho cả 3 horizons → models thực sự khác biệt
- **ACI conformal vs split conformal**: +10pp coverage tại h=1, h=5 (45 evidence points)
- **Regime detector**: Phase 3 rolling re-detect → 87/90 rows volatile cho 2024 rally fold (vs 0 trước)

### Top SHAP features (LightGBM h=1)
1. SJC_ban_ra_lag1 (2.49) — yesterday's price
2. SJC_ban_ra_lag2 (2.27)
3. SJC_mua_vao_lag1 (1.36)
4. SMA(30) of SJC (0.79)
5. USD_Close (0.14), 10y_Treasury (0.12), GLD_Close (0.10)

---

## 🚀 Production-Ready Deliverables

| Item | Status | Endpoint/Command |
|---|---|---|
| Streamlit dashboard | ✅ HTTP 200 | `streamlit run app/streamlit_app.py` |
| FastAPI server | ✅ 5 endpoints | `uvicorn app.api.main:app` |
| Telegram bot | ✅ 6 commands | `python app/telegram_bot.py` (need TOKEN) |
| Docker images | ✅ Multi-stage | `docker compose up -d` |
| GitHub Actions CI | ✅ pytest + ruff | trigger on push main + PR |
| Auto-retrain weekly | ✅ Cron + Task Scheduler | `bash scripts/retrain_weekly.sh` |
| TDTU report | ✅ Word DOCX | `reports/paper/tdtu_vi/report.docx` |
| IEEE LaTeX paper | ✅ Submission-ready | `reports/paper/ieee_en/main.tex` |

---

## 🔖 Git State (final)

```
Branch main: latest commit
Tags pushed:
├── pre-claude-v0       (rollback to legacy)
├── milestone-1-foundation
├── milestone-2-baselines
├── milestone-3-models      (24 models, Friedman p<0.001)
├── milestone-4-frontier    (foundation, conformal, XAI)
├── milestone-5-delivery    (Streamlit + FastAPI + papers)
├── pre-phase-2 / milestone-6-phase2-boost   (sentiment + ACI full)
├── pre-phase-3 / milestone-7-phase3         (rolling regime + CI/CD + Telegram bot)
├── pre-phase-4 / milestone-8-phase4         (TimesFM + Docker + auto-retrain)
└── pre-phase-5 / milestone-9-final          (papers updated + Lag-Llama + summary)
```

---

## 📚 Document Map

| File | Purpose |
|---|---|
| `README.md` | Quickstart + badges |
| `ARCHITECTURE.md` | Master design doc |
| `HANDOVER.md` | Onboarding cho session sau (after Phase 1) |
| `PHASE_2_PLAN.md` | Roadmap Phase 2 candidates |
| `prompt_for_next_session.md` | Prompt template Claude session mới |
| `docs/SUPER_PROMPT_SESSION_1.md` | Original super-prompt archive |
| `docs/HYBRID_PROMPT_PHASE_2.md` | Recommended hybrid prompt |
| `docs/DEPLOY_GUIDE.md` | Streamlit/Render/HF Spaces guide |
| `CHANGELOG.md` | Per-milestone release notes |
| `MONITORING.md` | Health dashboard |
| `CLAUDE_EXECUTION_LOG.md` | Audit trail |
| `FINAL_SUMMARY.md` | THIS FILE — final overview |
| `BM02_decuong.pdf` | TDTU đề cương gốc |

---

## 📄 Academic outputs

### TDTU cấp khoa (sẵn sàng nộp)
- `reports/paper/tdtu_vi/report.md` — Markdown 14 trang, 8 sections
- `reports/paper/tdtu_vi/report.docx` — Word converted (22KB)
- Hành động user cần làm:
  1. Mở DOCX trong Word
  2. Apply template TDTU BM01/BM02
  3. Bổ sung bìa, mục lục, ký GVHD/SV
  4. Nộp

### IEEE conference (bonus, optional)
- `reports/paper/ieee_en/main.tex` — LaTeX 2-column conference format
- `reports/paper/ieee_en/bib.bib` — 16 references
- Hành động user cần làm:
  1. Upload main.tex + bib.bib lên Overleaf (free)
  2. Compile pdflatex → PDF
  3. Submit RIVF / SoICT / KSE / ICONIP (deadline ~tháng 6/2026)

### Submission helper (Phase 5 mới)
- Khi user sẵn sàng submit → Claude session sau có thể hỗ trợ:
  - Cover letter cho editor
  - Reviewer response template
  - Camera-ready revision
  - Conference presentation slides

---

## 🎯 Future Work (Phase 6+ candidates)

### Scientific
1. Historical news scraping (Web Archive 2018-2024) → real sentiment impact
2. Fine-tune Chronos-Bolt trên SJC → vượt zero-shot
3. Multi-asset ensemble (SJC + gold quốc tế + USD/VND)
4. Markov regime-switching thay simple threshold

### Engineering
5. Hugging Face Spaces deploy (16GB RAM free)
6. Cron-job.org keep-alive cho Render free tier
7. PostgreSQL backend cho leaderboard history
8. Test coverage 90%+

### Product
9. Mobile PWA (Streamlit-PWA add-on)
10. Email daily digest cho subscribers
11. Premium tier với probabilistic forecast (Conformal full intervals)

---

## 💡 Lessons Learned

### Technical
- Linear regularized + engineered features dominate ML/DL trên dataset nhỏ (~1000 obs)
- Foundation models zero-shot competitive với classical → useful khi cold start
- Walk-forward CV + rolling regime detection critical cho volatile asset
- ACI conformal > split conformal khi distribution shift (regime change)
- yfinance 1.1.0+ rate-limit aggressive → cần `yf.download()` + backoff
- Multilingual zero-shot (mDeBERTa) tốt cho VN news mà không cần label data

### Process
- Autonomous execution với clear safety guardrails > micromanage
- Conventional commits + tag every milestone = professional repo
- Memory + handover docs critical cho session continuity
- Free tier chỉ cần creative engineering, không cần budget

---

## 🙏 Acknowledgments

- **TDTU NCKH SV program** 2025-2026
- **Open-source community**: Nixtla (statsforecast/neuralforecast/mlforecast), Amazon (Chronos), Google (TimesFM), IBM (TTM), Salesforce (Moirai), HuggingFace
- **Co-architected with**: Claude Opus 4.7 (Anthropic) — pipeline design, code, statistical analysis, documentation

---

🎓 **Project COMPLETE — Ready cho TDTU cấp khoa + IEEE conference submission.** Chúc bạn nghiệm thu thành công!

🤖 *Generated by Claude Opus 4.7 — final summary across Phase 1-5 (2026-04-26 → 2026-04-27).*
