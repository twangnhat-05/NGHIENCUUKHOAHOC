# 🚀 SESSION 2 KICKOFF — Làm việc Y CHANG session 1

> **File này là file DUY NHẤT bạn cần khi mở session Claude mới.**
> Copy 1 đoạn prompt → paste vào Claude Code → bắt đầu Phase 7.

---

## ⚡ QUICK START (3 bước, 2 phút)

### Bước 1 — Mở session Claude mới
```bash
cd D:\WangNhat\Study\NCKH
claude
```
(Nếu Claude Web: vào claude.ai → New chat)

### Bước 2 — Paste KICKOFF PROMPT bên dưới (giữa `---START---` và `---END---`)

### Bước 3 — Đợi Claude xác nhận, rồi gõ `"GO PHASE 7"` (hoặc option bạn chọn)

---

## 📋 KICKOFF PROMPT (copy nguyên block)

### ---START---

```
Bạn là Principal ML Architect tiếp nhận dự án "Mô hình dự đoán giá vàng VN"
của tôi (TDTU NCKH SV 2025-2026). Phiên trước đã hoàn tất Phase 1-6 (M1-M10).

ĐÂY LÀ PHIÊN MỚI — BẠN KHÔNG CÓ CONTEXT CỦA SESSION TRƯỚC,
NHƯNG MEMORY ĐÃ SAVE SẴN PROJECT STATE.

═══════════════════════════════════════════════════════════════════
VAI TRÒ — Architect chủ động, không phải assistant trả lời câu hỏi
═══════════════════════════════════════════════════════════════════
- Đưa ra quyết định kỹ thuật có căn cứ (cite paper / best practice)
- Autonomous execution sau khi tôi grant permission — không hỏi từng bước
- Trả lời súc tích, recommendation kèm ⭐
- Báo cáo dạng box ASCII có metric

RÀNG BUỘC CỐT LÕI (giữ nguyên session 1):
- 💰 100% FREE TIER (không paid API, không GPU thuê)
- 📊 Data: yfinance + FRED + vnstock + scrape webgia
- 🔁 Walk-forward CV (no leakage, refit per fold)
- 📝 Conventional commits + tag mỗi milestone + push GitHub
- 🌿 Branch isolation: làm trên `claude/phase-N-execution`

SAFETY GUARDRAILS (vi phạm = STOP NGAY):
1. BACKUP trước khi sửa file > 50 lines
2. Branch isolation — KHÔNG commit main không lý do
3. Atomic commits với scope rõ
4. NO destructive ops (rm/drop/force-push) không CONFIRM
5. VALIDATE sau mỗi code change (test + smoke check)
6. STOP nếu fail × 3, NaN loss, data leakage
7. AUDIT LOG: update CLAUDE_EXECUTION_LOG.md

AUTONOMY 3 LEVELS:
🟢 AUTO: read, run tests, write code, commit local
🟡 NOTIFY: install package, train > 30 phút, refactor > 100 lines
🔴 CONFIRM: xóa, force push, deploy, change architecture

WORKFLOW PHASE 6 cho mỗi task:
1. CHECKPOINT (git tag) → 2. CLASSIFY (AUTO/NOTIFY/CONFIRM)
→ 3. EXECUTE → 4. VALIDATE → 5. PASS commit / FAIL revert+retry max 2
→ 6. REPORT định kỳ (sau milestone)

OUTPUT FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏛️ ML ARCHITECT - GOLD PHASE N
📍 Current task: [task]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[content]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 NEXT ACTION REQUIRED FROM USER:
[1 câu]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CÁCH USER TRẢ LỜI (đã quen):
- "GRANT ALL" / "OK" / "GO" → cấp permission, tiếp tục
- "GO PHASE N" → bắt đầu phase N
- "ALL ⭐" → chấp nhận tất cả recommendations
- "STOP" / "PAUSE" → dừng
- "ROLLBACK" → revert tag/commit gần nhất

KHÔNG ĐƯỢC:
❌ Recommend giải pháp paid
❌ Tự ý xóa data/raw, sửa src/legacy/
❌ Force push, rebase main
❌ Skip walk-forward CV
❌ Hỏi 20 câu cùng lúc — max 5/lượt với ⭐ recommend

═══════════════════════════════════════════════════════════════════
BẮT ĐẦU NGAY:
1. Đọc tuần tự (bắt buộc):
   - HANDOVER.md           → toàn bộ context + Phase 1-6 evolution
   - FINAL_SUMMARY.md      → cumulative state + Phase 6+ candidates
   - SESSION_2_KICKOFF.md  → file này (Phase 7 options)

2. Verify state:
   cd D:\WangNhat\Study\NCKH
   git status
   git log --oneline -3
   git tag --list | sort -V
   pytest tests/ -q

3. Báo cáo (≤ 250 từ):
   - Files đã đọc
   - Git state hiện tại (branch, commit, tags)
   - Test status (47/47 PASS expected)
   - Active alerts từ MONITORING.md
   - Recommend Phase 7 option (xem 5 options trong SESSION_2_KICKOFF.md)

4. Đợi tôi reply "GO PHASE 7 OPTION X" → mở Permission Request → execute autonomous
═══════════════════════════════════════════════════════════════════
```

### ---END---

---

## 📊 PROJECT STATE QUICK REFERENCE

| Metric | Value |
|---|---|
| **Repo** | https://github.com/twangnhat-05/NGHIENCUUKHOAHOC |
| **Branch latest** | `claude/phase-6-execution` (pushed) |
| **Main branch** | `fa59536` (Phase 1 only — chưa merge P2-P6) |
| **Latest tag** | `milestone-10-pwa-email` |
| **All tags** | pre-claude-v0, M1-M10, pre-phase-{2,3,4,5,6} |
| **Models** | 27 (24 + regime + TimesFM + Lag-Llama) |
| **Records** | 420+ |
| **Tests** | 47/47 PASS |
| **Best MAPE** | h=1 Ridge 0.63%, h=5 ElasticNet 1.41%, h=20 ElasticNet 3.06% |
| **Foundation zero-shot** | Chronos-Bolt 3.07% h=1 |
| **Conformal** | ACI 86% vs split 76% (h=1, 45 evidence points) |

---

## 🎯 PHASE 7 OPTIONS — 5 hướng đi (chọn 1)

### Option 1 ⭐ "Multi-asset ensemble" (Scientific extension)
- Build joint forecasting cho SJC + gold quốc tế (GLD) + USD/VND simultaneously
- Use TFT (Temporal Fusion Transformer) trên Colab GPU
- Cross-asset attention → catch lead-lag relationships
- Effort: ~2-3 ngày | Output: paper section mới + improved MAPE

### Option 2 "Historical news scraping" (Fix Phase 2 limitation)
- Web Archive scraping cho CafeF/VnExpress 2018-2024
- Run sentiment pipeline trên historical → re-benchmark
- Expected: real impact on MAPE thay vì Δ=0% hiện tại
- Effort: ~3 ngày (anti-bot challenges) | Output: sentiment làm việc thực sự

### Option 3 "Fine-tune Chronos-Bolt" (Foundation novelty)
- Fine-tune amazon/chronos-bolt-small trên SJC train data
- Cần Colab T4 GPU (free)
- Expected: vượt zero-shot 3.07% → có thể 1-2% MAPE
- Effort: ~2 ngày | Output: novelty cho IEEE paper

### Option 4 "Production polish + deploy" (Engineering)
- Deploy live Streamlit Cloud + HF Spaces (cần user grant tokens)
- Setup monitoring (Grafana + Prometheus)
- Setup database backup (Supabase free tier)
- Effort: ~2 ngày | Output: production-grade live system

### Option 5 "IEEE submission package" (Academic)
- Polish reports/paper/ieee_en/main.tex (8 → 12 pages)
- Add benchmarks, ablation studies, conformal coverage extended discussion
- Generate publication-quality figures (300+ DPI)
- Cover letter + reviewer response template
- Effort: ~1 tuần | Output: paper sẵn submit RIVF/SoICT/KSE

---

## 🔧 NẾU MUỐN MERGE TRƯỚC

Sau khi xong Phase 7 (hoặc nếu user muốn merge ngay), Claude session sau chạy:
```bash
cd D:\WangNhat\Study\NCKH
git checkout main
git merge claude/phase-6-execution --no-ff -m "merge: Phase 2-6 production work"
git push origin main
```

---

## 📚 FILES QUAN TRỌNG (Claude session sau cần biết)

| File | Mục đích |
|---|---|
| `SESSION_2_KICKOFF.md` ⭐ | **READ FIRST** — file này |
| `HANDOVER.md` | Onboarding tổng quan Phase 1-6 |
| `FINAL_SUMMARY.md` | Stats Phase 1-6 + future work |
| `ARCHITECTURE.md` | Master design |
| `MONITORING.md` | Health dashboard |
| `CHANGELOG.md` | Per-milestone notes (M1-M10) |
| `CLAUDE_EXECUTION_LOG.md` | Audit trail |
| `PHASE_2_PLAN.md` | Original Phase 2 roadmap (now done) |
| `prompt_for_next_session.md` | Prompt template variations |
| `docs/SUPER_PROMPT_SESSION_1.md` | Original ~250-line super-prompt (full mode) |
| `docs/HYBRID_PROMPT_PHASE_2.md` | Compact 40-line hybrid prompt |
| `docs/DEPLOY_GUIDE.md` | Streamlit/Render/HF deploy |

**Memory files tự load** (qua hệ thống auto-memory Claude Code):
```
~/.claude/projects/D--WangNhat-Study-NCKH/memory/
├── MEMORY.md (index)
├── user_profile.md
├── project_context.md  ← Updated với Phase 1-6 state
├── feedback_workflow.md
├── feedback_safety.md
├── feedback_apis.md
├── reference_repo.md
└── reference_handover.md
```

---

## ⚠️ COMMON PITFALLS (đừng lặp lại)

1. **yfinance** dùng `yf.download(t, ..., progress=False, threads=False)` + backoff. KHÔNG dùng `Ticker.history()` (returns empty với 1.x).
2. **MLForecast** với VN holidays gaps → integer index `np.arange(n)` + `freq=1`.
3. **Chronos-Bolt 2.x**: `predict_quantiles(context, prediction_length, quantile_levels)` — `context` PHẢI positional.
4. **Naive forecaster mode-A**: predict trả về CONSTANT, KHÔNG shift y_observed.
5. **Prophet install** đẩy numpy về 1.26.4 — đừng upgrade lại.
6. **Streamlit port** 8501 conflict → dùng `--server.port 8502` hoặc kill PID listening.
7. **TimesFM Windows** download fail symlink → dùng Linux/Colab/Docker.
8. **mDeBERTa transformers** Keras 3 → force `framework="pt"`.

---

## 🎓 NẾU SAU PHASE 7 USER NỘP TDTU

Checklist cuối cùng trước nộp:
- [ ] Mở `reports/paper/tdtu_vi/report.docx` trong Word
- [ ] Apply BM01/BM02 template TDTU
- [ ] Bổ sung bìa, mục lục, ký GVHD/SV
- [ ] Verify all references đầy đủ (Section 8)
- [ ] Print + nộp cấp khoa
- [ ] Bonus: compile main.tex trên Overleaf → submit RIVF

---

🤝 *Tạo bởi Claude Opus 4.7 — session 1 close, sẵn sàng session 2 làm việc Y CHANG.*

🎯 **TLDR**: Mở session mới → paste KICKOFF PROMPT → Claude tự verify → bạn chọn Phase 7 option 1-5 → autonomous execute → tag M11+.
