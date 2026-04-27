# 📋 PROMPT TEMPLATE — Paste vào session Claude mới (Phase 2)

> **Cách dùng**: Copy toàn bộ section "PROMPT" bên dưới (giữa 2 dòng `---PROMPT START---` và `---PROMPT END---`) và paste vào ô chat đầu tiên của session Claude mới (Claude Code hoặc Claude Web).

---

## 🎯 Hướng dẫn ngắn gọn cho user

1. **Mở session Claude mới** (terminal `claude` hoặc claude.ai)
2. **Copy toàn bộ prompt bên dưới**
3. **Paste vào chat đầu tiên** — Claude sẽ tự đọc HANDOVER.md, ARCHITECTURE.md, MONITORING.md, CHANGELOG.md để onboard 100% project
4. **Phản hồi GO** sau khi Claude báo đã đọc xong → Claude bắt đầu Phase 2

---

## ---PROMPT START---

```
Bạn là Principal ML Architect tiếp quản dự án "Mô hình dự đoán giá vàng VN" của tôi. Đây là phiên Claude mới, bạn KHÔNG có context của session trước.

PHIÊN TRƯỚC (2026-04-26 → 2026-04-27, ~3 giờ active execution):
- Đã hoàn thành 5 milestones (M1-M5) → tag `milestone-5-delivery` trên repo
- Repo: https://github.com/twangnhat-05/NGHIENCUUKHOAHOC
- Branch `main` và `claude/auto-execution` đã sync, push remote đầy đủ
- 24 models × 5 walk-forward folds × 3 horizons = 360 records benchmarked
- 47/47 tests pass, Streamlit + FastAPI deployed-ready, TDTU + IEEE papers drafted

TÔI MUỐN BẠN LÀM TRƯỚC TIÊN (BẮT BUỘC):
1. Đọc tuần tự các file sau từ working directory `D:\WangNhat\Study\NCKH\`:
   a. `HANDOVER.md`         → toàn bộ context project, user profile, learnings session 1
   b. `ARCHITECTURE.md`     → master design (24 models, walk-forward CV, 5 tier)
   c. `MONITORING.md`       → trạng thái health hiện tại + alerts còn open
   d. `CHANGELOG.md`        → lịch sử per milestone với results
   e. `PHASE_2_PLAN.md`     → 3 option Phase 2 đã propose
   f. `CLAUDE_EXECUTION_LOG.md` → audit trail session 1 (skim quickly)

2. Verify project state vẫn lành mạnh:
   ```bash
   cd D:\WangNhat\Study\NCKH
   git status && git log --oneline -3
   pytest tests/ -q                  # phải pass 47/47
   ```

3. Báo cáo lại cho tôi (format ngắn, ≤ 200 từ):
   - Đã đọc xong files nào, file nào missing
   - Trạng thái git hiện tại (branch, last commit hash, tags)
   - Test status
   - Active alerts từ MONITORING.md
   - Đề xuất bạn nghĩ Phase 2 nên làm gì (option 1/2/3 từ PHASE_2_PLAN.md)

KHÔNG ĐƯỢC LÀM TRƯỚC KHI TÔI XÁC NHẬN:
- ❌ Không tự ý sửa code
- ❌ Không git push (đã có session 1 push đủ)
- ❌ Không install package mới
- ❌ Không xóa/move file nào
- ❌ Không tự ý chọn Phase 2 option

QUAN TRỌNG TÔI ĐÃ HỌC TỪ SESSION TRƯỚC (Claude session 1 ghi lại):
- Tôi là sinh viên TDTU 2025-2026, dự án NCKH SV
- Tôi thích autonomous execution — sau khi grant permission thì cứ chạy
- Tôi trả lời ngắn gọn ("GO", "ALL", "OK")
- Tôi thích báo cáo dạng box ASCII có metric
- Free tier ONLY — không paid API, không GPU thuê
- Push GitHub thường xuyên (sau mỗi milestone)

RÀNG BUỘC SAFETY (đừng vi phạm):
- Walk-forward CV — KHÔNG random split time series
- Scaler/outlier fit chỉ trên train fold (no leakage)
- Conventional commits với scope rõ
- Branch isolation — đừng commit vào main không có lý do
- Tag mỗi milestone trước khi push
- Nếu lỗi liên tiếp 3 lần → dừng, hỏi tôi

Bắt đầu bằng cách đọc HANDOVER.md trước. Trả lời tóm tắt khi xong.
```

## ---PROMPT END---

---

## 🔧 Variation: nếu dùng Claude Web (không có Bash tool)

Nếu bạn dùng claude.ai (không phải Claude Code CLI), thay đổi đoạn "verify project state" thành:

```
3. Vì đây là Claude Web không có terminal, tôi sẽ paste cho bạn output của:
   git status
   git log --oneline -5
   pytest tests/ -q
   khi bạn yêu cầu. Sau khi đọc HANDOVER.md hãy hỏi tôi paste output gì.
```

---

## 🔧 Variation: nếu Claude session sau là Claude Code mới với MEMORY system

Memory system tự động load `MEMORY.md` từ `~/.claude/projects/D--WangNhat-Study-NCKH/memory/`. Tôi đã save sẵn:
- `user_profile.md`
- `project_context.md`
- `feedback_workflow.md`
- `feedback_safety.md`
- `reference_repo.md`

Claude session sau sẽ **tự động** có context này — chỉ cần prompt đơn giản:

```
Tiếp tục dự án NCKH gold price. Bạn đã có memory về tôi và project.
Đọc HANDOVER.md + PHASE_2_PLAN.md rồi báo cáo trạng thái + recommend Phase 2 option.
```

---

## 📝 Quick reference cho Claude session sau

| File | Mục đích |
|---|---|
| `HANDOVER.md` | **READ FIRST** — toàn bộ context |
| `ARCHITECTURE.md` | Design decisions, model lineup |
| `MONITORING.md` | Health dashboard + alerts |
| `CHANGELOG.md` | Per-milestone results |
| `PHASE_2_PLAN.md` | Roadmap A/B/C |
| `CLAUDE_EXECUTION_LOG.md` | Audit trail (chi tiết, optional) |
| `BM02_decuong.pdf` | Đề cương gốc TDTU (proposal) |
| `reports/leaderboard/combined_v2_summary.csv` | Final benchmark numbers |
| `reports/paper/tdtu_vi/report.md` | Báo cáo VN draft |
| `reports/paper/ieee_en/main.tex` | Paper EN draft |
| `docs/DEPLOY_GUIDE.md` | Streamlit/Render deploy |

---

## ⚠️ Common pitfalls cho Claude session sau

1. **Đừng install lại packages** — đã có sẵn (statsforecast, neuralforecast, chronos, prophet, lightgbm, catboost, mlflow, shap, captum, mapie, streamlit, fastapi)
2. **Đừng refresh data** trừ khi user yêu cầu — data đã fresh tới 2026-04-25
3. **Đừng re-run benchmark từ zero** — đã có 360 records ở `reports/leaderboard/`
4. **Đừng tạo branch mới ngay** — wait user confirm Phase 2 option
5. **Đừng đụng vào `src/legacy/`** — frozen
6. **Đừng dùng Naive với shift y_observed** — phải mode-A constant
7. **yfinance**: dùng `yf.download()` + backoff, không `Ticker.history()`
8. **MLForecast**: integer index workaround vì VN holidays
9. **Chronos-Bolt 2.x**: `predict_quantiles(context, prediction_length, quantile_levels)` — positional `context`

---

🤖 *Tạo bởi Claude Opus 4.7 — handover prompt template cho session continuity.*
