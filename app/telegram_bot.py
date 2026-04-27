"""Telegram bot — gold price forecast on demand.

Commands:
  /start       — chào hello
  /predict h   — h ∈ {1, 5, 20} ngày, default 1
  /history N   — lịch sử N ngày SJC, default 30
  /leaderboard h — top 10 models @ horizon h
  /shap        — top SHAP features
  /help        — danh sách commands

Setup:
  1. Tạo bot qua @BotFather → lấy TELEGRAM_BOT_TOKEN
  2. export TELEGRAM_BOT_TOKEN="..."
  3. pip install python-telegram-bot==21.6
  4. python app/telegram_bot.py

Deploy Render free:
  - Web Service không phù hợp (bot là long-poll)
  - Background Worker: $0/month nhưng need card; OR
  - Run trên Replit free worker (always-on với Hacker plan free)
  - HOẶC cron-job.org → ping daily endpoint chạy bot batch
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# LAZY IMPORTS (cho test mà không cần telegram package)
# ============================================================

def _check_telegram_lib():
    try:
        from telegram.ext import Application
        return True
    except ImportError:
        log.error("Cần cài: pip install python-telegram-bot==21.6")
        return False


# ============================================================
# CACHED MODEL + DATA (single load on startup)
# ============================================================

_cached_features = None
_cached_models = {}


def _features():
    global _cached_features
    if _cached_features is None:
        from src.utils.io import read_parquet
        p = _PROJECT_ROOT / "data" / "processed" / "features_v2_with_sentiment.parquet"
        if not p.exists():
            return None
        _cached_features = read_parquet(p)
    return _cached_features


def _model(horizon: int = 1):
    if horizon not in _cached_models:
        from src.models.ml import ElasticNetForecaster
        df = _features()
        if df is None:
            return None
        m = ElasticNetForecaster(horizon=horizon)
        m.fit(df, target_col="SJC_ban_ra")
        _cached_models[horizon] = m
    return _cached_models[horizon]


# ============================================================
# COMMAND HANDLERS
# ============================================================

WELCOME = (
    "👋 Chào! Tôi là bot dự báo giá vàng SJC (Việt Nam).\n\n"
    "📊 Pipeline: 25 mô hình ML/DL/Foundation, walk-forward CV no-leakage.\n"
    "📄 Repo: github.com/twangnhat-05/NGHIENCUUKHOAHOC\n\n"
    "Commands:\n"
    "  /predict 1    — dự báo SJC 1 ngày tới\n"
    "  /predict 5    — 5 ngày tới\n"
    "  /predict 20   — 20 ngày tới\n"
    "  /history 30   — 30 ngày qua\n"
    "  /leaderboard 1 — top models @ h=1\n"
    "  /shap         — top features\n"
    "  /help         — help\n\n"
    "⚠️ Chỉ tham khảo. KHÔNG khuyến nghị đầu tư."
)


async def cmd_start(update, context):
    await update.message.reply_text(WELCOME)


async def cmd_help(update, context):
    await update.message.reply_text(WELCOME)


async def cmd_predict(update, context):
    args = context.args
    h = int(args[0]) if args and args[0].isdigit() else 1
    if h not in (1, 5, 20):
        await update.message.reply_text("⚠️ h phải là 1, 5 hoặc 20.")
        return
    df = _features()
    if df is None:
        await update.message.reply_text("❌ Features chưa sẵn sàng. Liên hệ admin.")
        return
    m = _model(h)
    last_row = df.iloc[[-1]]
    import pandas as pd  # local import
    pred = float(m.predict(last_row)[0])
    last_sjc = float(last_row["SJC_ban_ra"].iloc[0])
    last_date = pd.to_datetime(last_row["Date"].iloc[0]).strftime("%d/%m/%Y")
    change = pred - last_sjc
    pct = change / last_sjc * 100
    arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
    msg = (
        f"🏅 *Dự báo SJC ban ra (h={h} ngày)*\n\n"
        f"📅 Dữ liệu đến: {last_date}\n"
        f"💰 Giá hiện tại: *{last_sjc:.2f}* triệu VND/lượng\n"
        f"🔮 Dự báo: *{pred:.2f}* triệu VND/lượng\n"
        f"{arrow} Thay đổi: *{change:+.2f}* ({pct:+.2f}%)\n\n"
        f"Mô hình: ElasticNet (108 features, MAPE ~0.67% h=1 trên CV)\n"
        f"⚠️ Chỉ tham khảo."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_history(update, context):
    args = context.args
    days = int(args[0]) if args and args[0].isdigit() else 30
    days = min(max(days, 1), 100)
    df = _features()
    if df is None:
        await update.message.reply_text("❌ Data not ready.")
        return
    sub = df.tail(days)[["Date", "SJC_ban_ra"]].copy()
    import pandas as pd
    sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%d/%m")
    lines = [f"📊 *SJC {days} ngày gần nhất*\n"]
    lines += [f"{r['Date']}  →  {r['SJC_ban_ra']:.2f}" for _, r in sub.iterrows()]
    txt = "\n".join(lines[:31])  # Telegram 4096 char limit
    await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_leaderboard(update, context):
    args = context.args
    h = int(args[0]) if args and args[0].isdigit() else 1
    if h not in (1, 5, 20):
        await update.message.reply_text("⚠️ h phải là 1, 5 hoặc 20.")
        return
    p = _PROJECT_ROOT / "reports" / "leaderboard" / "combined_v2_summary.csv"
    if not p.exists():
        await update.message.reply_text("❌ Leaderboard not ready.")
        return
    import pandas as pd
    df = pd.read_csv(p)
    sub = df[(df["horizon"] == h) & (df["metric"] == "MAPE")].sort_values("mean").head(10)
    lines = [f"🏆 *Top 10 models @ h={h} ngày (MAPE)*\n"]
    for i, r in enumerate(sub.itertuples(), 1):
        emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
        lines.append(f"{emoji} `{r.model}` — {r.mean:.3f}% (±{r.std:.2f})")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_shap(update, context):
    p = _PROJECT_ROOT / "reports" / "figures" / "shap_lightgbm_h1_top20.csv"
    if not p.exists():
        await update.message.reply_text("❌ SHAP not ready.")
        return
    import pandas as pd
    df = pd.read_csv(p).head(10)
    lines = ["🔍 *Top 10 features (LightGBM h=1, mean |SHAP|)*\n"]
    for r in df.itertuples():
        lines.append(f"`{r.feature}` — {r.mean_abs_shap:.3f}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    if not _check_telegram_lib():
        return 1
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("Missing env var TELEGRAM_BOT_TOKEN. Get from @BotFather.")
        return 1

    from telegram.ext import Application, CommandHandler
    log.info("Starting Telegram bot...")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
    app.add_handler(CommandHandler("shap", cmd_shap))
    # Pre-load features + ElasticNet h=1
    _features()
    _model(1)
    log.info("Bot ready. Polling for messages...")
    app.run_polling()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
