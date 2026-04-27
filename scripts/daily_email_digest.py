"""Daily email digest — gửi forecast SJC mỗi sáng cho subscribers.

Sử dụng SMTP free (Gmail App Password recommended).

Setup:
    1. Tạo Gmail App Password (Google Account → Security → 2-Step → App Passwords)
    2. export SMTP_USER="your@gmail.com"
       export SMTP_PASS="<16-char app password>"
       export SMTP_TO="recipient1@email.com,recipient2@email.com"
    3. python scripts/daily_email_digest.py

Cron daily (Linux):
    0 7 * * * cd /path/to/NGHIENCUUKHOAHOC && bash scripts/daily_email_digest.sh
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
import warnings
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def build_digest() -> str:
    """Build HTML email body với latest forecast 3 horizons."""
    from src.models.ml import ElasticNetForecaster
    from src.utils.io import read_parquet
    import pandas as pd

    p = _PROJECT_ROOT / "data" / "processed" / "features_v2_with_sentiment.parquet"
    if not p.exists():
        return "<p>❌ Features not ready</p>"
    df = read_parquet(p)
    last_row = df.iloc[[-1]]
    last_sjc = float(last_row["SJC_ban_ra"].iloc[0])
    last_date = pd.to_datetime(last_row["Date"].iloc[0]).strftime("%d/%m/%Y")

    forecasts = {}
    for h in [1, 5, 20]:
        m = ElasticNetForecaster(horizon=h)
        m.fit(df, target_col="SJC_ban_ra")
        pred = float(m.predict(last_row)[0])
        change_pct = (pred - last_sjc) / last_sjc * 100
        forecasts[h] = (pred, change_pct)

    today_str = datetime.now().strftime("%d/%m/%Y")
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px;">
        <h2 style="color: #FFB800;">🏅 Gold SJC Daily Forecast — {today_str}</h2>
        <p><strong>Dữ liệu đến ngày:</strong> {last_date}</p>
        <p><strong>Giá hiện tại:</strong> {last_sjc:.2f} triệu VND/lượng</p>

        <h3>📊 Dự báo (ElasticNet, MAPE ~0.67% h=1 trên CV)</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background: #f0f0f0;">
                <th style="padding: 8px; border: 1px solid #ccc;">Horizon</th>
                <th style="padding: 8px; border: 1px solid #ccc;">Predicted SJC</th>
                <th style="padding: 8px; border: 1px solid #ccc;">Change</th>
            </tr>
    """
    for h, (pred, ch) in forecasts.items():
        arrow = "📈" if ch > 0 else "📉" if ch < 0 else "➡️"
        color = "green" if ch > 0 else "red" if ch < 0 else "gray"
        html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc;">{h} ngày</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{pred:.2f}</td>
                <td style="padding: 8px; border: 1px solid #ccc; color: {color};">
                    {arrow} {ch:+.2f}%
                </td>
            </tr>
        """
    html += """
        </table>
        <p style="margin-top: 20px; color: #888; font-size: 12px;">
            ⚠️ Chỉ tham khảo. KHÔNG khuyến nghị đầu tư.<br>
            Repo: <a href="https://github.com/twangnhat-05/NGHIENCUUKHOAHOC">github</a>
        </p>
    </body>
    </html>
    """
    return html


def send_email(to_addrs: list[str], subject: str, html_body: str,
               smtp_host: str = "smtp.gmail.com", smtp_port: int = 587) -> bool:
    """Gửi email HTML qua SMTP."""
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    if not user or not pwd:
        print("❌ Missing SMTP_USER or SMTP_PASS env vars", file=sys.stderr)
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
        print(f"✅ Email sent to {len(to_addrs)} recipients")
        return True
    except Exception as e:
        print(f"❌ Send failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", help="Comma-separated emails (default: $SMTP_TO env)")
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--dry-run", action="store_true", help="Print HTML, don't send")
    args = parser.parse_args()

    to_str = args.to or os.environ.get("SMTP_TO", "")
    to_addrs = [t.strip() for t in to_str.split(",") if t.strip()]

    html = build_digest()
    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"🏅 Gold SJC Daily Forecast — {today}"

    if args.dry_run:
        print(f"=== DRY RUN ===\nSubject: {subject}\n")
        print(html)
        return 0

    if not to_addrs:
        print("❌ No recipients. Pass --to or set SMTP_TO env.", file=sys.stderr)
        return 1

    ok = send_email(to_addrs, subject, html, args.smtp_host, args.smtp_port)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
