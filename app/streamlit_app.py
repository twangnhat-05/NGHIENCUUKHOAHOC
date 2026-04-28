"""Streamlit dashboard — Vietnamese gold price (SJC) forecasting demo.

Usage (local):
    streamlit run app/streamlit_app.py

Deploy free:
    Push to GitHub → Streamlit Community Cloud (https://share.streamlit.io)
    requirements: cài từ requirements.txt
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Ensure src/ on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Gold Price Forecasting (SJC) — TDTU NCKH 2025-2026",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏅 Vietnamese Gold Price Forecasting (SJC)")
st.caption(
    "TDTU NCKH 2025-2026 — Multi-horizon forecasting với 24 models "
    "(classical + ML + DL + foundation) trên walk-forward CV."
)


# ============================================================
# DATA LOADERS (cached)
# ============================================================
@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame:
    p = _PROJECT_ROOT / "data" / "processed" / "features_v2_with_sentiment.parquet"
    if not p.exists():
        st.error(f"Không tìm thấy features: {p}. Chạy `python -m src.features.build` trước.")
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(ttl=3600)
def load_combined_summary() -> pd.DataFrame:
    p = _PROJECT_ROOT / "reports" / "leaderboard" / "combined_v2_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=3600)
def load_friedman() -> pd.DataFrame:
    p = _PROJECT_ROOT / "reports" / "leaderboard" / "friedman_test.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=3600)
def load_shap_top() -> pd.DataFrame:
    p = _PROJECT_ROOT / "reports" / "figures" / "shap_lightgbm_h1_top20.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Settings")
features_df = load_features()
summary_df = load_combined_summary()

if features_df.empty:
    st.warning("Chưa có data. Chạy pipeline trước theo README.")
    st.stop()

date_min = pd.to_datetime(features_df["Date"]).min().date()
date_max = pd.to_datetime(features_df["Date"]).max().date()

date_range = st.sidebar.date_input(
    "📅 Hiển thị từ ngày",
    value=(max(date_min, date_max - pd.Timedelta(days=365 * 2)), date_max),
    min_value=date_min,
    max_value=date_max,
)

horizon = st.sidebar.selectbox(
    "🎯 Horizon dự báo",
    options=[1, 5, 20],
    format_func=lambda h: f"h={h} ngày",
)

available_models = sorted(summary_df["model"].unique()) if not summary_df.empty else []
selected_models = st.sidebar.multiselect(
    "🤖 Models để hiển thị (leaderboard)",
    options=available_models,
    default=available_models,   # default = ALL 24 (was top-8 alphabetically)
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Repo**: [twangnhat-05/NGHIENCUUKHOAHOC](https://github.com/twangnhat-05/NGHIENCUUKHOAHOC)\n\n"
    "**Architect**: WangNhat (TDTU)"
)


# ============================================================
# MAIN: 5 TABS (added 🔴 Live News in Phase 9)
# ============================================================
tab_overview, tab_leaderboard, tab_predict, tab_xai, tab_news = st.tabs(
    ["📊 Overview", "🏆 Leaderboard", "📈 Predictions", "🔍 XAI / SHAP", "🔴 Live News"]
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab_overview:
    st.subheader("Lịch sử giá vàng SJC (mua/bán)")

    df_filt = features_df.copy()
    df_filt["Date"] = pd.to_datetime(df_filt["Date"])
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df_filt = df_filt[(df_filt["Date"] >= d0) & (df_filt["Date"] <= d1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filt["Date"], y=df_filt["SJC_ban_ra"],
        mode="lines", name="SJC bán ra", line=dict(color="#1f77b4", width=2),
    ))
    if "SJC_mua_vao" in df_filt.columns:
        fig.add_trace(go.Scatter(
            x=df_filt["Date"], y=df_filt["SJC_mua_vao"],
            mode="lines", name="SJC mua vào", line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
    fig.update_layout(
        height=420,
        xaxis_title="Ngày", yaxis_title="Giá (triệu VND/lượng)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    last = df_filt.iloc[-1] if len(df_filt) else None
    if last is not None:
        col1.metric("Giá SJC cuối kỳ", f"{last['SJC_ban_ra']:.2f} triệu/lượng")
        prev_30 = df_filt.iloc[-30:]
        if len(prev_30) > 1:
            change = (prev_30["SJC_ban_ra"].iloc[-1] - prev_30["SJC_ban_ra"].iloc[0])
            col2.metric("Thay đổi 30 ngày", f"{change:+.2f}",
                        delta=f"{change / prev_30['SJC_ban_ra'].iloc[0] * 100:.2f}%")
        col3.metric("Số dòng dữ liệu", f"{len(df_filt):,}")
        col4.metric("Số features", f"{features_df.shape[1] - 4}")

    st.markdown("---")
    st.markdown(
        "### Kiến trúc dự án\n"
        "- **Data**: 11 nguồn (yfinance + FRED + vnstock + webgia.com SJC scraper)\n"
        "- **Features V2**: 108 — lags + technical + macro + calendar (VN holidays + Tết)\n"
        "- **Models**: 24 — Naive/SeasonalNaive/AutoARIMA/Prophet/Ridge/XGBoost/LSTM/PatchTST/TSMixer/Chronos-Bolt/...\n"
        "- **Evaluation**: Walk-forward CV 5 folds (no leakage), Friedman test p < 0.001\n"
        "- **PI**: Adaptive Conformal Inference (Gibbs & Candès 2021)\n"
        "- **XAI**: SHAP TreeExplainer + Captum Integrated Gradients"
    )


# ============================================================
# TAB 2: LEADERBOARD
# ============================================================
with tab_leaderboard:
    st.subheader(f"🏆 Leaderboard — Horizon h={horizon}")

    if summary_df.empty:
        st.warning("Chưa có leaderboard. Chạy `scripts/run_*_baselines.py` + `combine_leaderboards.py`.")
    else:
        sub = summary_df[
            (summary_df["horizon"] == horizon)
            & (summary_df["metric"] == "MAPE")
            & (summary_df["model"].isin(selected_models if selected_models else summary_df["model"]))
        ].copy()
        sub = sub.sort_values("mean").reset_index(drop=True)
        sub["rank"] = sub.index + 1
        sub = sub[["rank", "model", "mean", "std", "count"]]
        sub.columns = ["#", "Model", "Mean MAPE (%)", "Std", "Folds"]

        # Highlight top 3
        def highlight_top(row):
            if row["#"] == 1:
                return ["background-color: #d4edda"] * len(row)
            elif row["#"] in (2, 3):
                return ["background-color: #fff3cd"] * len(row)
            else:
                return [""] * len(row)

        st.dataframe(
            sub.style.apply(highlight_top, axis=1).format({"Mean MAPE (%)": "{:.3f}", "Std": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

        # Bar plot
        fig = px.bar(
            sub.head(15), x="Mean MAPE (%)", y="Model",
            orientation="h", error_x="Std",
            title=f"Top 15 Models @ h={horizon} (lower = better)",
            color="Mean MAPE (%)", color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(yaxis={"categoryorder": "total descending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Friedman test
        st.markdown("---")
        st.subheader("📊 Friedman Statistical Test")
        fr = load_friedman()
        if not fr.empty:
            st.dataframe(fr, hide_index=True, use_container_width=True)
            st.caption(
                "**Friedman test**: kiểm định có sự khác biệt thống kê giữa các models không. "
                "p < 0.05 → models khác nhau có ý nghĩa thống kê. "
                "Ta thấy p << 0.001 cho cả 3 horizons → mô hình thực sự khác biệt."
            )
        else:
            st.info("Chưa có Friedman test. Chạy `combine_leaderboards.py`.")


# ============================================================
# TAB 3: PREDICTIONS
# ============================================================
with tab_predict:
    st.subheader(f"📈 Predictions — Horizon h={horizon}")
    st.caption(
        "Demo realtime prediction sử dụng features đã engineering. "
        "Click \"Generate Forecast\" để retrain ElasticNet trên full data và dự báo."
    )

    apply_tone = st.toggle(
        "📰 Apply live news tone adjustment",
        value=False,
        help="Shift the base forecast by a small calibrated amount based on "
             "the average GDELT tone of gold-relevant news in the last 24h.",
    )

    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Training ElasticNet và generate forecast..."):
            try:
                from src.models.ml import ElasticNetForecaster
                from src.forecast.live_tone import (
                    compute_live_tone, apply_tone_adjustment,
                )
                model = ElasticNetForecaster(horizon=horizon)
                n = len(features_df)
                cut = n - 100
                train = features_df.iloc[:cut].copy()
                test = features_df.iloc[cut:].copy()
                model.fit(train, target_col="SJC_ban_ra")
                base_preds = model.predict(test)
                test_dates = pd.to_datetime(test["Date"]).reset_index(drop=True)
                actual = test["SJC_ban_ra"].reset_index(drop=True)

                # Live-tone adjustment (only the LATEST point — context for
                # today's forecast, not retroactive recalibration).
                snap = compute_live_tone()
                adj_preds = base_preds.copy()
                if apply_tone and snap.tone_mean is not None:
                    adj_last, _delta = apply_tone_adjustment(
                        base_preds[-1], snap.tone_mean,
                    )
                    adj_preds[-1] = adj_last

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=actual, name="Actual",
                                          line=dict(color="black", width=2)))
                fig.add_trace(go.Scatter(x=test_dates, y=base_preds,
                                          name="ElasticNet base",
                                          line=dict(color="#2ca02c", width=2)))
                if apply_tone and snap.tone_mean is not None:
                    fig.add_trace(go.Scatter(
                        x=test_dates.tail(1), y=[adj_preds[-1]],
                        mode="markers", marker=dict(size=12, color="#1565C0",
                                                    symbol="star"),
                        name=f"News-adjusted (tone={snap.tone_mean:+.1f})",
                    ))
                fig.update_layout(
                    title=f"ElasticNet Forecast h={horizon} (last 100 days)",
                    xaxis_title="Date", yaxis_title="SJC bán ra (triệu VND)",
                    height=450, hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # MAPE on the test slice (computed on the BASE forecast — the
                # adjustment only touches the very last point).
                mape = np.mean(np.abs((actual.values - base_preds) / actual.values)) * 100
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("MAPE (test)", f"{mape:.3f}%")
                col_b.metric("Base forecast (latest)",
                             f"{base_preds[-1]:.2f}M VND")
                if apply_tone and snap.tone_mean is not None:
                    delta = (adj_preds[-1] - base_preds[-1]) / base_preds[-1] * 100
                    col_c.metric(
                        "News-adjusted",
                        f"{adj_preds[-1]:.2f}M VND",
                        f"{delta:+.3f}%",
                    )
                else:
                    col_c.metric("News-adjusted", "—",
                                 help="Toggle the option above to enable")

                # Tone explanation panel
                with st.expander("ℹ️ Live tone snapshot"):
                    if snap.tone_mean is None:
                        st.warning(f"No usable tone data: {snap.fallback_reason}")
                    else:
                        st.write(
                            f"- **Mean tone (last {snap.window_hours}h):** "
                            f"`{snap.tone_mean:+.2f}` (-100 strong negative → +100 strong positive)"
                        )
                        st.write(f"- **Articles aggregated:** {snap.n_articles}")
                        if snap.age_minutes is not None:
                            st.write(f"- **Latest article:** {snap.age_minutes:.0f} min ago")
                        st.caption(
                            "Calibration prior α = 0.001 per tone-point — small by design "
                            "because the live-news archive does not overlap the historical "
                            "CV window (see paper §4.5). Delta is informational, not a "
                            "validated MAPE improvement."
                        )
            except Exception as e:
                st.error(f"Forecast failed: {e}")


# ============================================================
# TAB 4: XAI
# ============================================================
with tab_xai:
    st.subheader("🔍 XAI — SHAP Feature Importance (LightGBM h=1)")

    shap_df = load_shap_top()
    if not shap_df.empty:
        fig = px.bar(
            shap_df, x="mean_abs_shap", y="feature",
            orientation="h",
            title="Top 20 Features (mean |SHAP|)",
            color="mean_abs_shap", color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chưa có SHAP. Chạy `scripts/run_xai_conformal_demo.py`.")

    # ACI conformal plot
    st.markdown("---")
    st.subheader("📐 Adaptive Conformal Prediction Intervals (ACI)")
    aci_path = _PROJECT_ROOT / "reports" / "figures" / "aci_conformal_elasticnet_h1.png"
    if aci_path.exists():
        st.image(str(aci_path), caption="ACI 90% PI cho ElasticNet h=1 (last fold = 2024 gold rally)",
                 use_container_width=True)
        st.caption(
            "ACI (Gibbs & Candès 2021) tự động adjust alpha online dựa trên coverage thực tế. "
            "Trong volatile period (2024 rally), split conformal under-cover (~83%); ACI vẫn đạt ~83% với target alpha=0.10."
        )
    else:
        st.info("Chưa có ACI plot. Chạy `scripts/run_xai_conformal_demo.py`.")


# ============================================================
# TAB 5: 🔴 LIVE NEWS (Phase 9 — real-time gold-relevant news)
# ============================================================
with tab_news:
    st.subheader("🔴 Real-time gold-relevant news")
    st.caption(
        "Aggregated từ GDELT 2.0 + Reddit + RSS feeds (free tier, không API key). "
        "GDELT cung cấp tone -100..+100; refresh cron 15 phút qua GitHub Actions."
    )

    @st.cache_data(ttl=300)  # 5-min cache
    def _load_news_realtime():
        p = _PROJECT_ROOT / "data" / "external" / "news_realtime.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        return df.dropna(subset=["ts"]).sort_values("ts", ascending=False)

    news_df = _load_news_realtime()

    if news_df.empty:
        st.warning(
            "Chưa có news data. Chạy `python -m scripts.refresh_news_realtime` "
            "để fetch lần đầu, hoặc đợi GitHub Actions cron."
        )
    else:
        # ── headline metrics ─────────────────────────────────
        last_ts = news_df["ts"].max()
        age_min = (pd.Timestamp.utcnow() - last_ts).total_seconds() / 60
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng tin (90 ngày)", f"{len(news_df):,}")
        col2.metric("Nguồn", news_df["source"].nunique())
        col3.metric("Tin mới nhất", f"{age_min:.0f} phút trước")
        gd = news_df[news_df["source"] == "gdelt"].copy()
        gd["tone"] = pd.to_numeric(gd["tone"], errors="coerce")
        if not gd.empty:
            today_tone = gd[gd["date"] == gd["date"].max()]["tone"].mean()
            col4.metric("GDELT tone hôm nay", f"{today_tone:+.2f}",
                        help="-100 (rất tiêu cực) → +100 (rất tích cực)")

        # ── sentiment trend chart ────────────────────────────
        if not gd.empty:
            daily = (
                gd.groupby("date")
                .agg(tone=("tone", "mean"), n=("tone", "count"))
                .reset_index()
                .tail(30)
            )
            chart = (
                alt.Chart(daily)
                .mark_line(point=True, color="#1565C0")
                .encode(
                    x=alt.X("date:T", title="Ngày"),
                    y=alt.Y("tone:Q", title="GDELT tone trung bình",
                            scale=alt.Scale(zero=False)),
                    tooltip=["date", "tone", "n"],
                )
                .properties(height=220)
            )
            zero_rule = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(strokeDash=[4, 4], color="#9E9E9E")
                .encode(y="y:Q")
            )
            st.altair_chart(chart + zero_rule, use_container_width=True)

        # ── source breakdown ─────────────────────────────────
        st.markdown("**Phân bố theo nguồn (90 ngày):**")
        src_counts = news_df["source"].value_counts().reset_index()
        src_counts.columns = ["source", "n"]
        st.dataframe(src_counts, use_container_width=True, hide_index=True)

        # ── latest headlines ─────────────────────────────────
        st.markdown("**📰 Tin mới nhất (top 30):**")
        recent = news_df.head(30).copy()
        recent["khi nào"] = (
            (pd.Timestamp.utcnow() - recent["ts"]).dt.total_seconds() / 60
        ).round().astype(int).astype(str) + " phút"
        recent["tone"] = pd.to_numeric(recent["tone"], errors="coerce").round(1)
        display = recent[["khi nào", "source", "title", "tone", "url"]].rename(
            columns={"source": "nguồn", "title": "tiêu đề", "url": "link"}
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "link": st.column_config.LinkColumn("link", display_text="↗"),
            },
        )

        st.caption(
            "ℹ️ Để score sentiment cho Reddit/RSS (mDeBERTa zero-shot), chạy: "
            "`python -m scripts.refresh_news_realtime --score-recent 50` "
            "(mất ~30s/50 tin trên CPU)."
        )


st.markdown("---")
st.caption(
    "📄 [GitHub](https://github.com/twangnhat-05/NGHIENCUUKHOAHOC) | "
    "🎓 TDTU NCKH SV 2025-2026"
)
