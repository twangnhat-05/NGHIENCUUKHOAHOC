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
    default=available_models[:8] if len(available_models) >= 8 else available_models,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Repo**: [twangnhat-05/NGHIENCUUKHOAHOC](https://github.com/twangnhat-05/NGHIENCUUKHOAHOC)\n\n"
    "**Architect**: Claude Opus 4.7 + WangNhat (TDTU)"
)


# ============================================================
# MAIN: 4 TABS
# ============================================================
tab_overview, tab_leaderboard, tab_predict, tab_xai = st.tabs(
    ["📊 Overview", "🏆 Leaderboard", "📈 Predictions", "🔍 XAI / SHAP"]
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

    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Training ElasticNet và generate forecast..."):
            try:
                from src.models.ml import ElasticNetForecaster
                model = ElasticNetForecaster(horizon=horizon)
                # Use last 200 rows as "test", rest as train
                n = len(features_df)
                cut = n - 100
                train = features_df.iloc[:cut].copy()
                test = features_df.iloc[cut:].copy()
                model.fit(train, target_col="SJC_ban_ra")
                preds = model.predict(test)
                test_dates = pd.to_datetime(test["Date"]).reset_index(drop=True)
                actual = test["SJC_ban_ra"].reset_index(drop=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=actual, name="Actual",
                                          line=dict(color="black", width=2)))
                fig.add_trace(go.Scatter(x=test_dates, y=preds, name="ElasticNet pred",
                                          line=dict(color="#2ca02c", width=2)))
                fig.update_layout(
                    title=f"ElasticNet Forecast h={horizon} (last 100 days)",
                    xaxis_title="Date", yaxis_title="SJC bán ra (triệu VND)",
                    height=450, hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Compute MAPE
                mape = np.mean(np.abs((actual.values - preds) / actual.values)) * 100
                st.success(f"✅ MAPE trên test = **{mape:.3f}%**")
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


st.markdown("---")
st.caption(
    "🤖 Co-architected with Claude Opus 4.7 (Anthropic) | "
    "📄 [GitHub](https://github.com/twangnhat-05/NGHIENCUUKHOAHOC) | "
    "🎓 TDTU NCKH SV 2025-2026"
)
