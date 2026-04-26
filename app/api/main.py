"""FastAPI server cho gold price predictions.

Usage:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              — health
    GET  /predict?h=1   — latest forecast (h ∈ {1, 5, 20})
    GET  /history?days=30 — gần đây N ngày SJC
    GET  /leaderboard?h=1 — top models theo MAPE
    GET  /shap          — top SHAP features
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.ml import ElasticNetForecaster, RidgeForecaster
from src.utils.io import read_parquet


app = FastAPI(
    title="Gold Price Forecasting API (SJC)",
    description="TDTU NCKH 2025-2026 — multi-horizon SJC gold forecasting",
    version="0.5.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ============================================================
# CACHE: load data once on startup
# ============================================================
_cached_features: pd.DataFrame | None = None
_cached_models: dict[int, ElasticNetForecaster] = {}


def _features() -> pd.DataFrame:
    global _cached_features
    if _cached_features is None:
        p = _PROJECT_ROOT / "data" / "processed" / "features_v2_with_sentiment.parquet"
        if not p.exists():
            raise HTTPException(503, "Features chưa build. Chạy `python -m src.features.build`.")
        _cached_features = read_parquet(p)
    return _cached_features


def _model(horizon: int) -> ElasticNetForecaster:
    if horizon not in _cached_models:
        df = _features()
        m = ElasticNetForecaster(horizon=horizon)
        m.fit(df, target_col="SJC_ban_ra")
        _cached_models[horizon] = m
    return _cached_models[horizon]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Gold Price Forecasting API",
        "version": "0.5.0",
        "endpoints": ["/predict", "/history", "/leaderboard", "/shap", "/docs"],
    }


@app.get("/predict")
def predict(h: int = Query(1, description="Horizon: 1, 5, hoặc 20 ngày")):
    """Predict SJC bán ra `h` ngày tới dựa trên latest features."""
    if h not in (1, 5, 20):
        raise HTTPException(400, "h phải là 1, 5, hoặc 20")
    df = _features()
    model = _model(h)
    last_row = df.iloc[[-1]].copy()
    pred = float(model.predict(last_row)[0])
    last_date = pd.to_datetime(last_row["Date"].iloc[0])
    last_sjc = float(last_row["SJC_ban_ra"].iloc[0])
    return {
        "horizon_days": h,
        "as_of_date": last_date.strftime("%Y-%m-%d"),
        "current_sjc_ban_ra": round(last_sjc, 4),
        "predicted_sjc_ban_ra": round(pred, 4),
        "predicted_change_pct": round((pred - last_sjc) / last_sjc * 100, 4),
        "model": "ElasticNet (108 engineered features, no Optuna tune)",
        "note": "Chỉ tham khảo. KHÔNG khuyến nghị đầu tư.",
    }


@app.get("/history")
def history(days: int = Query(30, ge=1, le=2000)):
    df = _features()
    sub = df.tail(days)[["Date", "SJC_ban_ra", "SJC_mua_vao"]].copy()
    sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")
    return {"days": len(sub), "data": sub.to_dict(orient="records")}


@app.get("/leaderboard")
def leaderboard(h: int = Query(1), top: int = Query(10, ge=1, le=30)):
    p = _PROJECT_ROOT / "reports" / "leaderboard" / "combined_v2_summary.csv"
    if not p.exists():
        raise HTTPException(503, "Leaderboard chưa generated. Chạy combine_leaderboards.py.")
    df = pd.read_csv(p)
    sub = df[(df["horizon"] == h) & (df["metric"] == "MAPE")].sort_values("mean").head(top)
    return {
        "horizon": h, "metric": "MAPE",
        "leaderboard": sub[["model", "mean", "std", "count"]]
            .rename(columns={"mean": "mean_mape_pct", "count": "n_folds"})
            .to_dict(orient="records"),
    }


@app.get("/shap")
def shap_top():
    p = _PROJECT_ROOT / "reports" / "figures" / "shap_lightgbm_h1_top20.csv"
    if not p.exists():
        raise HTTPException(503, "SHAP chưa generated. Chạy `scripts/run_xai_conformal_demo.py`.")
    df = pd.read_csv(p)
    return {"model": "LightGBM h=1", "top_features": df.to_dict(orient="records")}
