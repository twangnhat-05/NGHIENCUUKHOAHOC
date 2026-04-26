#!/bin/bash
# Reproduce toàn bộ pipeline cho NCKH gold price project.
# Usage: bash scripts/reproduce_all.sh
# Estimated: ~70 phút trên CPU; <30 phút nếu skip DL benchmark.

set -e
cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════════"
echo " GOLD PRICE FORECASTING — REPRODUCE FULL PIPELINE (TDTU NCKH)"
echo "════════════════════════════════════════════════════════════════"

export PYTHONPATH=.
export PYTHONIOENCODING=utf-8
export TF_CPP_MIN_LOG_LEVEL=3

echo ""
echo "[1/9] Refresh data (delta) — ~2 phút"
python -m src.data.refresh

echo ""
echo "[2/9] Schema validation"
python -m src.data.schema

echo ""
echo "[3/9] Merge raw → interim/merged.parquet"
python -m src.data.merge

echo ""
echo "[4/9] Build features V2 → processed/features_v2.parquet"
python -m src.features.build

echo ""
echo "[5/9] Add sentiment STUB → features_v2_with_sentiment.parquet"
python -m src.features.sentiment stub

echo ""
echo "[6/9] Run classical baselines (~5 phút)"
python scripts/run_classical_baselines.py --horizons 1 5 20 --name classical_full

echo ""
echo "[7/9] Run ML baselines (~3 phút)"
python scripts/run_ml_baselines.py --horizons 1 5 20 --name ml

echo ""
echo "[8/9] Run DL baselines (~50-60 phút) — UNCOMMENT để chạy"
echo "  python scripts/run_dl_baselines.py --horizons 1 5 20 --fast --include-simple --name dl"

echo ""
echo "[9/9] Run foundation models (~5 phút)"
python scripts/run_foundation_baselines.py --horizons 1 5 20 --name foundation

echo ""
echo "[bonus] Combined leaderboard + Friedman test"
python scripts/combine_leaderboards.py \
    --inputs classical_full_long.csv ml_long.csv foundation_long.csv \
    --output-name combined_v2

echo ""
echo "[bonus] XAI + Conformal demo"
python scripts/run_xai_conformal_demo.py --horizon 1

echo ""
echo "[bonus] Run unit tests"
python -m pytest tests/ -v

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " DONE. Kết quả tại reports/leaderboard/ và reports/figures/"
echo " Dashboard: streamlit run app/streamlit_app.py"
echo "════════════════════════════════════════════════════════════════"
