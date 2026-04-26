@echo off
REM Reproduce toàn bộ pipeline (Windows version)
REM Usage: scripts\reproduce_all.bat

cd /d "%~dp0\.."

set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
set TF_CPP_MIN_LOG_LEVEL=3

echo ================================================================
echo  GOLD PRICE FORECASTING - REPRODUCE FULL PIPELINE (TDTU NCKH)
echo ================================================================

echo.
echo [1/9] Refresh data
python -m src.data.refresh
if errorlevel 1 goto :fail

echo.
echo [2/9] Schema validation
python -m src.data.schema
if errorlevel 1 goto :fail

echo.
echo [3/9] Merge raw
python -m src.data.merge
if errorlevel 1 goto :fail

echo.
echo [4/9] Build features V2
python -m src.features.build
if errorlevel 1 goto :fail

echo.
echo [5/9] Sentiment STUB
python -m src.features.sentiment stub
if errorlevel 1 goto :fail

echo.
echo [6/9] Classical baselines
python scripts\run_classical_baselines.py --horizons 1 5 20 --name classical_full
if errorlevel 1 goto :fail

echo.
echo [7/9] ML baselines
python scripts\run_ml_baselines.py --horizons 1 5 20 --name ml
if errorlevel 1 goto :fail

echo.
echo [8/9] DL baselines (50-60 phut) - UNCOMMENT khi chay
REM python scripts\run_dl_baselines.py --horizons 1 5 20 --fast --include-simple --name dl

echo.
echo [9/9] Foundation models
python scripts\run_foundation_baselines.py --horizons 1 5 20 --name foundation
if errorlevel 1 goto :fail

echo.
echo [bonus] Combined leaderboard
python scripts\combine_leaderboards.py --inputs classical_full_long.csv ml_long.csv foundation_long.csv --output-name combined_v2
if errorlevel 1 goto :fail

echo.
echo [bonus] XAI + Conformal demo
python scripts\run_xai_conformal_demo.py --horizon 1
if errorlevel 1 goto :fail

echo.
echo [bonus] Unit tests
python -m pytest tests\ -v
if errorlevel 1 goto :fail

echo.
echo ================================================================
echo  DONE. Ket qua tai reports\leaderboard va reports\figures
echo  Dashboard: streamlit run app\streamlit_app.py
echo ================================================================
exit /b 0

:fail
echo.
echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
echo  PIPELINE FAILED - kiem tra log o tren
echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
exit /b 1
