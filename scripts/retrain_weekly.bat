@echo off
REM Weekly retrain — Windows Task Scheduler entry
REM Setup:
REM   schtasks /create /tn "GoldRetrain" /tr "%CD%\scripts\retrain_weekly.bat" /sc weekly /d MON /st 06:00

cd /d "%~dp0\.."
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
set TF_CPP_MIN_LOG_LEVEL=3

echo [%date% %time%] Starting weekly retrain...
python scripts\retrain_weekly.py --horizon 1 --alert-threshold-pct 20
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] Done with exit code %EXIT_CODE%
exit /b %EXIT_CODE%
