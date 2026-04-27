#!/bin/bash
# Weekly retrain — Linux/Mac/WSL cron entry
# Setup:
#   crontab -e
#   0 6 * * 1 cd /path/to/NGHIENCUUKHOAHOC && bash scripts/retrain_weekly.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTHONIOENCODING=utf-8
export TF_CPP_MIN_LOG_LEVEL=3

echo "[$(date)] Starting weekly retrain..."
python scripts/retrain_weekly.py --horizon 1 --alert-threshold-pct 20
EXIT_CODE=$?

echo "[$(date)] Done with exit code $EXIT_CODE"
exit $EXIT_CODE
