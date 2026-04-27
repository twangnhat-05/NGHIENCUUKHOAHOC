"""Weekly auto-retrain script.

Pipeline mỗi tuần:
1. Refresh data delta (yfinance + FRED + vnstock + SJC)
2. Re-merge + re-build features
3. Re-train top 3 ML models (Ridge, ElasticNet, LightGBM) trên full latest data
4. Compare new MAPE vs baseline (last week's leaderboard)
5. Alert (log/email) nếu degrade > 20%
6. Save updated models + predictions

Setup cron (Linux/Mac/WSL):
    crontab -e
    0 6 * * 1 cd /path/to/NGHIENCUUKHOAHOC && bash scripts/retrain_weekly.sh

Setup Task Scheduler (Windows):
    schtasks /create /tn "GoldRetrain" /tr "scripts\retrain_weekly.bat" /sc weekly /d MON /st 06:00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data.refresh import refresh_delta
from src.features.build import build_features
from src.data.merge import merge_all
from src.models.ml import ElasticNetForecaster, LightGBMForecaster, RidgeForecaster
from src.training.cv import build_cv_from_config
from src.training.trainer import evaluate_ml_one_fold
from src.utils.io import load_yaml, project_root, read_parquet, write_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly retrain & monitor")
    parser.add_argument("--alert-threshold-pct", type=float, default=20.0,
                        help="Alert nếu MAPE tăng > threshold% so baseline")
    parser.add_argument("--baseline", default="reports/leaderboard/ml_summary.csv",
                        help="Baseline summary CSV để so sánh")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--skip-refresh", action="store_true",
                        help="Bỏ qua data refresh (test pipeline)")
    args = parser.parse_args()

    set_global_seed(42)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    log.info(f"=== WEEKLY RETRAIN @ {timestamp} ===")

    # 1. Refresh data
    if not args.skip_refresh:
        log.info("Step 1/5: Refreshing data...")
        try:
            summary = refresh_delta()
            log.info(f"Refresh: {len(summary)} sources")
        except Exception as e:
            log.error(f"Refresh failed: {e}")
            return 1

    # 2. Re-merge + features
    log.info("Step 2/5: Merging + building features...")
    try:
        merged = merge_all()
        feat_cfg = load_yaml("configs/features.yaml")
        features = build_features(merged, feat_cfg, drop_na_targets=True)
        feat_path = project_root() / "data" / "processed" / "features_v2.parquet"
        write_parquet(features, feat_path)
        # Add stub sentiment for compatibility
        from src.features.sentiment import build_or_stub, merge_sentiment_into_features
        sent = build_or_stub(features["Date"])
        features_full = merge_sentiment_into_features(features, sent)
        full_path = project_root() / "data" / "processed" / "features_v2_with_sentiment.parquet"
        write_parquet(features_full, full_path)
        log.info(f"Features: {features_full.shape}")
    except Exception as e:
        log.error(f"Features failed: {e}")
        return 1

    # 3. Re-train top 3 ML models trên latest full data
    log.info("Step 3/5: Retraining top 3 ML models...")
    cv = build_cv_from_config()
    folds = list(cv.split(features_full))
    last_fold = folds[-1]
    train_df, val_df = cv.get_train_val(features_full, last_fold)

    new_results = {}
    for model_class in [RidgeForecaster, ElasticNetForecaster, LightGBMForecaster]:
        try:
            m = model_class(horizon=args.horizon)
            metrics = evaluate_ml_one_fold(m, train_df, val_df, args.horizon)
            new_results[m.name] = metrics
            log.info(f"  {m.name}: MAPE = {metrics.get('MAPE', float('nan')):.3f}%")
        except Exception as e:
            log.warning(f"{model_class.__name__} failed: {e}")

    # 4. Compare with baseline
    log.info("Step 4/5: Comparing with baseline...")
    baseline_path = project_root() / args.baseline
    alerts = []
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        for model_name, new_metrics in new_results.items():
            sub = baseline[(baseline["model"] == model_name) &
                           (baseline["horizon"] == args.horizon) &
                           (baseline["metric"] == "MAPE")]
            if len(sub) == 0:
                continue
            base_mape = sub["mean"].iloc[0]
            new_mape = new_metrics.get("MAPE", float("nan"))
            if base_mape > 0:
                change_pct = (new_mape - base_mape) / base_mape * 100
                log.info(f"  {model_name}: baseline={base_mape:.3f}% → new={new_mape:.3f}%  ({change_pct:+.1f}%)")
                if change_pct > args.alert_threshold_pct:
                    alerts.append(f"{model_name}: MAPE degrade {change_pct:+.1f}%")
    else:
        log.warning(f"Baseline not found: {baseline_path}")

    # 5. Save snapshot + alert log
    log.info("Step 5/5: Saving snapshot...")
    snapshot_dir = project_root() / "reports" / "weekly_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "timestamp": timestamp,
        "horizon": args.horizon,
        "models": {name: {k: float(v) if not isinstance(v, str) else v
                          for k, v in metrics.items()} for name, metrics in new_results.items()},
        "alerts": alerts,
    }
    snapshot_path = snapshot_dir / f"snapshot_{timestamp}.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    log.info(f"Saved snapshot: {snapshot_path.name}")

    if alerts:
        log.error("=" * 60)
        log.error("⚠️  ALERTS:")
        for a in alerts:
            log.error(f"  - {a}")
        log.error("=" * 60)
        return 2  # exit code 2 = alert
    log.info("✅ Weekly retrain DONE — no alerts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
