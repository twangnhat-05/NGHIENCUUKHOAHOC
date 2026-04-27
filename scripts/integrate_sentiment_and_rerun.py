"""Integrate scored sentiment vào features → re-run ML leaderboard → compare with old.

Pipeline:
1. Load news_headlines_scored.parquet (đã có signed_score per row)
2. Aggregate daily → sentiment features per Date
3. Merge vào features_v2 → features_v2_real_sentiment.parquet
4. Re-run ML baselines trên features mới
5. So sánh MAPE before (stub zeros) vs after (real sentiment)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.evaluation.leaderboard import aggregate_results, save_leaderboard
from src.features.sentiment import aggregate_daily, merge_sentiment_into_features
from src.models.ml import build_ml_models
from src.training.cv import build_cv_from_config
from src.training.trainer import run_walk_forward
from src.utils.io import load_yaml, project_root, read_parquet, write_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="data/external/news_headlines_scored.parquet")
    parser.add_argument("--features-stub", default="data/processed/features_v2.parquet")
    parser.add_argument("--features-out", default="data/processed/features_v2_real_sentiment.parquet")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--name", default="ml_real_sentiment")
    args = parser.parse_args()

    set_global_seed(42)

    # 1. Load scored news
    p = project_root() / args.scored
    if not p.exists():
        log.error(f"Missing {p}. Chạy news_score.py trước.")
        return 1
    news_df = pd.read_parquet(p)
    log.info(f"Loaded {len(news_df)} scored headlines, range {news_df['date'].min()} → {news_df['date'].max()}")
    log.info(f"Label distribution: {news_df['label'].value_counts().to_dict()}")

    # 2. Aggregate daily
    daily = aggregate_daily(news_df, date_col="date", score_col="signed_score")
    log.info(f"Aggregated to {len(daily)} unique days")
    log.info(daily.head().to_string())

    # 3. Load features (no sentiment) + merge real sentiment
    features = read_parquet(args.features_stub)
    log.info(f"Features (no sentiment): {features.shape}")
    features_with = merge_sentiment_into_features(features, daily)
    log.info(f"Features (with real sentiment): {features_with.shape}")
    write_parquet(features_with, args.features_out)

    # 4. Re-run ML baselines
    feat_cfg = load_yaml("configs/features.yaml")
    target_col = feat_cfg["target_column"]
    cv = build_cv_from_config()
    all_results = []
    for h in args.horizons:
        models = build_ml_models(horizon=h, include_stacking=False)
        log.info(f"\n=== Horizon h={h}: {len(models)} models ===")
        results = run_walk_forward(df=features_with, models=models, cv=cv,
                                   horizons=[h], target_col=target_col)
        all_results.extend(results)

    long_df = aggregate_results(all_results)
    out_dir = save_leaderboard(long_df, name=args.name)

    # 5. Compare with stub baseline
    stub_summary = pd.read_csv(project_root() / "reports" / "leaderboard" / "ml_summary.csv")
    new_summary = pd.read_csv(out_dir / f"{args.name}_summary.csv")
    log.info("\n" + "=" * 70)
    log.info("BEFORE vs AFTER real sentiment (mean MAPE)")
    log.info("=" * 70)
    log.info(f"{'Model':<18s} {'h=1 stub':>10s} {'h=1 real':>10s} {'Δ':>8s} | "
             f"{'h=5 stub':>10s} {'h=5 real':>10s} {'Δ':>8s} | "
             f"{'h=20 stub':>10s} {'h=20 real':>10s} {'Δ':>8s}")
    log.info("-" * 130)
    for model in stub_summary["model"].unique():
        row = [f"{model:<18s}"]
        for h in [1, 5, 20]:
            stub = stub_summary[(stub_summary["model"] == model) & (stub_summary["horizon"] == h) &
                                  (stub_summary["metric"] == "MAPE")]
            new = new_summary[(new_summary["model"] == model) & (new_summary["horizon"] == h) &
                                (new_summary["metric"] == "MAPE")]
            if len(stub) and len(new):
                s_v = stub["mean"].iloc[0]
                n_v = new["mean"].iloc[0]
                delta = n_v - s_v
                row.append(f"{s_v:>9.3f}% {n_v:>9.3f}% {delta:>+7.3f}%")
            else:
                row.append(f"{'n/a':>10s} {'n/a':>10s} {'n/a':>8s}")
        log.info(" | ".join(row))

    log.info(f"\nSaved leaderboard → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
