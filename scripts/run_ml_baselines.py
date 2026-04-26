"""Entry point: chạy ML baselines (Tier 2) trên walk-forward CV.

Output: reports/leaderboard/ml_*.{csv,png}
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.evaluation.leaderboard import aggregate_results, save_leaderboard
from src.models.ml import build_ml_models
from src.training.cv import build_cv_from_config
from src.training.trainer import run_walk_forward
from src.utils.io import load_yaml, read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ML baselines on walk-forward CV")
    parser.add_argument("--features", default="data/processed/features_v2_with_sentiment.parquet")
    parser.add_argument("--cv-config", default="configs/cv.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--include-stacking", action="store_true",
                        help="Thêm StackingRegressor (slow ~2-5 phút/fold)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="ml")
    args = parser.parse_args()

    set_global_seed(args.seed)
    df = read_parquet(args.features)
    log.info(f"Loaded features: {df.shape}, range="
             f"{df['Date'].min().date()}→{df['Date'].max().date()}")

    feat_cfg = load_yaml(args.features_config)
    horizons = args.horizons or feat_cfg["target_horizons"]
    target_col = feat_cfg["target_column"]
    cv = build_cv_from_config(args.cv_config)

    # Build ML models 1 lần — fit lại per fold per horizon (horizon attr sẽ được trainer set)
    # Tạo 1 model object/horizon vì horizon baked-in constructor
    all_results = []
    for h in horizons:
        models = build_ml_models(horizon=h, include_stacking=args.include_stacking)
        log.info(f"Models @ h={h}: {[m.name for m in models]}")
        results = run_walk_forward(df=df, models=models, cv=cv, horizons=[h], target_col=target_col)
        all_results.extend(results)

    log.info(f"Total ML records: {len(all_results)}")
    long_df = aggregate_results(all_results)
    out_dir = save_leaderboard(long_df, name=args.name)
    log.info(f"Done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
