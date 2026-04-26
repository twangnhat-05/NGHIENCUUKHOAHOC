"""Entry point W2.7: chạy classical baselines trên walk-forward CV.

Output:
- reports/leaderboard/classical_long.csv
- reports/leaderboard/classical_summary.csv
- reports/leaderboard/classical_h{1,5,20}_{MAPE,RMSE,MAE,DA}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.evaluation.leaderboard import aggregate_results, save_leaderboard
from src.models.classical import build_classical_models
from src.training.cv import build_cv_from_config
from src.training.trainer import run_walk_forward
from src.utils.io import load_yaml, project_root, read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run classical baselines on walk-forward CV")
    parser.add_argument("--features", default="data/processed/features_v2_with_sentiment.parquet",
                        help="Path tới features parquet")
    parser.add_argument("--cv-config", default="configs/cv.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Override horizons (default từ features.yaml)")
    parser.add_argument("--no-prophet", action="store_true", help="Bỏ qua Prophet (slow)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="classical", help="Output file prefix")
    args = parser.parse_args()

    set_global_seed(args.seed)

    df = read_parquet(args.features)
    log.info(f"Loaded features: {df.shape}, range="
             f"{df['Date'].min().date()}→{df['Date'].max().date()}")

    feat_cfg = load_yaml(args.features_config)
    horizons = args.horizons or feat_cfg["target_horizons"]
    target_col = feat_cfg["target_column"]
    log.info(f"Horizons: {horizons}, target: {target_col}")

    cv = build_cv_from_config(args.cv_config)
    log.info(f"CV: {cv.scheme}, n_folds={cv.n_folds}, init_train={cv.initial_train_size}, "
             f"val={cv.val_size}, step={cv.step_size}")

    models = build_classical_models(freq="B", include_prophet=not args.no_prophet)
    log.info(f"Models: {[m.name for m in models]}")

    results = run_walk_forward(
        df=df, models=models, cv=cv, horizons=horizons, target_col=target_col,
    )
    log.info(f"Total records collected: {len(results)}")

    long_df = aggregate_results(results)
    out_dir = save_leaderboard(long_df, name=args.name)
    log.info(f"Done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
