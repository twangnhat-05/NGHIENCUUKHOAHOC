"""Run foundation models zero-shot benchmark."""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.evaluation.leaderboard import aggregate_results, save_leaderboard
from src.models.foundation import build_foundation_models
from src.training.cv import build_cv_from_config
from src.training.trainer import run_walk_forward
from src.utils.io import load_yaml, read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run foundation models on walk-forward CV")
    parser.add_argument("--features", default="data/processed/features_v2_with_sentiment.parquet")
    parser.add_argument("--cv-config", default="configs/cv.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--include-ttm", action="store_true",
                        help="Thêm TTM (IBM) — needs trust_remote_code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="foundation")
    args = parser.parse_args()

    set_global_seed(args.seed)
    df = read_parquet(args.features)
    log.info(f"Loaded features: {df.shape}")

    feat_cfg = load_yaml(args.features_config)
    horizons = args.horizons or feat_cfg["target_horizons"]
    target_col = feat_cfg["target_column"]
    cv = build_cv_from_config(args.cv_config)

    all_results = []
    for h in horizons:
        models = build_foundation_models(horizon=h, include_ttm=args.include_ttm)
        log.info(f"Models @ h={h}: {[m.name for m in models]}")
        results = run_walk_forward(df=df, models=models, cv=cv, horizons=[h], target_col=target_col)
        all_results.extend(results)

    log.info(f"Total foundation records: {len(all_results)}")
    long_df = aggregate_results(all_results)
    out_dir = save_leaderboard(long_df, name=args.name)
    log.info(f"Done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
