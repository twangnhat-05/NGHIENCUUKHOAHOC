"""Entry point: chạy DL models (Tier-3) trên walk-forward CV.

Bao gồm:
- Custom: LSTM v2, GRU (PyTorch sequence)
- NeuralForecast SOTA: N-HiTS, N-BEATS, PatchTST, TimeMixer, TSMixer
- Optional: iTransformer, TFT (CPU slow ~60-120s/fold)

Usage:
    python scripts/run_dl_baselines.py --horizons 1 5 20 --fast
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings

# Silence noisy libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
for log_name in ("pytorch_lightning", "lightning_fabric", "lightning", "neuralforecast"):
    logging.getLogger(log_name).setLevel(logging.ERROR)

from src.evaluation.leaderboard import aggregate_results, save_leaderboard
from src.models.dl_neuralforecast import build_neuralforecast_models
from src.models.dl_simple import build_dl_simple_models
from src.training.cv import build_cv_from_config
from src.training.trainer import run_walk_forward
from src.utils.io import load_yaml, read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DL models on walk-forward CV")
    parser.add_argument("--features", default="data/processed/features_v2_with_sentiment.parquet")
    parser.add_argument("--cv-config", default="configs/cv.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Bỏ iTransformer + TFT (slow trên CPU)")
    parser.add_argument("--include-simple", action="store_true",
                        help="Thêm LSTM v2 + GRU (custom PyTorch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="dl")
    args = parser.parse_args()

    set_global_seed(args.seed)
    df = read_parquet(args.features)
    log.info(f"Loaded features: {df.shape}, range="
             f"{df['Date'].min().date()}→{df['Date'].max().date()}")

    feat_cfg = load_yaml(args.features_config)
    horizons = args.horizons or feat_cfg["target_horizons"]
    target_col = feat_cfg["target_column"]
    cv = build_cv_from_config(args.cv_config)

    all_results = []
    for h in horizons:
        models = build_neuralforecast_models(horizon=h, fast_only=args.fast)
        if args.include_simple:
            models = build_dl_simple_models(horizon=h) + models
        log.info(f"Models @ h={h}: {[m.name for m in models]}")
        results = run_walk_forward(df=df, models=models, cv=cv, horizons=[h], target_col=target_col)
        all_results.extend(results)

    log.info(f"Total DL records: {len(all_results)}")
    long_df = aggregate_results(all_results)
    out_dir = save_leaderboard(long_df, name=args.name)
    log.info(f"Done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
