"""Merge classical + ml + dl leaderboards → reports/leaderboard/combined_*.{csv,png}.

Run Friedman test trên per-fold MAPE để check statistical significance giữa models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.leaderboard import save_leaderboard, summary_per_model
from src.evaluation.stat_tests import friedman_nemenyi
from src.utils.io import project_root
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine leaderboards + Friedman test")
    parser.add_argument("--inputs", nargs="+",
                        default=["classical_full_long.csv", "ml_long.csv", "dl_long.csv"])
    parser.add_argument("--output-name", default="combined")
    args = parser.parse_args()

    leaderboard_dir = project_root() / "reports" / "leaderboard"
    dfs = []
    for fname in args.inputs:
        p = leaderboard_dir / fname
        if not p.exists():
            log.warning(f"Skip missing: {p.name}")
            continue
        df = pd.read_csv(p)
        df["source"] = fname.replace("_long.csv", "")
        dfs.append(df)
    if not dfs:
        log.error("No leaderboards found")
        return 1
    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Combined: {len(combined)} records, models: {combined['model'].nunique()}")

    # Save combined
    out_dir = save_leaderboard(combined, name=args.output_name)
    log.info(f"Combined leaderboard → {out_dir}")

    # Friedman test per horizon (using MAPE)
    log.info("\n" + "=" * 70)
    log.info("FRIEDMAN TEST (per horizon, on MAPE per fold)")
    log.info("=" * 70)
    friedman_results = []
    for horizon in sorted(combined["horizon"].unique()):
        sub = combined[(combined["horizon"] == horizon) & (combined["metric"] == "MAPE")]
        # Pivot: rows=fold_id, cols=model, values=MAPE
        pivot = sub.pivot_table(index="fold_id", columns="model", values="value", aggfunc="mean")
        # Drop columns với NaN
        pivot = pivot.dropna(axis=1)
        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            log.warning(f"Horizon {horizon}: not enough data")
            continue
        r = friedman_nemenyi(pivot, alpha=0.05)
        log.info(f"\nHorizon h={horizon}: n_folds={pivot.shape[0]}, n_models={pivot.shape[1]}")
        log.info(f"  Friedman stat = {r['friedman_stat']:.4f}, p-value = {r['p_value']:.6f}")
        log.info(f"  → {r['interpretation']}")
        log.info(f"  Nemenyi critical difference @ alpha=0.05: {r['nemenyi_cd']:.4f}")
        log.info(f"  Top-5 mean ranks (lower better):")
        ranks = sorted(r['mean_ranks'].items(), key=lambda kv: kv[1])
        for name, rank in ranks[:5]:
            log.info(f"    {name:25s} mean_rank = {rank:.3f}")
        friedman_results.append({
            "horizon": horizon, "n_folds": pivot.shape[0], "n_models": pivot.shape[1],
            **{k: v for k, v in r.items() if k != "mean_ranks"},
            "best_model": ranks[0][0], "best_rank": ranks[0][1],
        })

    if friedman_results:
        fr_df = pd.DataFrame(friedman_results)
        fr_path = out_dir / "friedman_test.csv"
        fr_df.to_csv(fr_path, index=False, encoding="utf-8-sig")
        log.info(f"\nFriedman summary → {fr_path}")

    # Top-N MAPE summary table per horizon
    log.info("\n" + "=" * 70)
    log.info("TOP-15 MAPE leaderboard per horizon (combined)")
    log.info("=" * 70)
    summary = summary_per_model(combined)
    for horizon in sorted(combined["horizon"].unique()):
        sub = summary[(summary["horizon"] == horizon) & (summary["metric"] == "MAPE")]
        sub = sub.sort_values("mean").head(15)
        log.info(f"\nHorizon h={horizon}:")
        log.info(f"{'Model':<25s} {'Mean MAPE':>10s} {'Std':>8s} {'Folds':>6s}")
        log.info("-" * 53)
        for _, r in sub.iterrows():
            log.info(f"{r['model']:<25s} {r['mean']:>9.3f}%  {r['std']:>7.3f}  {int(r['count']):>6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
