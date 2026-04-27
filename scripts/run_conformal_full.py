"""Conformal Coverage full report: ACI cho top-3 ML models trên TẤT CẢ 5 folds.

Output:
- reports/figures/conformal_full_table.csv (model × horizon × fold × method × coverage × width)
- reports/figures/conformal_full_summary.csv (mean coverage / width per model & horizon)
- reports/figures/conformal_full_per_horizon.png (visual)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.evaluation.conformal import aci_intervals, coverage_rate, split_conformal_intervals
from src.models.ml import ElasticNetForecaster, LightGBMForecaster, RidgeForecaster
from src.training.cv import build_cv_from_config
from src.utils.io import project_root, read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)


def evaluate_conformal_one_fold(
    model_class, model_kwargs: dict,
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    horizon: int, alpha: float, target_col: str = "SJC_ban_ra",
) -> dict:
    """Fit model trên train (split for cal), predict val, compute conformal intervals."""
    n_train = len(train_df)
    cal_size = max(100, n_train // 8)
    fit_df = train_df.iloc[:n_train - cal_size]
    cal_df = train_df.iloc[n_train - cal_size:]

    target_h_col = f"y_h{horizon}"
    if target_h_col not in cal_df.columns:
        cal_df = cal_df.copy()
        cal_df[target_h_col] = cal_df[target_col].shift(-horizon)
    if target_h_col not in val_df.columns:
        val_df = val_df.copy()
        val_df[target_h_col] = val_df[target_col].shift(-horizon)

    # Fit on smaller train, predict cal
    m = model_class(horizon=horizon, **model_kwargs)
    m.fit(fit_df, target_col=target_col)
    cal_eval = cal_df.dropna(subset=[target_h_col])
    cal_y_pred = m.predict(cal_eval)
    cal_y_true = cal_eval[target_h_col].to_numpy()
    n_cal = min(len(cal_y_pred), len(cal_y_true))
    cal_y_pred, cal_y_true = cal_y_pred[:n_cal], cal_y_true[:n_cal]

    # Refit on full train, predict val
    m.fit(train_df, target_col=target_col)
    val_eval = val_df.dropna(subset=[target_h_col])
    test_y_pred = m.predict(val_eval)
    test_y_true = val_eval[target_h_col].to_numpy()
    n_test = min(len(test_y_pred), len(test_y_true))
    test_y_pred, test_y_true = test_y_pred[:n_test], test_y_true[:n_test]

    # Split conformal
    s_lower, s_upper = split_conformal_intervals(cal_y_true, cal_y_pred, test_y_pred, alpha=alpha)
    s_cov = coverage_rate(test_y_true, s_lower, s_upper)
    s_width = float(np.mean(s_upper - s_lower))

    # ACI (online adaptive)
    a_lower, a_upper, _ = aci_intervals(test_y_true, test_y_pred, initial_alpha=alpha,
                                         gamma=0.005, warmup=10)
    a_cov = coverage_rate(test_y_true, a_lower, a_upper)
    a_width = float(np.mean(a_upper - a_lower))

    return {
        "split_coverage": s_cov, "split_width": s_width,
        "aci_coverage": a_cov, "aci_width": a_width,
        "n_test": n_test,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features_v2_with_sentiment.parquet")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    set_global_seed(42)
    df = read_parquet(args.features)
    cv = build_cv_from_config()
    folds = list(cv.split(df))

    models = {
        "Ridge":      (RidgeForecaster, {}),
        "ElasticNet": (ElasticNetForecaster, {}),
        "LightGBM":   (LightGBMForecaster, {"n_estimators": 200}),
    }

    rows = []
    for h in args.horizons:
        log.info(f"\n=== HORIZON h={h} (alpha={args.alpha}, target coverage {(1-args.alpha)*100:.0f}%) ===")
        for model_name, (cls, kw) in models.items():
            for fold in folds:
                tr, va = cv.get_train_val(df, fold)
                try:
                    r = evaluate_conformal_one_fold(cls, kw, tr, va, h, args.alpha)
                    log.info(f"  {model_name:12s} fold {fold.fold_id} → "
                             f"split cov={r['split_coverage']:5.1f}% w={r['split_width']:.2f} | "
                             f"ACI cov={r['aci_coverage']:5.1f}% w={r['aci_width']:.2f}")
                    rows.append({"model": model_name, "horizon": h, "fold_id": fold.fold_id, **r})
                except Exception as e:
                    log.warning(f"{model_name} fold {fold.fold_id} h={h}: {e}")

    if not rows:
        log.error("No results")
        return 1

    df_full = pd.DataFrame(rows)
    out_dir = project_root() / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(out_dir / "conformal_full_table.csv", index=False, encoding="utf-8-sig")

    # Summary
    summary = df_full.groupby(["horizon", "model"]).agg({
        "split_coverage": "mean", "split_width": "mean",
        "aci_coverage": "mean", "aci_width": "mean",
    }).reset_index()
    summary.to_csv(out_dir / "conformal_full_summary.csv", index=False, encoding="utf-8-sig")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    target_cov = (1 - args.alpha) * 100
    for ax, h in zip(axes, args.horizons):
        sub = summary[summary["horizon"] == h]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, sub["split_coverage"], w, label="Split conformal", alpha=0.85)
        ax.bar(x + w/2, sub["aci_coverage"], w, label="ACI", alpha=0.85)
        ax.axhline(target_cov, color="red", ls="--", label=f"Target {target_cov:.0f}%")
        ax.set_xticks(x); ax.set_xticklabels(sub["model"], rotation=15)
        ax.set_ylabel("Actual coverage (%)")
        ax.set_title(f"Horizon h={h}")
        ax.set_ylim(0, 105)
        ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = out_dir / "conformal_full_per_horizon.png"
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"\nSaved: {plot_path.name}")

    log.info("\n=== SUMMARY (mean across 5 folds) ===")
    log.info(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
