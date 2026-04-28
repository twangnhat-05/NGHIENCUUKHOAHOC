"""Demo: SHAP cho top ML model + Conformal PI cho top 3 models, save to reports/.

Sử dụng fold cuối cùng (most recent val period) cho meaningful interpretation.
"""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from src.evaluation.conformal import conformal_summary, split_conformal_intervals, aci_intervals
from src.evaluation.metrics import summary as metrics_summary
from src.models.classical import build_classical_models
from src.models.ml import (
    ElasticNetForecaster,
    LightGBMForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)
from src.training.cv import WalkForwardCV, build_cv_from_config
from src.training.trainer import evaluate_ml_one_fold, evaluate_one_fold
from src.utils.io import load_yaml, project_root, read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed
from src.xai.shap_utils import compute_shap_tree, plot_shap_summary, shap_top_features

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="XAI + Conformal PI demo")
    parser.add_argument("--features", default="data/processed/features_v2_with_sentiment.parquet")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    df = read_parquet(args.features)
    cv = build_cv_from_config()
    folds = list(cv.split(df))
    last_fold = folds[-1]
    train_df, val_df = cv.get_train_val(df, last_fold)
    log.info(f"Last fold: train {train_df['Date'].min().date()}→{train_df['Date'].max().date()}, "
             f"val {val_df['Date'].min().date()}→{val_df['Date'].max().date()}")

    h = args.horizon
    target_h_col = f"y_h{h}"
    val_eval = val_df.dropna(subset=[target_h_col]).copy() if target_h_col in val_df.columns else val_df.copy()

    # ============================================================
    # A. SHAP cho LightGBM (TreeExplainer)
    # ============================================================
    log.info("\n" + "=" * 60)
    log.info(f"SHAP analysis: LightGBM (h={h})")
    log.info("=" * 60)
    lgbm = LightGBMForecaster(horizon=h, n_estimators=300)
    lgbm.fit(train_df, target_col="SJC_ban_ra")
    feature_cols = lgbm._feature_cols
    X_train = train_df[feature_cols].ffill().fillna(0.0).values

    shap_result = compute_shap_tree(lgbm._estimator, X_train, feature_names=feature_cols, sample_size=300)
    if "error" not in shap_result:
        top = shap_top_features(shap_result, top_k=20)
        log.info(f"Top 20 features (mean |SHAP|):")
        for _, r in top.iterrows():
            log.info(f"  {r['feature']:30s} {r['mean_abs_shap']:.4f}")
        out_path = project_root() / "reports" / "figures" / f"shap_lightgbm_h{h}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_shap_summary(shap_result, str(out_path), plot_type="bar", max_display=20)

        # Save top features CSV
        top.to_csv(project_root() / "reports" / "figures" / f"shap_lightgbm_h{h}_top20.csv",
                   index=False, encoding="utf-8-sig")

    # ============================================================
    # B. Conformal Prediction Intervals
    # ============================================================
    log.info("\n" + "=" * 60)
    log.info(f"Conformal PI: ElasticNet (h={h})")
    log.info("=" * 60)
    en = ElasticNetForecaster(horizon=h)
    # Calibration: split train trong fold
    n_train = len(train_df)
    cal_size = 200
    fit_df = train_df.iloc[:n_train - cal_size]
    cal_df = train_df.iloc[n_train - cal_size:]
    en.fit(fit_df, target_col="SJC_ban_ra")
    cal_y_pred = en.predict(cal_df.dropna(subset=[target_h_col] if target_h_col in cal_df.columns else []))
    cal_y_true = cal_df[target_h_col].dropna().to_numpy() if target_h_col in cal_df.columns else \
                 cal_df["SJC_ban_ra"].shift(-h).dropna().to_numpy()
    n_cal = min(len(cal_y_pred), len(cal_y_true))
    cal_y_pred, cal_y_true = cal_y_pred[:n_cal], cal_y_true[:n_cal]

    # Refit trên full train
    en.fit(train_df, target_col="SJC_ban_ra")
    test_y_pred = en.predict(val_eval)
    test_y_true = val_eval[target_h_col].to_numpy()
    n_test = min(len(test_y_pred), len(test_y_true))
    test_y_pred, test_y_true = test_y_pred[:n_test], test_y_true[:n_test]

    # Split conformal
    cov_summary = conformal_summary(test_y_true, test_y_pred, cal_y_true, cal_y_pred,
                                     alphas=(0.05, 0.10, 0.20))
    log.info(f"\nSplit conformal coverage (ElasticNet h={h}):")
    log.info(cov_summary.to_string(index=False))
    cov_summary.to_csv(project_root() / "reports" / "figures" / f"conformal_split_elasticnet_h{h}.csv",
                       index=False, encoding="utf-8-sig")

    # ACI (online adaptive)
    aci_lower, aci_upper, alpha_trace = aci_intervals(
        test_y_true, test_y_pred, initial_alpha=0.10, gamma=0.005, warmup=20,
    )
    inside = (test_y_true >= aci_lower) & (test_y_true <= aci_upper)
    actual_cov = float(np.mean(inside) * 100)
    avg_w = float(np.mean(aci_upper - aci_lower))
    log.info(f"\nACI (alpha=0.10, gamma=0.005): coverage={actual_cov:.1f}%, avg_width={avg_w:.3f}")

    # Plot ACI
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(test_y_true))
    axes[0].plot(x, test_y_true, "k-", label="Actual", linewidth=1)
    axes[0].plot(x, test_y_pred, "C0-", label="ElasticNet pred", linewidth=1)
    axes[0].fill_between(x, aci_lower, aci_upper, alpha=0.25, color="C2",
                         label=f"ACI 90% PI (cov={actual_cov:.1f}%)")
    axes[0].set_ylabel("SJC ban_ra (triệu VND/lượng)")
    axes[0].set_title(f"ElasticNet + ACI Conformal PI (h={h}, fold={last_fold.fold_id})")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(x, alpha_trace, "C3-")
    axes[1].axhline(0.10, color="gray", ls="--", label="target alpha")
    axes[1].set_ylabel("alpha_t (ACI)")
    axes[1].set_xlabel("val day index")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    aci_plot_path = project_root() / "reports" / "figures" / f"aci_conformal_elasticnet_h{h}.png"
    plt.savefig(aci_plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {aci_plot_path.name}")

    log.info("\n" + "=" * 60)
    log.info("XAI + Conformal demo COMPLETE")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
