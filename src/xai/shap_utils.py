"""SHAP (SHapley Additive exPlanations) utilities cho ML models.

TreeExplainer cho XGBoost/LightGBM/CatBoost/RandomForest (fast, exact).
KernelExplainer cho Ridge/SVR/etc. (slow, approximate).

References:
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
- Lundberg et al. (2020). From local explanations to global understanding with explainable AI for trees.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


def compute_shap_tree(
    estimator: Any,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    sample_size: int | None = 500,
    seed: int = 42,
) -> dict:
    """SHAP TreeExplainer cho tree-based model.

    Returns dict: {shap_values, base_value, feature_importance (mean abs), feature_names}
    """
    import shap

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    feature_names = feature_names or (list(X.columns) if isinstance(X, pd.DataFrame) else
                                       [f"f{i}" for i in range(X_arr.shape[1])])

    # Sample to speed up (TreeExplainer fast nhưng KernelExplainer chậm)
    if sample_size and sample_size < len(X_arr):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_arr), size=sample_size, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):  # multi-class
            shap_values = shap_values[0]
        base_value = float(np.atleast_1d(explainer.expected_value)[0])
    except Exception as e:
        log.error(f"TreeExplainer failed: {e}")
        return {"error": str(e)}

    fi = np.abs(shap_values).mean(axis=0)
    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "feature_importance": fi,
        "feature_names": feature_names,
        "sample_size": len(X_sample),
    }


def shap_top_features(shap_result: dict, top_k: int = 20) -> pd.DataFrame:
    """Trả về top-K features theo |SHAP| trung bình."""
    if "error" in shap_result:
        return pd.DataFrame()
    df = pd.DataFrame({
        "feature": shap_result["feature_names"],
        "mean_abs_shap": shap_result["feature_importance"],
    })
    return df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)


def plot_shap_summary(
    shap_result: dict,
    output_path: str,
    plot_type: str = "bar",
    max_display: int = 20,
) -> None:
    """Plot SHAP summary, save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if "error" in shap_result:
        log.warning(f"Cannot plot SHAP: {shap_result['error']}")
        return

    plt.figure(figsize=(10, max(6, max_display * 0.3)))
    shap.summary_plot(
        shap_result["shap_values"],
        features=None,  # we don't pass feature values for simplicity
        feature_names=shap_result["feature_names"],
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Saved SHAP plot: {output_path}")
