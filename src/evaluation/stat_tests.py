"""Statistical tests cho so sánh forecast: Diebold-Mariano + Friedman + Nemenyi.

Reference:
- Diebold & Mariano (1995). Comparing Predictive Accuracy.
- Demšar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================
# DIEBOLD-MARIANO TEST (pairwise)
# ============================================================

def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    h: int = 1,
    loss: str = "MSE",
    one_sided: bool = False,
) -> dict:
    """Diebold-Mariano test cho equal forecast accuracy.

    H0: E[loss(pred_a) - loss(pred_b)] = 0 (equal accuracy)
    H1: hai forecast có accuracy khác nhau (two-sided) hoặc A tốt hơn B (one-sided)

    Parameters
    ----------
    y_true, pred_a, pred_b : np.ndarray, length n
    h : int — forecast horizon (dùng để tính HAC variance, h-1 lags)
    loss : "MSE" | "MAE" | "MAPE"
    one_sided : True → H1: A tốt hơn B (loss_a < loss_b)

    Returns
    -------
    {dm_stat: float, p_value: float, conclusion: str, n: int}
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    if n < 10:
        return {"dm_stat": float("nan"), "p_value": float("nan"),
                "conclusion": "n too small", "n": n}

    err_a = y_true - pred_a
    err_b = y_true - pred_b
    if loss == "MSE":
        d = err_a ** 2 - err_b ** 2
    elif loss == "MAE":
        d = np.abs(err_a) - np.abs(err_b)
    elif loss == "MAPE":
        denom = np.where(np.abs(y_true) >= 1e-10, np.abs(y_true), np.nan)
        d = np.abs(err_a / denom) - np.abs(err_b / denom)
        d = d[~np.isnan(d)]
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_mean = float(np.mean(d))

    # HAC variance (Newey-West) với h-1 lags
    gamma_0 = float(np.var(d, ddof=0))
    var_d = gamma_0
    for k in range(1, h):
        gamma_k = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        var_d += 2 * gamma_k
    var_d = max(var_d, 1e-12)

    dm_stat = d_mean / np.sqrt(var_d / n)

    # Harvey-Leybourne-Newbold small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n) if n > 2 * h else 1.0
    dm_stat *= correction

    if one_sided:
        p_val = stats.norm.cdf(dm_stat)  # H1: d_mean < 0 → A tốt hơn B
    else:
        p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if p_val < 0.01:
        sig = "***"
    elif p_val < 0.05:
        sig = "**"
    elif p_val < 0.10:
        sig = "*"
    else:
        sig = "n.s."

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_val),
        "n": n,
        "loss": loss,
        "horizon": h,
        "significance": sig,
        "conclusion": "A better" if dm_stat < 0 else "B better",
    }


def dm_pairwise_table(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    h: int = 1,
    loss: str = "MSE",
) -> pd.DataFrame:
    """Bảng pairwise DM test. Cell [i,j] = p-value của H0: model_i = model_j (two-sided)."""
    names = list(predictions.keys())
    n = len(names)
    p_matrix = np.full((n, n), np.nan)
    sig_matrix = np.full((n, n), "", dtype=object)

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                p_matrix[i, j] = 1.0
                sig_matrix[i, j] = "—"
                continue
            r = diebold_mariano(y_true, predictions[name_a], predictions[name_b],
                                h=h, loss=loss, one_sided=False)
            p_matrix[i, j] = r["p_value"]
            sig_matrix[i, j] = r["significance"]

    out = pd.DataFrame(p_matrix, index=names, columns=names)
    return out


# ============================================================
# FRIEDMAN TEST + NEMENYI POST-HOC
# ============================================================

def friedman_nemenyi(
    metric_table: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Friedman test + Nemenyi post-hoc.

    Parameters
    ----------
    metric_table : DataFrame shape (n_datasets, n_models) — value càng nhỏ càng tốt
        Trong context của ta: rows = folds, cols = models, values = MAPE/RMSE/etc.
    alpha : significance level

    Returns
    -------
    {friedman_stat, p_value, ranks (mean rank per model),
     nemenyi_cd: critical difference for Nemenyi @ alpha}
    """
    df = metric_table.dropna(axis=1, how="any")
    if df.shape[0] < 2 or df.shape[1] < 2:
        return {"error": "Need ≥ 2 datasets and 2 models"}

    # Friedman
    arrays = [df[c].to_numpy() for c in df.columns]
    fried_stat, p_val = stats.friedmanchisquare(*arrays)

    # Mean ranks (lower better, so rank ascending)
    ranks = df.rank(axis=1, ascending=True).mean(axis=0).sort_values()

    # Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1)/(6N))
    # q_alpha là Studentized range stat — table values cho infinity dof
    n = df.shape[0]
    k = df.shape[1]
    # Approximate q_alpha via studentized range (use scipy if available)
    # Common values for alpha=0.05: q(2)=1.960, q(3)=2.343, q(4)=2.569, q(5)=2.728,
    # q(6)=2.850, q(7)=2.948, q(8)=3.031, q(9)=3.102, q(10)=3.164
    q_table_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                  7: 2.948, 8: 3.031, 9: 3.102, 10: 3.164,
                  11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}
    q_table_10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.589,
                  7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}
    q_alpha = q_table_05.get(k, 3.391) if alpha == 0.05 else q_table_10.get(k, 2.92)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    return {
        "friedman_stat": float(fried_stat),
        "p_value": float(p_val),
        "n_datasets": n,
        "n_models": k,
        "mean_ranks": ranks.to_dict(),
        "nemenyi_cd": float(cd),
        "alpha": alpha,
        "interpretation": "Reject H0 (models differ)" if p_val < alpha else "Cannot reject H0",
    }


# ============================================================
# RUN ALL DM PAIRS FROM LEADERBOARD
# ============================================================

def dm_test_from_predictions_long(predictions_df: pd.DataFrame, loss: str = "MSE") -> pd.DataFrame:
    """Predictions long DataFrame: cols=[model, horizon, fold_id, idx, y_true, y_pred].

    Trả về long DataFrame của pairwise DM tests aggregated qua folds.
    """
    # Group by horizon
    rows = []
    for horizon, hdf in predictions_df.groupby("horizon"):
        # For each model, concat predictions across folds
        model_preds = {}
        y_true_concat = None
        for model, mdf in hdf.groupby("model"):
            mdf_sorted = mdf.sort_values(["fold_id", "idx"])
            model_preds[model] = mdf_sorted["y_pred"].to_numpy()
            if y_true_concat is None:
                y_true_concat = mdf_sorted["y_true"].to_numpy()

        names = list(model_preds.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                # Both arrays same length (assumed across folds)
                if len(model_preds[a]) != len(model_preds[b]):
                    continue
                r = diebold_mariano(y_true_concat, model_preds[a], model_preds[b],
                                     h=int(horizon), loss=loss, one_sided=False)
                rows.append({
                    "horizon": horizon, "model_a": a, "model_b": b,
                    "dm_stat": r["dm_stat"], "p_value": r["p_value"],
                    "significance": r["significance"], "n": r["n"],
                })
    return pd.DataFrame(rows)
