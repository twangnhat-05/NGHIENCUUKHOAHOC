"""Forecasting metrics — point + probabilistic + financial."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# POINT FORECAST METRICS
# ============================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error (%). Bỏ qua giá trị y_true ~ 0."""
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_t) >= epsilon
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Symmetric MAPE (%). Range [0, 200]."""
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    mask = denom >= epsilon
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_t[mask] - y_p[mask]) / denom[mask]) * 100.0)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Hyndman & Koehler (2006). Scale errors by in-sample MAE of seasonal naive
    on training data.
    """
    y_t, y_p, y_tr = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_train)
    if len(y_tr) <= seasonal_period:
        raise ValueError("len(y_train) phải > seasonal_period")
    naive_errors = np.abs(y_tr[seasonal_period:] - y_tr[:-seasonal_period])
    scale = np.mean(naive_errors)
    if scale < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(y_t - y_p)) / scale)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ============================================================
# DIRECTIONAL / FINANCIAL METRICS
# ============================================================

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """% lần dự báo đúng hướng (lên/xuống) so với giá trị trước.

    y_prev: giá trị tại t (để so y_true tại t+h và y_pred tại t+h cùng có hướng so với t).
    """
    y_t, y_p, y_pv = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prev)
    actual_dir = np.sign(y_t - y_pv)
    pred_dir = np.sign(y_p - y_pv)
    # Bỏ trường hợp giá trị bằng nhau (sign = 0)
    mask = (actual_dir != 0) & (pred_dir != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(actual_dir[mask] == pred_dir[mask]) * 100.0)


def hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray,
    threshold_pct: float = 0.5,
) -> float:
    """% lần dự báo đúng hướng VÀ độ lớn return ≥ threshold_pct (%)."""
    y_t, y_p, y_pv = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prev)
    actual_ret = (y_t - y_pv) / np.where(np.abs(y_pv) >= 1e-10, y_pv, np.nan) * 100.0
    pred_ret = (y_p - y_pv) / np.where(np.abs(y_pv) >= 1e-10, y_pv, np.nan) * 100.0
    mask = ~(np.isnan(actual_ret) | np.isnan(pred_ret))
    actual_ret, pred_ret = actual_ret[mask], pred_ret[mask]
    if len(actual_ret) == 0:
        return float("nan")
    correct = (np.sign(actual_ret) == np.sign(pred_ret)) & (np.abs(actual_ret) >= threshold_pct)
    return float(np.mean(correct) * 100.0)


# ============================================================
# PROBABILISTIC METRICS
# ============================================================

def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Continuous Ranked Probability Score giả sử predictive Gaussian.

    Gneiting & Raftery (2007). Closed-form CRPS for Normal:
    CRPS = sigma * (z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
    where z = (y - mu) / sigma.
    """
    y_t, mu, sigma = np.asarray(y_true), np.asarray(mu), np.asarray(sigma)
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    z = (y_t - mu) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


# ============================================================
# CONVENIENCE: SUMMARY DICT
# ============================================================

def summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    y_prev: np.ndarray | None = None,
    seasonal_period: int = 5,
) -> dict[str, float]:
    """Trả về dict tất cả metrics dùng được với input cung cấp."""
    out: dict[str, float] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r_squared(y_true, y_pred),
    }
    if y_train is not None:
        try:
            out["MASE"] = mase(y_true, y_pred, y_train, seasonal_period=seasonal_period)
        except ValueError:
            out["MASE"] = float("nan")
    if y_prev is not None:
        out["DA"] = directional_accuracy(y_true, y_pred, y_prev)
        out["HitRate_0.5pct"] = hit_rate(y_true, y_pred, y_prev, threshold_pct=0.5)
    return out


def metrics_to_dataframe(
    metrics_per_model: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Convert {model_name: {metric: value}} → DataFrame (rows=model, cols=metric)."""
    return pd.DataFrame(metrics_per_model).T
