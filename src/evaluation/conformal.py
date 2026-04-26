"""Conformal prediction intervals cho time series.

Methods:
- Split conformal (vanilla): chia residuals từ calibration set, take quantile
- ACI (Adaptive Conformal Inference, Gibbs & Candès 2021): adapt alpha online
- EnbPI (Xu & Xie 2021): bootstrap-based, no holdout

For volatile asset như SJC: ACI khuyến nghị (auto adapt khi distribution shift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================
# SPLIT CONFORMAL (simplest)
# ============================================================

def split_conformal_intervals(
    cal_y_true: np.ndarray,
    cal_y_pred: np.ndarray,
    test_y_pred: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal: returns (lower, upper) intervals cho test predictions.

    H = quantile (1-alpha) of |residuals| trên calibration set.
    Interval = [pred - H, pred + H].

    Coverage guarantee: P(y_test in interval) ≥ 1 - alpha (under exchangeability).
    """
    residuals = np.abs(cal_y_true - cal_y_pred)
    n = len(residuals)
    # Adjusted quantile cho finite-sample
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(max(q_level, 0.0), 1.0)
    H = np.quantile(residuals, q_level)
    return test_y_pred - H, test_y_pred + H


# ============================================================
# ACI - Adaptive Conformal Inference
# ============================================================

def aci_intervals(
    y_true_stream: np.ndarray,
    y_pred_stream: np.ndarray,
    initial_alpha: float = 0.1,
    gamma: float = 0.005,
    warmup: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive Conformal Inference (Gibbs & Candès 2021).

    Online: tại mỗi t, adjust alpha_t = alpha_{t-1} + gamma * (alpha_target - err_t).
    Non-coverage err_t = 1 if y_t outside interval.

    Returns (lower, upper, alpha_trace).
    """
    n = len(y_true_stream)
    alpha_target = initial_alpha
    alpha_t = alpha_target
    lower = np.zeros(n)
    upper = np.zeros(n)
    alpha_trace = np.zeros(n)

    residuals_history = []
    for t in range(n):
        if t < warmup:
            # No conformal yet — just point + dummy
            residuals_history.append(abs(y_true_stream[t] - y_pred_stream[t]))
            lower[t] = y_pred_stream[t] - 1.0
            upper[t] = y_pred_stream[t] + 1.0
            alpha_trace[t] = alpha_t
            continue

        # Compute current quantile from residuals
        q_level = max(0.0, min(1.0, 1 - alpha_t))
        H = np.quantile(residuals_history, q_level)
        lower[t] = y_pred_stream[t] - H
        upper[t] = y_pred_stream[t] + H
        alpha_trace[t] = alpha_t

        # Update alpha based on coverage of t
        err = 0 if (lower[t] <= y_true_stream[t] <= upper[t]) else 1
        alpha_t = alpha_t + gamma * (alpha_target - err)
        alpha_t = max(0.001, min(0.999, alpha_t))

        # Update residuals history
        residuals_history.append(abs(y_true_stream[t] - y_pred_stream[t]))

    return lower, upper, alpha_trace


# ============================================================
# COVERAGE METRICS
# ============================================================

def coverage_rate(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """% lần y_true rơi vào [lower, upper]."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside) * 100.0)


def average_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Trung bình width của intervals (proxy cho informativeness — càng hẹp càng tốt nếu coverage đủ)."""
    return float(np.mean(upper - lower))


def conformal_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cal_y_true: np.ndarray,
    cal_y_pred: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Báo cáo coverage tại nhiều mức alpha cho split conformal."""
    rows = []
    for a in alphas:
        lower, upper = split_conformal_intervals(cal_y_true, cal_y_pred, y_pred, alpha=a)
        rows.append({
            "alpha": a,
            "expected_coverage": (1 - a) * 100,
            "actual_coverage": coverage_rate(y_true, lower, upper),
            "avg_width": average_interval_width(lower, upper),
        })
    return pd.DataFrame(rows)
