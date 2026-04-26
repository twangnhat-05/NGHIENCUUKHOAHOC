"""Unit tests cho src/evaluation/metrics.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    crps_gaussian,
    directional_accuracy,
    hit_rate,
    mae,
    mape,
    mase,
    r_squared,
    rmse,
    smape,
    summary,
)


def test_mae_perfect_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_rmse_known() -> None:
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mape_basic() -> None:
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    # |10/100| + |10/200| = 0.10 + 0.05 = 0.15 / 2 = 0.075 → 7.5%
    assert mape(y_true, y_pred) == pytest.approx(7.5)


def test_smape_symmetric() -> None:
    y_true = np.array([100.0])
    y_pred = np.array([90.0])
    # |100-90| / ((100+90)/2) = 10/95 ≈ 0.1053 → 10.526%
    assert smape(y_true, y_pred) == pytest.approx(10.5263, rel=1e-4)


def test_mase_basic() -> None:
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.0, 7.0])
    val = mase(y_true, y_pred, y_train, seasonal_period=1)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_mase_invalid_train_length() -> None:
    with pytest.raises(ValueError):
        mase(np.array([1, 2]), np.array([1, 2]), np.array([1.0]), seasonal_period=5)


def test_r2_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r2_baseline_mean() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y, y.mean())
    assert r_squared(y, y_pred) == pytest.approx(0.0)


def test_directional_accuracy_perfect_up() -> None:
    y_prev = np.array([100.0, 100.0, 100.0])
    y_true = np.array([101.0, 99.0, 102.0])  # up, down, up
    y_pred = np.array([105.0, 98.0, 110.0])  # up, down, up
    assert directional_accuracy(y_true, y_pred, y_prev) == pytest.approx(100.0)


def test_directional_accuracy_all_wrong() -> None:
    y_prev = np.array([100.0, 100.0, 100.0])
    y_true = np.array([101.0, 99.0, 102.0])
    y_pred = np.array([95.0, 105.0, 99.0])  # wrong, wrong, wrong
    assert directional_accuracy(y_true, y_pred, y_prev) == pytest.approx(0.0)


def test_hit_rate_threshold() -> None:
    y_prev = np.array([100.0, 100.0, 100.0])
    y_true = np.array([100.3, 102.0, 99.0])
    y_pred = np.array([100.4, 102.5, 99.0])
    # Returns: actual = [0.3, 2.0, -1.0], pred = [0.4, 2.5, -1.0]
    # threshold 0.5: bỏ obs 1 (|0.3| < 0.5)
    # obs 2: same sign, |2.0| ≥ 0.5 → hit
    # obs 3: same sign, |-1.0| ≥ 0.5 → hit
    assert hit_rate(y_true, y_pred, y_prev, threshold_pct=0.5) == pytest.approx(2 / 3 * 100.0)


def test_crps_gaussian_perfect_point() -> None:
    """Khi sigma → 0, CRPS → MAE (deterministic forecast)."""
    y_true = np.array([10.0, 20.0])
    mu = np.array([10.0, 20.0])
    sigma = np.array([1e-10, 1e-10])
    val = crps_gaussian(y_true, mu, sigma)
    # gần 0 (perfect forecast)
    assert abs(val) < 1e-3


def test_summary_returns_dict_with_known_keys() -> None:
    y_true = np.array([100.0, 102.0, 101.0])
    y_pred = np.array([99.0, 102.5, 100.5])
    y_train = np.array([95.0, 96.0, 97.0, 98.0, 99.0, 100.0])
    y_prev = np.array([99.5, 100.0, 100.5])
    out = summary(y_true, y_pred, y_train=y_train, y_prev=y_prev, seasonal_period=1)
    for key in ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE", "DA", "HitRate_0.5pct"):
        assert key in out
        assert not math.isnan(out[key]) or key in ("MASE",)  # MASE có thể nan nếu scale = 0
