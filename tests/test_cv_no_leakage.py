"""CRITICAL no-leakage tests cho walk-forward CV.

Mỗi test gắn marker `no_leakage` để chạy riêng:
    pytest -m no_leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.cv import Fold, WalkForwardCV, build_cv_from_config, split_test_holdout


pytestmark = pytest.mark.no_leakage


# ============================================================
# 1. Train không chứa val (no overlap)
# ============================================================

def test_train_val_no_overlap(synthetic_daily_df: pd.DataFrame) -> None:
    cv = WalkForwardCV(n_folds=5, initial_train_size=500, val_size=100, step_size=100)
    for fold in cv.split(synthetic_daily_df):
        overlap = set(fold.train_idx) & set(fold.val_idx)
        assert len(overlap) == 0, f"Fold {fold.fold_id}: train/val overlap = {overlap}"


# ============================================================
# 2. Train luôn TRƯỚC val theo thời gian
# ============================================================

def test_train_dates_before_val(synthetic_daily_df: pd.DataFrame) -> None:
    cv = WalkForwardCV(n_folds=5, initial_train_size=500, val_size=100, step_size=100)
    df_sorted = synthetic_daily_df.sort_values("Date").reset_index(drop=True)
    for fold in cv.split(synthetic_daily_df):
        train_max_date = pd.to_datetime(df_sorted.loc[fold.train_idx, "Date"]).max()
        val_min_date = pd.to_datetime(df_sorted.loc[fold.val_idx, "Date"]).min()
        assert train_max_date < val_min_date, (
            f"Fold {fold.fold_id}: leakage — train_max ({train_max_date}) "
            f">= val_min ({val_min_date})"
        )


# ============================================================
# 3. Expanding window: train luôn lớn dần
# ============================================================

def test_expanding_window_grows(synthetic_daily_df: pd.DataFrame) -> None:
    cv = WalkForwardCV(
        n_folds=5, initial_train_size=500, val_size=100, step_size=100, scheme="expanding",
    )
    sizes = [len(f.train_idx) for f in cv.split(synthetic_daily_df)]
    for prev, curr in zip(sizes, sizes[1:]):
        assert curr > prev, f"Expanding fail: train size {prev} → {curr}"


# ============================================================
# 4. Rolling window: train size luôn = initial_train_size
# ============================================================

def test_rolling_window_constant_size(synthetic_daily_df: pd.DataFrame) -> None:
    cv = WalkForwardCV(
        n_folds=5, initial_train_size=500, val_size=100, step_size=100, scheme="rolling",
    )
    sizes = [len(f.train_idx) for f in cv.split(synthetic_daily_df)]
    assert all(s == 500 for s in sizes), f"Rolling fail: sizes = {sizes}"


# ============================================================
# 5. Splits luôn yield > 0 fold (sanity)
# ============================================================

def test_yields_at_least_one_fold(synthetic_daily_df: pd.DataFrame) -> None:
    cv = WalkForwardCV(n_folds=3, initial_train_size=500, val_size=50, step_size=50)
    folds = list(cv.split(synthetic_daily_df))
    assert len(folds) >= 1


# ============================================================
# 6. Test holdout không leak vào train_val
# ============================================================

def test_holdout_excluded_from_trainval(synthetic_daily_df: pd.DataFrame) -> None:
    train_val, test = split_test_holdout(
        synthetic_daily_df, test_start="2023-01-01", test_end="2024-01-01",
    )
    assert pd.to_datetime(train_val["Date"]).max() < pd.Timestamp("2023-01-01")
    assert pd.to_datetime(test["Date"]).min() >= pd.Timestamp("2023-01-01")
    assert pd.to_datetime(test["Date"]).max() <= pd.Timestamp("2024-01-01")


# ============================================================
# 7. Scaler refit-per-fold semantics — verify behavior contract
# ============================================================

def test_scaler_must_fit_only_on_train(synthetic_daily_df: pd.DataFrame) -> None:
    """Demonstrate that fitting StandardScaler on train slice → applying to val
    gives DIFFERENT results than fitting on full data. Đây là contract test:
    nếu user fit_on_full → leakage; user phải fit_on_train_per_fold."""
    from sklearn.preprocessing import StandardScaler

    cv = WalkForwardCV(n_folds=3, initial_train_size=500, val_size=100, step_size=100)
    df = synthetic_daily_df.copy()
    feature_col = "Gold_Close"

    # WRONG (leakage): fit trên full
    bad_scaler = StandardScaler().fit(df[[feature_col]].values)
    bad_mean = bad_scaler.mean_[0]

    # CORRECT: fit chỉ trên train của fold đầu
    fold0 = next(cv.split(df))
    train_df, _ = cv.get_train_val(df, fold0)
    good_scaler = StandardScaler().fit(train_df[[feature_col]].values)
    good_mean = good_scaler.mean_[0]

    # Hai mean phải khác → bằng chứng full-fit leak future info
    assert abs(bad_mean - good_mean) > 1e-3, (
        "Synthetic data quá đơn điệu — nâng cấp test data; nếu equal thì "
        "không phát hiện được leakage"
    )


# ============================================================
# 8. Outlier threshold (winsorize) phải fit trên train fold
# ============================================================

def test_outlier_threshold_only_from_train(synthetic_daily_df: pd.DataFrame) -> None:
    cv = WalkForwardCV(n_folds=3, initial_train_size=500, val_size=100, step_size=100)
    fold = next(cv.split(synthetic_daily_df))
    train_df, val_df = cv.get_train_val(synthetic_daily_df, fold)

    # Threshold tính trên train
    train_low = train_df["Gold_Close"].quantile(0.01)
    train_high = train_df["Gold_Close"].quantile(0.99)

    # Threshold tính trên full (BAD)
    full_low = synthetic_daily_df["Gold_Close"].quantile(0.01)
    full_high = synthetic_daily_df["Gold_Close"].quantile(0.99)

    # Phải khác (ngược dấu cũng OK; chỉ cần khác)
    diff_low = abs(train_low - full_low)
    diff_high = abs(train_high - full_high)
    assert diff_low + diff_high > 1e-6, (
        "Threshold trên train và full GIỐNG nhau → không phát hiện được leakage; "
        "lý do: synthetic data có thể quá uniform. Cần regen seed."
    )


# ============================================================
# 9. Build CV từ config phải thành công
# ============================================================

def test_build_cv_from_config() -> None:
    cv = build_cv_from_config()
    assert cv.n_folds == 5
    assert cv.scheme == "expanding"
    assert cv.initial_train_size == 1000
    assert cv.val_size == 90


# ============================================================
# 10. Fold dataclass repr không crash
# ============================================================

def test_fold_repr(synthetic_daily_df: pd.DataFrame) -> None:
    cv = WalkForwardCV(n_folds=2, initial_train_size=500, val_size=50, step_size=50)
    fold = next(cv.split(synthetic_daily_df))
    s = repr(fold)
    assert "Fold" in s and "train" in s and "val" in s


# ============================================================
# 11. Reject scheme không hợp lệ
# ============================================================

def test_reject_invalid_scheme() -> None:
    with pytest.raises(ValueError, match="scheme phải là"):
        WalkForwardCV(scheme="bogus")


# ============================================================
# 12. Reject DataFrame thiếu cột date
# ============================================================

def test_reject_missing_date_column() -> None:
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    cv = WalkForwardCV(n_folds=1, initial_train_size=2, val_size=1, step_size=1)
    with pytest.raises(KeyError):
        list(cv.split(df))
