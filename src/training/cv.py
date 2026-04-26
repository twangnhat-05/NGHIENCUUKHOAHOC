"""Walk-forward cross-validation cho time-series.

CRITICAL: tuyệt đối không leak future vào train fold.
Mỗi fold: scaler + outlier threshold refit trên train slice của fold đó.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd

from src.utils.io import load_yaml
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """Một fold của walk-forward CV.

    Attributes
    ----------
    fold_id : int
        Index của fold (0-based).
    train_idx : np.ndarray
        Index (positional) của train trong dataframe sorted by Date.
    val_idx : np.ndarray
        Index (positional) của validation.
    train_dates : tuple[pd.Timestamp, pd.Timestamp]
        (start, end) của train.
    val_dates : tuple[pd.Timestamp, pd.Timestamp]
        (start, end) của validation.
    """

    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    val_dates: tuple[pd.Timestamp, pd.Timestamp]

    def __repr__(self) -> str:
        return (
            f"Fold(id={self.fold_id}, "
            f"train={self.train_dates[0].date()}→{self.train_dates[1].date()} "
            f"({len(self.train_idx)} obs), "
            f"val={self.val_dates[0].date()}→{self.val_dates[1].date()} "
            f"({len(self.val_idx)} obs))"
        )


class WalkForwardCV:
    """Walk-forward CV với expanding hoặc rolling window.

    Parameters
    ----------
    n_folds : int
        Số fold.
    initial_train_size : int
        Số observations cho fold đầu tiên (train).
    val_size : int
        Số observations cho mỗi validation set.
    step_size : int
        Khoảng cách (số observations) giữa các fold start.
    scheme : str
        "expanding" — train grow theo thời gian
        "rolling"   — train slide cố định kích thước = initial_train_size
    date_column : str
        Tên cột date trong dataframe.
    """

    def __init__(
        self,
        n_folds: int = 5,
        initial_train_size: int = 1000,
        val_size: int = 90,
        step_size: int = 90,
        scheme: str = "expanding",
        date_column: str = "Date",
    ) -> None:
        if scheme not in ("expanding", "rolling"):
            raise ValueError(f"scheme phải là 'expanding' hoặc 'rolling', nhận {scheme}")
        self.n_folds = n_folds
        self.initial_train_size = initial_train_size
        self.val_size = val_size
        self.step_size = step_size
        self.scheme = scheme
        self.date_column = date_column

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        """Yield các Fold từ dataframe (đã sort theo date)."""
        if self.date_column not in df.columns:
            raise KeyError(f"DataFrame thiếu cột date '{self.date_column}'")
        df_sorted = df.sort_values(self.date_column).reset_index(drop=True)
        dates = pd.to_datetime(df_sorted[self.date_column])
        n = len(df_sorted)

        for k in range(self.n_folds):
            train_end = self.initial_train_size + k * self.step_size
            val_start = train_end
            val_end = val_start + self.val_size
            if val_end > n:
                log.warning(
                    f"Fold {k}: val_end={val_end} vượt n={n}; cắt còn {n - val_start} obs"
                )
                val_end = n
                if val_start >= n:
                    log.warning(f"Fold {k}: không còn dữ liệu cho validation. Dừng.")
                    break

            if self.scheme == "expanding":
                train_start = 0
            else:  # rolling
                train_start = max(0, train_end - self.initial_train_size)

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)
            yield Fold(
                fold_id=k,
                train_idx=train_idx,
                val_idx=val_idx,
                train_dates=(dates.iloc[train_start], dates.iloc[train_end - 1]),
                val_dates=(dates.iloc[val_start], dates.iloc[val_end - 1]),
            )

    def get_train_val(
        self, df: pd.DataFrame, fold: Fold
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Trả về (train_df, val_df) cho 1 fold."""
        df_sorted = df.sort_values(self.date_column).reset_index(drop=True)
        return df_sorted.iloc[fold.train_idx].copy(), df_sorted.iloc[fold.val_idx].copy()


def split_test_holdout(
    df: pd.DataFrame,
    test_start: str | datetime,
    test_end: str | datetime,
    date_column: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia df thành (train_val, test) theo date range.

    Test set là HOLD-OUT cuối cùng — KHÔNG dùng trong CV.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)
    train_val = df[df[date_column] < test_start_dt].sort_values(date_column).reset_index(drop=True)
    test = (
        df[(df[date_column] >= test_start_dt) & (df[date_column] <= test_end_dt)]
        .sort_values(date_column)
        .reset_index(drop=True)
    )
    log.info(
        f"split_test_holdout: train_val={len(train_val)} ({train_val[date_column].min().date()}"
        f"→{train_val[date_column].max().date()}), "
        f"test={len(test)} ({test[date_column].min().date() if len(test) else 'N/A'}"
        f"→{test[date_column].max().date() if len(test) else 'N/A'})"
    )
    return train_val, test


def build_cv_from_config(config_path: str = "configs/cv.yaml") -> WalkForwardCV:
    """Factory: build WalkForwardCV từ YAML config."""
    cfg = load_yaml(config_path)
    wf = cfg["walk_forward"]
    scheme = "expanding" if cfg["scheme"] == "walk_forward_expanding" else "rolling"
    return WalkForwardCV(
        n_folds=wf["n_folds"],
        initial_train_size=wf["initial_train_size"],
        val_size=wf["val_size"],
        step_size=wf["step_size"],
        scheme=scheme,
    )
