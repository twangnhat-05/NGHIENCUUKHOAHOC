"""Base interface cho mọi model wrapper.

Mọi model phải implement:
    fit(train_df: pd.DataFrame, target_col: str, **kwargs) -> "Model"
    predict(test_df: pd.DataFrame, h: int) -> np.ndarray   # length = len(test_df)

Train df có cột Date + features + target.
Test df có cột Date + features (không cần target).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base cho tất cả forecasters."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_col: str, **kwargs) -> "BaseForecaster":
        """Fit model trên train_df. Return self."""

    @abstractmethod
    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        """Predict cho test_df. Trả về np.ndarray length = len(test_df).

        h=1 → 1-step-ahead; nếu model multi-horizon, lấy cột h.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
