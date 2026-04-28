"""Custom PyTorch LSTM v2 + GRU forecasters — CPU-friendly.

Khác legacy LSTM:
- Sử dụng features_v2 (108 features) thay vì chỉ giá
- Sliding window 30 days × n_features → LSTM/GRU → Linear → y_h{horizon}
- Mode-B per-row: mỗi val row predict bằng window 30 days kết thúc tại row đó
- Anti-leakage: scaler fit chỉ trên train; train_tail lưu để bridge val
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.models.ml import select_feature_columns
from src.utils.logging import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class _DLBaseWrapper(BaseForecaster):
    """Base cho LSTM/GRU. Subclass phải override `_make_model()`."""
    name = "dl_base"
    use_features = True   # mode-B path

    def __init__(
        self,
        horizon: int = 1,
        window_size: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        early_stopping_patience: int = 5,
        seed: int = 42,
    ) -> None:
        self.horizon = horizon
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        self._model = None
        self._scaler = None
        self._target_scaler = None
        self._feature_cols: list[str] = []
        self._train_tail: np.ndarray | None = None  # last window rows of train (scaled)
        self._device = None

    def _make_model(self, input_size: int):
        raise NotImplementedError

    @staticmethod
    def _build_sequences(X: np.ndarray, y: np.ndarray, window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """X shape (n, n_features), y shape (n,). Returns (X_seq, y_seq) shape (n-window-horizon+1, window, n_features), (n-window-horizon+1,)"""
        n = len(X)
        n_samples = n - window - horizon + 1
        if n_samples <= 0:
            return np.empty((0, window, X.shape[1])), np.empty((0,))
        X_seq = np.zeros((n_samples, window, X.shape[1]), dtype=np.float32)
        y_seq = np.zeros((n_samples,), dtype=np.float32)
        for i in range(n_samples):
            X_seq[i] = X[i : i + window]
            y_seq[i] = y[i + window + horizon - 1]
        return X_seq, y_seq

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "_DLBaseWrapper":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._device = torch.device("cpu")  # free tier
        self._feature_cols = select_feature_columns(train_df, target_col=target_col)
        if len(self._feature_cols) == 0:
            raise RuntimeError("No feature columns selected for DL")

        X_raw = train_df[self._feature_cols].ffill().fillna(0.0).values
        y_raw = train_df[target_col].ffill().fillna(0.0).values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)

        # Scale target separately for stability
        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

        X_seq, y_seq = self._build_sequences(X_scaled, y_scaled, self.window_size, self.horizon)
        if len(X_seq) < 50:
            log.warning(f"{self.name}: too few sequences ({len(X_seq)}) — skip")
            return self

        # Save tail of train (last window rows scaled) cho predict
        self._train_tail = X_scaled[-(self.window_size + self.horizon - 1):].copy()

        # Train/val split (last 10% for early stopping)
        n_val_es = max(20, int(0.1 * len(X_seq)))
        X_tr, X_va = X_seq[:-n_val_es], X_seq[-n_val_es:]
        y_tr, y_va = y_seq[:-n_val_es], y_seq[-n_val_es:]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
            batch_size=self.batch_size, shuffle=False,
        )

        self._model = self._make_model(input_size=len(self._feature_cols)).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_val = float("inf")
        patience = 0
        best_state = None

        for epoch in range(self.epochs):
            self._model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                pred = self._model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            # Validation
            self._model.eval()
            val_loss = 0.0
            n_va = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    pred = self._model(xb).squeeze(-1)
                    val_loss += loss_fn(pred, yb).item() * len(xb)
                    n_va += len(xb)
            val_loss /= max(n_va, 1)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    log.debug(f"{self.name} early stop at epoch {epoch+1}, best val_loss={best_val:.5f}")
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        import torch

        if self._model is None or self._scaler is None or self._train_tail is None:
            return np.zeros(len(test_df))

        # Prepare val features
        X_val_raw = test_df[self._feature_cols].ffill().fillna(0.0).values
        X_val_scaled = self._scaler.transform(X_val_raw)

        # Concatenate train tail + val to form sequences
        X_full = np.concatenate([self._train_tail, X_val_scaled], axis=0)
        n_val = len(X_val_raw)

        # For each val row i (in test_df), build window ending at row i (in val)
        # Position in X_full: tail_len + i
        tail_len = len(self._train_tail)
        predictions = np.zeros(n_val, dtype=np.float32)

        self._model.eval()
        with torch.no_grad():
            for i in range(n_val):
                # window covers X_full[tail_len + i - window_size + 1 : tail_len + i + 1]
                end = tail_len + i + 1
                start = end - self.window_size
                if start < 0:
                    # not enough data — use first available window (zero pad)
                    win = np.zeros((self.window_size, X_full.shape[1]), dtype=np.float32)
                    win[-end:] = X_full[:end]
                else:
                    win = X_full[start:end].astype(np.float32)
                xb = torch.from_numpy(win).unsqueeze(0)  # (1, window, n_features)
                pred = self._model(xb).squeeze().item()
                predictions[i] = pred

        # Inverse scale target
        predictions_unscaled = self._target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        return predictions_unscaled


# ============================================================
# CONCRETE: LSTM v2 + GRU
# ============================================================

class LSTMv2Forecaster(_DLBaseWrapper):
    name = "LSTMv2"

    def _make_model(self, input_size: int):
        from torch import nn
        class _LSTMNet(nn.Module):
            def __init__(self, in_size, hidden, layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(in_size, hidden, num_layers=layers,
                                    batch_first=True, dropout=dropout if layers > 1 else 0)
                self.head = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])
        return _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)


class GRUForecaster(_DLBaseWrapper):
    name = "GRU"

    def _make_model(self, input_size: int):
        from torch import nn
        class _GRUNet(nn.Module):
            def __init__(self, in_size, hidden, layers, dropout):
                super().__init__()
                self.gru = nn.GRU(in_size, hidden, num_layers=layers,
                                  batch_first=True, dropout=dropout if layers > 1 else 0)
                self.head = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :])
        return _GRUNet(input_size, self.hidden_size, self.num_layers, self.dropout)


def build_dl_simple_models(horizon: int = 1) -> list[BaseForecaster]:
    """LSTM v2 + GRU defaults."""
    return [
        LSTMv2Forecaster(horizon=horizon, window_size=30, hidden_size=64, num_layers=2,
                         epochs=30, batch_size=64, lr=1e-3),
        GRUForecaster(horizon=horizon, window_size=30, hidden_size=64, num_layers=2,
                      epochs=30, batch_size=64, lr=1e-3),
    ]
