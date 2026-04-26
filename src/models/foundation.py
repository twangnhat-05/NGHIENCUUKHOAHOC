"""Foundation models zero-shot wrappers (Chronos-Bolt, TTM, TimesFM, Lag-Llama).

Tất cả zero-shot — fit() chỉ store training context, predict() dùng pretrained weights.
Free tier: chạy trên CPU. First call download weights (~50MB Chronos-Bolt-Small,
~1.5MB TTM, ~500MB TimesFM, ~10MB Lag-Llama).

License notes:
- Chronos-Bolt: Apache-2.0
- TTM (granite-timeseries-ttm-r2): Apache-2.0
- TimesFM: Apache-2.0
- Lag-Llama: Apache-2.0
- Moirai: CC-BY-NC (small/base) hoặc Apache-2.0 (MoE)
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# ============================================================
# CHRONOS-BOLT (Amazon)
# ============================================================

class ChronosBoltForecaster(BaseForecaster):
    """Chronos-Bolt zero-shot forecaster (CPU-friendly).

    Chronos-Bolt là quantized variant nhanh hơn ~250x so với Chronos-T5 cũ.
    Sizes: tiny (9M), mini (21M), small (48M), base (205M).
    """
    name = "Chronos-Bolt-Small"
    use_features = False  # mode-A — chỉ dùng target series

    def __init__(self, horizon: int = 1, model_id: str = "amazon/chronos-bolt-small",
                 context_length: int = 256, num_samples: int = 20, seed: int = 42) -> None:
        self.horizon = horizon
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.seed = seed
        self._pipeline = None
        self._train_target: np.ndarray | None = None
        self._last_value: float | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "ChronosBoltForecaster":
        # Lazy load pipeline
        if self._pipeline is None:
            try:
                from chronos import BaseChronosPipeline
                import torch
                log.info(f"Loading {self.model_id} (first call downloads weights ~50-200MB)")
                self._pipeline = BaseChronosPipeline.from_pretrained(
                    self.model_id, device_map="cpu", torch_dtype=torch.float32,
                )
            except Exception as e:
                log.error(f"Chronos pipeline load failed: {e}")
                self._pipeline = None

        target = train_df[target_col].dropna().to_numpy().astype(np.float32)
        # Truncate to context_length (foundation models hate long noisy context)
        if len(target) > self.context_length:
            target = target[-self.context_length:]
        self._train_target = target
        self._last_value = float(target[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self._pipeline is None or self._train_target is None:
            return np.full(len(test_df), self._last_value or 0.0)

        import torch
        n = len(test_df)
        try:
            torch.manual_seed(self.seed)
            context = torch.tensor(self._train_target)  # 1D tensor
            # Chronos-Bolt 2.x API: predict_quantiles returns (quantiles, mean) tuple
            # Use predict_quantiles cho stable median forecast
            quantiles, mean_pred = self._pipeline.predict_quantiles(
                context, prediction_length=n,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            # mean_pred shape: (1, prediction_length) hoặc tương tự
            arr = mean_pred.cpu().numpy() if hasattr(mean_pred, "cpu") else np.asarray(mean_pred)
            arr = arr.flatten()
            if len(arr) < n:
                arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
            return arr[:n]
        except Exception as e:
            log.warning(f"Chronos predict failed: {e}; naive fallback")
            return np.full(n, self._last_value or 0.0)


# ============================================================
# TTM — Tiny Time Mixer (IBM)
# ============================================================

class TTMForecaster(BaseForecaster):
    """IBM Granite TTM r2 — ~1M params, CPU-friendly, native exog support.

    Pipeline: HuggingFace transformers with granite-timeseries-ttm-r2.
    """
    name = "TTM-r2"
    use_features = False  # univariate mode (target only — exog có thể thêm sau)

    def __init__(self, horizon: int = 1, model_id: str = "ibm-granite/granite-timeseries-ttm-r2",
                 context_length: int = 512, seed: int = 42) -> None:
        self.horizon = horizon
        self.model_id = model_id
        self.context_length = context_length
        self.seed = seed
        self._model = None
        self._train_target: np.ndarray | None = None
        self._last_value: float | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "TTMForecaster":
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM
                import torch
                log.info(f"Loading {self.model_id} (first call downloads ~5MB)")
                # TTM uses TimeSeriesModel — try transformers AutoModel
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, trust_remote_code=True,
                )
                self._model.eval()
            except Exception as e:
                log.warning(f"TTM load failed: {e}; will use naive fallback in predict")
                self._model = None
        target = train_df[target_col].dropna().to_numpy().astype(np.float32)
        if len(target) > self.context_length:
            target = target[-self.context_length:]
        self._train_target = target
        self._last_value = float(target[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self._model is None or self._train_target is None:
            return np.full(len(test_df), self._last_value or 0.0)

        import torch
        n = len(test_df)
        try:
            torch.manual_seed(self.seed)
            ctx = torch.tensor(self._train_target).unsqueeze(0).unsqueeze(-1)  # (1, ctx, 1)
            with torch.no_grad():
                out = self._model(past_values=ctx)
            # TTM output: prediction_outputs shape (batch, prediction_length, n_features)
            preds = out.prediction_outputs.squeeze().cpu().numpy()
            if preds.ndim > 1:
                preds = preds[:, 0]  # univariate
            preds = preds.flatten()
            # TTM has fixed prediction_length (96 by default)
            if len(preds) < n:
                preds = np.concatenate([preds, np.full(n - len(preds), preds[-1])])
            return preds[:n]
        except Exception as e:
            log.warning(f"TTM predict failed: {e}; naive fallback")
            return np.full(n, self._last_value or 0.0)


# ============================================================
# REGISTRY
# ============================================================

def build_foundation_models(horizon: int = 1, include_ttm: bool = True) -> list[BaseForecaster]:
    """Build danh sách foundation models.

    TimesFM + Lag-Llama có thể thêm sau (TimesFM cần ~500MB; Lag-Llama clone repo).
    """
    models: list[BaseForecaster] = [
        ChronosBoltForecaster(horizon=horizon, model_id="amazon/chronos-bolt-small"),
    ]
    if include_ttm:
        models.append(TTMForecaster(horizon=horizon))
    return models
