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
# CHRONOS-BOLT FINE-TUNED (per-fold checkpoint, P8)
# ============================================================

class FineTunedChronosBoltForecaster(BaseForecaster):
    """Fine-tuned Chronos-Bolt loaded from a per-fold local HF directory.

    Use after `python -m scripts.finetune_chronos` has produced
    `models/chronos_finetuned/fold_{k}/` with the fine-tuned weights.

    The forecaster expects `fold_id` to be set externally before predict()
    (the trainer sets it via `model.fold_id = fold.fold_id` if available).
    Otherwise it loads `fold_0` as a fallback.
    """
    name = "Chronos-Bolt-FT"
    use_features = False

    def __init__(
        self,
        horizon: int = 1,
        checkpoint_root: str = "models/chronos_finetuned",
        context_length: int = 256,
        seed: int = 42,
    ) -> None:
        self.horizon = horizon
        self.checkpoint_root = checkpoint_root
        self.context_length = context_length
        self.seed = seed
        self.fold_id: int | None = None
        self._pipeline = None
        self._loaded_fold: int | None = None
        self._train_target: np.ndarray | None = None
        self._last_value: float | None = None

    def _load_for_fold(self, fold_id: int) -> None:
        from pathlib import Path as _P
        from chronos import BaseChronosPipeline
        import torch

        fold_dir = _P(self.checkpoint_root) / f"fold_{fold_id}"
        if not fold_dir.exists():
            log.warning(f"Fine-tuned checkpoint missing: {fold_dir}; naive fallback")
            self._pipeline = None
            return
        log.info(f"Loading fine-tuned Chronos-Bolt from {fold_dir}")
        self._pipeline = BaseChronosPipeline.from_pretrained(
            str(fold_dir), device_map="cpu", torch_dtype=torch.float32,
        )
        self._loaded_fold = fold_id

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "FineTunedChronosBoltForecaster":
        target_fold = self.fold_id if self.fold_id is not None else 0
        if self._loaded_fold != target_fold:
            self._load_for_fold(target_fold)

        target = train_df[target_col].dropna().to_numpy().astype(np.float32)
        if len(target) > self.context_length:
            target = target[-self.context_length:]
        self._train_target = target
        self._last_value = float(target[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        n = len(test_df)
        if self._pipeline is None or self._train_target is None:
            return np.full(n, self._last_value or 0.0)

        import torch
        try:
            torch.manual_seed(self.seed)
            context = torch.tensor(self._train_target)
            quantiles, mean_pred = self._pipeline.predict_quantiles(
                context, prediction_length=max(n, 1),
                quantile_levels=[0.1, 0.5, 0.9],
            )
            arr = mean_pred.cpu().numpy() if hasattr(mean_pred, "cpu") else np.asarray(mean_pred)
            arr = arr.flatten()
            if len(arr) < n:
                arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
            return arr[:n]
        except Exception as e:
            log.warning(f"Fine-tuned Chronos predict failed: {e}; naive fallback")
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
# LAG-LLAMA (Time-Llama 2024, probabilistic, ~2.4M params)
# ============================================================

class LagLlamaForecaster(BaseForecaster):
    """Lag-Llama wrapper — probabilistic foundation model.

    Setup (Linux/Mac/Colab — Windows có thể lỗi PyTorch Lightning paths):
        git clone https://github.com/time-series-foundation-models/lag-llama
        pip install -e lag-llama
        huggingface-cli download time-series-foundation-models/Lag-Llama \
            lag-llama.ckpt --local-dir ./checkpoints/

    Sau setup, instantiate với checkpoint_path trỏ tới `./checkpoints/lag-llama.ckpt`.
    """
    name = "Lag-Llama"
    use_features = False

    def __init__(self, horizon: int = 1, checkpoint_path: str = "checkpoints/lag-llama.ckpt",
                 context_length: int = 32, num_samples: int = 100) -> None:
        self.horizon = horizon
        self.checkpoint_path = checkpoint_path
        self.context_length = context_length
        self.num_samples = num_samples
        self._estimator = None
        self._train_target: np.ndarray | None = None
        self._last_value: float | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "LagLlamaForecaster":
        from pathlib import Path as _P
        if self._estimator is None:
            try:
                from lag_llama.gluon.estimator import LagLlamaEstimator
                import torch
                if not _P(self.checkpoint_path).exists():
                    log.warning(f"Lag-Llama checkpoint missing: {self.checkpoint_path}; predict naive fallback")
                else:
                    ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                    args = ckpt["hyper_parameters"]["model_kwargs"]
                    self._estimator = LagLlamaEstimator(
                        ckpt_path=self.checkpoint_path,
                        prediction_length=max(self.horizon, 32),
                        context_length=self.context_length,
                        input_size=args["input_size"], n_layer=args["n_layer"],
                        n_embd_per_head=args["n_embd_per_head"], n_head=args["n_head"],
                        scaling=args["scaling"], time_feat=args["time_feat"],
                        nonnegative_pred_samples=False, aug_prob=0.0, lr=5e-4,
                        num_samples=self.num_samples, device="cpu",
                    )
            except ImportError:
                log.warning("Lag-Llama deps missing — clone repo + pip install -e lag-llama")
                self._estimator = None
            except Exception as e:
                log.error(f"Lag-Llama load failed: {e}")
                self._estimator = None

        target = train_df[target_col].dropna().to_numpy().astype(np.float32)
        self._train_target = target
        self._last_value = float(target[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        n = len(test_df)
        if self._estimator is None or self._train_target is None:
            return np.full(n, self._last_value or 0.0)
        # Implementation cụ thể defer — cần GluonTS dataset wrap
        log.warning("Lag-Llama predict scaffolded — implement GluonTS dataset wrap riêng")
        return np.full(n, self._last_value or 0.0)


# ============================================================
# REGISTRY
# ============================================================

class TimesFMForecaster(BaseForecaster):
    """Google TimesFM 2.0 wrapper (PyTorch, ~200M params, Apache-2.0).

    Pretrained decoder-only foundation model cho TS, supports horizon up to 96 default.
    First call download ~200MB từ HF (google/timesfm-1.0-200m-pytorch hoặc 2.0-500m).
    """
    name = "TimesFM-2.0"
    use_features = False  # univariate

    def __init__(self, horizon: int = 1, repo_id: str = "google/timesfm-2.0-500m-pytorch",
                 context_length: int = 512, freq: int = 0) -> None:
        self.horizon = horizon
        self.repo_id = repo_id
        self.context_length = context_length
        self.freq = freq  # 0 = high-freq (daily/intraday); 1 = medium; 2 = low
        self._tfm = None
        self._train_target: np.ndarray | None = None
        self._last_value: float | None = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "TimesFMForecaster":
        if self._tfm is None:
            try:
                import timesfm
                log.info(f"Loading {self.repo_id} (first call download ~200-500MB)")
                self._tfm = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend="cpu",
                        per_core_batch_size=32,
                        horizon_len=max(self.horizon, 96),
                        input_patch_len=32,
                        output_patch_len=128,
                        num_layers=20,    # for 200M model
                        model_dims=1280,
                        use_positional_embedding=False,
                    ),
                    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id),
                )
            except Exception as e:
                log.error(f"TimesFM load failed: {e}; predict sẽ naive fallback")
                self._tfm = None
        target = train_df[target_col].dropna().to_numpy().astype(np.float32)
        if len(target) > self.context_length:
            target = target[-self.context_length:]
        self._train_target = target
        self._last_value = float(target[-1])
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self._tfm is None or self._train_target is None:
            return np.full(len(test_df), self._last_value or 0.0)
        n = len(test_df)
        try:
            point_forecast, _quantile = self._tfm.forecast(
                inputs=[self._train_target.tolist()],
                freq=[self.freq],
            )
            arr = np.asarray(point_forecast).flatten()
            if len(arr) < n:
                arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
            return arr[:n]
        except Exception as e:
            log.warning(f"TimesFM predict failed: {e}; naive fallback")
            return np.full(n, self._last_value or 0.0)


def build_foundation_models(
    horizon: int = 1,
    include_ttm: bool = True,
    include_timesfm: bool = False,
    include_lagllama: bool = False,
) -> list[BaseForecaster]:
    """Build danh sách foundation models.

    include_lagllama: cần clone https://github.com/time-series-foundation-models/lag-llama
    """
    models: list[BaseForecaster] = [
        ChronosBoltForecaster(horizon=horizon, model_id="amazon/chronos-bolt-small"),
    ]
    if include_ttm:
        models.append(TTMForecaster(horizon=horizon))
    if include_timesfm:
        models.append(TimesFMForecaster(horizon=horizon))
    if include_lagllama:
        models.append(LagLlamaForecaster(horizon=horizon))
    return models
