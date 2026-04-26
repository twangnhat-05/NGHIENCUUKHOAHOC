"""NeuralForecast SOTA wrappers — N-HiTS, PatchTST, iTransformer, TimeMixer, TSMixer.

Mode-A: fit trên train target, predict h=n_val. Tương tự classical wrappers.
Free-tier: tất cả chạy được CPU; iTransformer + TFT hơi chậm (~30-60s/fold).

NeuralForecast API yêu cầu df format long: unique_id, ds, y.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

log = get_logger(__name__)


class _NFWrapper(BaseForecaster):
    """Base cho mọi neuralforecast model.

    Subclass override `_make_models(h, input_size)` trả về list of NF models.
    """
    name = "nf_base"
    use_features = False  # mode-A path (no engineered features used directly)

    def __init__(
        self,
        horizon: int = 1,
        input_size: int = 60,    # lookback window — typical 2-3x horizon
        max_steps: int = 200,
        seed: int = 42,
    ) -> None:
        self.horizon = horizon
        self.input_size = input_size
        self.max_steps = max_steps
        self.seed = seed
        self._nf = None
        self._train_df_nf: pd.DataFrame | None = None
        self._last_train_value: float | None = None

    def _make_models(self, h: int, input_size: int) -> list:
        raise NotImplementedError

    def fit(self, train_df: pd.DataFrame, target_col: str = "SJC_ban_ra", **kwargs) -> "_NFWrapper":
        from neuralforecast import NeuralForecast

        df_nf = pd.DataFrame({
            "unique_id": "sjc",
            "ds": pd.to_datetime(train_df["Date"]).reset_index(drop=True),
            "y": train_df[target_col].reset_index(drop=True).astype(float).to_numpy(),
        }).dropna()

        # NF cần freq nhất quán; data có VN holidays → dùng integer index
        n = len(df_nf)
        df_nf = pd.DataFrame({
            "unique_id": "sjc",
            "ds": np.arange(n, dtype=np.int64),
            "y": df_nf["y"].to_numpy(),
        })
        # h is the FORECAST horizon NF will produce. We need ≥ horizon and ideally ≥ n_val.
        # For multi-horizon eval we'll forecast n_val_max=200 steps.
        h_forecast = max(self.horizon, 100)
        models = self._make_models(h=h_forecast, input_size=self.input_size)

        self._nf = NeuralForecast(models=models, freq=1)
        try:
            self._nf.fit(df_nf, val_size=0)
        except Exception as e:
            log.error(f"{self.name} NF fit failed: {e}")
            self._nf = None
        self._last_train_value = float(df_nf["y"].iloc[-1])
        self._train_n = n
        return self

    def predict(self, test_df: pd.DataFrame, h: int = 1) -> np.ndarray:
        if self._nf is None:
            return np.full(len(test_df), self._last_train_value or 0.0)
        n = len(test_df)
        try:
            fcst = self._nf.predict()
            # Get the model's output column (first non-id/ds column)
            cols = [c for c in fcst.columns if c not in ("unique_id", "ds")]
            preds = fcst[cols[0]].to_numpy()
            if len(preds) >= n:
                return preds[:n]
            # Pad with last value
            return np.concatenate([preds, np.full(n - len(preds), preds[-1])])
        except Exception as e:
            log.warning(f"{self.name} predict failed: {e}; naive fallback")
            return np.full(n, self._last_train_value or 0.0)


# ============================================================
# CONCRETE MODELS
# ============================================================

class NHITSForecaster(_NFWrapper):
    name = "N-HiTS"

    def _make_models(self, h: int, input_size: int) -> list:
        from neuralforecast.models import NHITS
        return [NHITS(h=h, input_size=input_size, max_steps=self.max_steps,
                      random_seed=self.seed, scaler_type="standard",
                      enable_progress_bar=False)]


class NBEATSForecaster(_NFWrapper):
    name = "N-BEATS"

    def _make_models(self, h: int, input_size: int) -> list:
        from neuralforecast.models import NBEATS
        return [NBEATS(h=h, input_size=input_size, max_steps=self.max_steps,
                       random_seed=self.seed, scaler_type="standard",
                       enable_progress_bar=False)]


class PatchTSTForecaster(_NFWrapper):
    name = "PatchTST"

    def _make_models(self, h: int, input_size: int) -> list:
        from neuralforecast.models import PatchTST
        return [PatchTST(h=h, input_size=input_size, max_steps=self.max_steps,
                          random_seed=self.seed, scaler_type="standard",
                          patch_len=8, stride=4,
                          enable_progress_bar=False)]


class TimeMixerForecaster(_NFWrapper):
    name = "TimeMixer"

    def _make_models(self, h: int, input_size: int) -> list:
        from neuralforecast.models import TimeMixer
        return [TimeMixer(h=h, input_size=input_size, max_steps=self.max_steps,
                           random_seed=self.seed, scaler_type="standard",
                           n_series=1,
                           enable_progress_bar=False)]


class TSMixerForecaster(_NFWrapper):
    name = "TSMixer"

    def _make_models(self, h: int, input_size: int) -> list:
        from neuralforecast.models import TSMixer
        return [TSMixer(h=h, input_size=input_size, max_steps=self.max_steps,
                         random_seed=self.seed, scaler_type="standard",
                         n_series=1,
                         enable_progress_bar=False)]


class iTransformerForecaster(_NFWrapper):
    name = "iTransformer"

    def _make_models(self, h: int, input_size: int) -> list:
        from neuralforecast.models import iTransformer
        return [iTransformer(h=h, input_size=input_size, max_steps=self.max_steps,
                              random_seed=self.seed, scaler_type="standard",
                              n_series=1,
                              enable_progress_bar=False)]


class TFTForecaster(_NFWrapper):
    name = "TFT"

    def _make_models(self, h: int, input_size: int) -> list:
        from neuralforecast.models import TFT
        return [TFT(h=h, input_size=input_size, max_steps=self.max_steps,
                     random_seed=self.seed, scaler_type="standard",
                     enable_progress_bar=False)]


def build_neuralforecast_models(horizon: int = 1, fast_only: bool = False) -> list[BaseForecaster]:
    """Build list of NF models. fast_only=True bỏ TFT/iTransformer (slow trên CPU)."""
    models: list[BaseForecaster] = [
        NHITSForecaster(horizon=horizon),
        NBEATSForecaster(horizon=horizon),
        PatchTSTForecaster(horizon=horizon),
        TimeMixerForecaster(horizon=horizon),
        TSMixerForecaster(horizon=horizon),
    ]
    if not fast_only:
        models.append(iTransformerForecaster(horizon=horizon))
        models.append(TFTForecaster(horizon=horizon))
    return models
