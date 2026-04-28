"""End-to-end no-leakage tests (Phase 12 audit fixes).

These complement tests/test_cv_no_leakage.py — that file tests the
WalkForwardCV class contract; this one tests the trained-model inference
path. We corrupt the *future* rows of the validation slice with sentinel
values and assert the prediction at val[0] is unchanged. If a fit/predict
implementation ever re-introduces .bfill() over feature columns the
prediction at val[0] will absorb the corrupted future and the assert
will fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pd.options.mode.chained_assignment = None


def _make_synthetic_features(
    n_train: int = 200,
    n_val: int = 50,
    n_features: int = 8,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build (train, val) frames with the columns ML wrappers expect.

    val[0] has a few NaN feature cells so the ffill/bfill paths see real
    NaN handling. The target column is non-NaN throughout train + val
    (so dropna(subset=[target]) keeps everything).
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val
    feature_cols = [f"f{i}" for i in range(n_features)]
    X = rng.normal(size=(n_total, n_features))
    # Inject NaN at val[0] (= row index n_train) for half of features
    X[n_train, : n_features // 2] = np.nan
    df = pd.DataFrame(X, columns=feature_cols)
    df["Date"] = pd.date_range("2024-01-01", periods=n_total, freq="B")
    # Target column: linear combination + small noise (keeps Ridge happy)
    df["SJC_ban_ra"] = X.sum(axis=1) + rng.normal(scale=0.1, size=n_total)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy(), feature_cols


def _predict_val0(model_cls, train: pd.DataFrame, val: pd.DataFrame, horizon: int = 1) -> float:
    model = model_cls(horizon=horizon)
    model.fit(train, target_col="SJC_ban_ra")
    preds = np.asarray(model.predict(val))
    return float(preds[0])


@pytest.mark.parametrize("model_name", ["Ridge", "ElasticNet", "LightGBM"])
def test_predict_val0_does_not_depend_on_future_rows(model_name):
    """val[0] prediction must be invariant under corruption of val[1:].

    If the predict path uses .bfill() over feature columns, NaN at val[0]
    will be filled from val[1:], the corrupted future will leak into the
    prediction and the assertion will fail.
    """
    from src.models.ml import (
        ElasticNetForecaster,
        LightGBMForecaster,
        RidgeForecaster,
    )
    cls = {
        "Ridge": RidgeForecaster,
        "ElasticNet": ElasticNetForecaster,
        "LightGBM": LightGBMForecaster,
    }[model_name]

    train, val_clean, feature_cols = _make_synthetic_features()

    # Add y_h{horizon} for the horizon under test (mirrors the trainer's
    # behaviour). For h=1, simple shift is enough.
    horizon = 1
    train["y_h1"] = train["SJC_ban_ra"].shift(-horizon)
    val_clean["y_h1"] = val_clean["SJC_ban_ra"].shift(-horizon)

    pred_clean = _predict_val0(cls, train, val_clean, horizon=horizon)

    # Corrupt every future row with sentinel values
    val_corrupt = val_clean.copy()
    for col in feature_cols:
        val_corrupt.loc[val_corrupt.index[1:], col] = -999.0

    pred_corrupt = _predict_val0(cls, train, val_corrupt, horizon=horizon)

    assert np.isfinite(pred_clean), "pred_clean must be finite"
    assert pred_clean == pytest.approx(pred_corrupt, rel=1e-6, abs=1e-6), (
        f"{model_name}: future-row corruption changed val[0] prediction "
        f"({pred_clean} -> {pred_corrupt}); a backward-fill is leaking the "
        f"future into the present."
    )


def test_finetuned_chronos_loads_correct_fold_via_trainer_hook():
    """Verify the trainer's run_walk_forward sets `model.fold_id` so that
    a per-fold checkpoint loader can use the right weights. We mock both
    the model and the CV split so the test stays light.
    """
    from dataclasses import dataclass
    from src.training.trainer import run_walk_forward

    @dataclass
    class _FakeFold:
        fold_id: int
        train_idx: np.ndarray
        val_idx: np.ndarray
        train_dates: tuple
        val_dates: tuple

    class _FakeCV:
        def split(self, df):
            n = len(df)
            cut = n // 2
            return [
                _FakeFold(
                    fold_id=k,
                    train_idx=np.arange(cut),
                    val_idx=np.arange(cut, n),
                    train_dates=(df["Date"].iloc[0], df["Date"].iloc[cut - 1]),
                    val_dates=(df["Date"].iloc[cut], df["Date"].iloc[-1]),
                )
                for k in range(3)
            ]

        def get_train_val(self, df, fold):
            return df.iloc[fold.train_idx].copy(), df.iloc[fold.val_idx].copy()

    class _FoldAwareModel:
        name = "fake_fold_aware"
        use_features = False
        fold_id = None
        observed_fold_ids: list[int] = []

        def __init__(self):
            self.horizon = 1

        def fit(self, train_df, target_col="SJC_ban_ra"):
            # record what fold_id the trainer gave us
            type(self).observed_fold_ids.append(self.fold_id)
            return self

        def predict(self, test_df, h=1):
            return np.full(len(test_df), 0.0)

    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=120, freq="B"),
        "SJC_ban_ra": np.arange(120, dtype=float),
    })
    model = _FoldAwareModel()
    _FoldAwareModel.observed_fold_ids = []
    _ = run_walk_forward(df=df, models=[model], cv=_FakeCV(), horizons=[1])

    assert _FoldAwareModel.observed_fold_ids == [0, 1, 2], (
        "trainer must propagate fold.fold_id into the model before fit; "
        f"got {_FoldAwareModel.observed_fold_ids}"
    )
