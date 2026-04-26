"""Attention extraction cho DL models (PyTorch Transformer/LSTM).

Đơn giản: lấy attention weights nếu model có (TFT, iTransformer, PatchTST).
Fallback: gradient-based feature attribution via Captum (Integrated Gradients).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def integrated_gradients(
    model: Any,
    input_tensor: Any,
    target: int | None = None,
    n_steps: int = 50,
) -> np.ndarray:
    """Integrated Gradients (Sundararajan et al. 2017) qua Captum.

    Returns attribution scores cùng shape với input_tensor.
    """
    try:
        from captum.attr import IntegratedGradients
        import torch
    except ImportError:
        log.error("captum chưa cài: pip install captum")
        return np.zeros_like(input_tensor.detach().cpu().numpy() if hasattr(input_tensor, 'detach') else input_tensor)

    if not hasattr(input_tensor, "requires_grad_"):
        import torch
        input_tensor = torch.from_numpy(input_tensor).float() if not isinstance(input_tensor, torch.Tensor) else input_tensor
    input_tensor = input_tensor.requires_grad_(True)

    ig = IntegratedGradients(model)
    try:
        attributions = ig.attribute(input_tensor, n_steps=n_steps, target=target)
        return attributions.detach().cpu().numpy()
    except Exception as e:
        log.error(f"IG attribution failed: {e}")
        return np.zeros_like(input_tensor.detach().cpu().numpy())


def attention_rollout(attention_weights: list[np.ndarray]) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020) — combine multi-layer attention.

    Input: list of attention matrices từ mỗi Transformer layer (shape (n, n)).
    Output: rollout matrix.
    """
    if not attention_weights:
        return np.array([])
    rollout = attention_weights[0]
    for attn in attention_weights[1:]:
        rollout = np.matmul(attn, rollout)
    return rollout


def temporal_importance_from_lstm(
    model: Any,
    sample_input: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Approximate temporal+feature importance cho LSTM bằng gradient × input.

    sample_input: shape (window, n_features)
    Returns: DataFrame (window, n_features) với importance scores.
    """
    try:
        import torch
    except ImportError:
        return pd.DataFrame()

    if isinstance(sample_input, np.ndarray):
        x = torch.from_numpy(sample_input).float().unsqueeze(0).requires_grad_(True)
    else:
        x = sample_input.requires_grad_(True)

    model.eval()
    out = model(x)
    if hasattr(out, "shape") and out.numel() > 1:
        out = out.sum()
    out.backward()
    grads = x.grad.detach().cpu().numpy()[0]
    inp = x.detach().cpu().numpy()[0]
    importance = np.abs(grads * inp)

    feature_names = feature_names or [f"f{i}" for i in range(importance.shape[1])]
    df = pd.DataFrame(importance, columns=feature_names)
    df.index.name = "timestep"
    return df
