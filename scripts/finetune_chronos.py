"""Fine-tune Chronos-Bolt-Small on per-fold SJC training data (walk-forward CV).

Usage (CPU smoke test, ~5 min):
  python -m scripts.finetune_chronos --fold 0 --device cpu --max-steps 50 --batch-size 8

Usage (Colab T4 GPU, all folds):
  python -m scripts.finetune_chronos --device cuda --batch-size 32 --epochs 5 --learning-rate 1e-5

Output:
  models/chronos_finetuned/fold_{k}/  (HF model directory, loadable via ChronosBoltPipeline)
  reports/leaderboard/chronos_finetune_log.csv  (training loss per step per fold)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.cv import build_cv_from_config
from src.utils.io import read_parquet
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

log = get_logger(__name__)

DEFAULT_FEATURES = "data/processed/features_v2_with_sentiment.parquet"
DEFAULT_OUT_DIR = "models/chronos_finetuned"
DEFAULT_LOG_PATH = "reports/leaderboard/chronos_finetune_log.csv"


class SlidingWindowDataset(Dataset):
    """Yields (context, target) pairs from a 1-D series via sliding windows."""

    def __init__(self, series: np.ndarray, context_length: int, prediction_length: int) -> None:
        self.series = series.astype(np.float32)
        self.context_length = context_length
        self.prediction_length = prediction_length
        max_start = len(series) - context_length - prediction_length
        if max_start < 1:
            raise ValueError(
                f"series too short: len={len(series)}, "
                f"need >= context+pred = {context_length + prediction_length}",
            )
        self.starts = np.arange(0, max_start)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[i])
        ctx = self.series[s : s + self.context_length]
        tgt = self.series[s + self.context_length : s + self.context_length + self.prediction_length]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def finetune_one_fold(
    train_series: np.ndarray,
    fold_id: int,
    output_dir: Path,
    *,
    base_model: str = "amazon/chronos-bolt-small",
    context_length: int = 256,
    epochs: int = 5,
    max_steps: int | None = None,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    device: str = "cpu",
    seed: int = 42,
) -> list[dict]:
    """Fine-tune Chronos-Bolt on one fold's training series. Returns step-level log."""
    from chronos.chronos_bolt import ChronosBoltModelForForecasting

    set_global_seed(seed)
    torch.manual_seed(seed)

    log.info(f"[fold {fold_id}] loading base model {base_model}")
    model = ChronosBoltModelForForecasting.from_pretrained(base_model)
    pred_len = model.chronos_config.prediction_length
    model.train()
    model.to(device)

    dataset = SlidingWindowDataset(
        series=train_series,
        context_length=context_length,
        prediction_length=pred_len,
    )
    log.info(f"[fold {fold_id}] dataset windows: {len(dataset)} (ctx={context_length}, pred={pred_len})")

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=g, drop_last=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    log_records: list[dict] = []
    step = 0
    start_time = time.time()
    for epoch in range(epochs):
        for ctx, tgt in loader:
            ctx = ctx.to(device)
            tgt = tgt.to(device)
            optim.zero_grad(set_to_none=True)
            out = model(context=ctx, target=tgt)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            step += 1
            log_records.append({
                "fold": fold_id,
                "epoch": epoch,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "elapsed_s": round(time.time() - start_time, 2),
            })
            if step % 10 == 0:
                log.info(f"[fold {fold_id}] step {step} | loss {loss.item():.4f} | "
                         f"epoch {epoch + 1}/{epochs} | elapsed {time.time() - start_time:.1f}s")
            if max_steps is not None and step >= max_steps:
                log.info(f"[fold {fold_id}] hit max_steps={max_steps}, stopping early")
                break
        if max_steps is not None and step >= max_steps:
            break

    # Save
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.cpu()
    model.save_pretrained(fold_dir)
    log.info(f"[fold {fold_id}] saved to {fold_dir} after {step} steps "
             f"({time.time() - start_time:.1f}s)")
    return log_records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--target-col", default="SJC_ban_ra")
    p.add_argument("--cv-config", default="configs/cv.yaml")
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    p.add_argument("--base-model", default="amazon/chronos-bolt-small")
    p.add_argument("--fold", default="all", help="Fold id (0-4) or 'all'")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap total steps per fold (smoke testing on CPU)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU")
        device = "cpu"

    df = read_parquet(args.features)
    log.info(f"Loaded features: {df.shape}")
    cv = build_cv_from_config(args.cv_config)
    folds = list(cv.split(df))

    fold_ids: list[int]
    if args.fold == "all":
        fold_ids = [f.fold_id for f in folds]
    else:
        fold_ids = [int(args.fold)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for k in fold_ids:
        fold = folds[k]
        train_df, _ = cv.get_train_val(df, fold)
        train_series = train_df[args.target_col].dropna().to_numpy().astype(np.float32)
        log.info(f"[fold {k}] train range "
                 f"{fold.train_dates[0].date()} -> {fold.train_dates[1].date()} "
                 f"({len(train_series)} obs)")
        records = finetune_one_fold(
            train_series, fold_id=k, output_dir=output_dir,
            base_model=args.base_model,
            context_length=args.context_length,
            epochs=args.epochs, max_steps=args.max_steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            device=device, seed=args.seed,
        )
        all_records.extend(records)

    # Save log
    log_df = pd.DataFrame(all_records)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        prev = pd.read_csv(log_path)
        log_df = pd.concat([prev, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    log.info(f"Training log -> {log_path} ({len(log_df)} rows)")

    # Save run metadata
    meta = {
        "base_model": args.base_model,
        "fold_ids": fold_ids,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "learning_rate": args.learning_rate,
        "device": device,
        "seed": args.seed,
    }
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Done. Models saved under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
