"""Score news headlines bằng mDeBERTa zero-shot multilingual NLI.

Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
- Multilingual (hỗ trợ EN + VI + 100+ langs)
- ~280MB tải về first time
- CPU OK, ~0.2-0.5s/headline

Output: same parquet với thêm cols `label`, `score`, `signed_score`
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import project_root, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore")


def score_news_zero_shot(
    headlines_df: pd.DataFrame,
    model_id: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    batch_size: int = 8,
    text_col: str = "title",
) -> pd.DataFrame:
    """Score mỗi headline với 3-class zero-shot (positive/neutral/negative)."""
    from transformers import pipeline
    log.info(f"Loading {model_id} (first call download ~280MB)")
    classifier = pipeline(
        "zero-shot-classification",
        model=model_id,
        device=-1,            # CPU
        framework="pt",       # force PyTorch (Keras 3 incompat với TF backend)
    )

    candidate_labels = ["positive for gold price", "neutral", "negative for gold price"]
    df = headlines_df.copy()
    n = len(df)
    log.info(f"Scoring {n} headlines (batch_size={batch_size})")

    labels, scores, signed = [], [], []
    for i in range(0, n, batch_size):
        batch = df[text_col].iloc[i:i + batch_size].fillna("").tolist()
        try:
            outs = classifier(batch, candidate_labels=candidate_labels, multi_label=False)
        except Exception as e:
            log.warning(f"Batch {i} failed: {e}")
            outs = [{"labels": ["neutral"], "scores": [0.5]}] * len(batch)
        if isinstance(outs, dict):
            outs = [outs]
        for o in outs:
            top = o["labels"][0]
            sc = o["scores"][0]
            # Map "positive..." → "positive"
            if "positive" in top:
                short_label, signed_score = "positive", sc
            elif "negative" in top:
                short_label, signed_score = "negative", -sc
            else:
                short_label, signed_score = "neutral", 0.0
            labels.append(short_label)
            scores.append(sc)
            signed.append(signed_score)
        if (i // batch_size) % 5 == 0:
            log.info(f"  Progress: {i+len(batch)}/{n}")

    df["label"] = labels
    df["score"] = scores
    df["signed_score"] = signed
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Score news headlines with mDeBERTa zero-shot")
    parser.add_argument("--input", default="data/external/news_headlines.parquet")
    parser.add_argument("--output", default="data/external/news_headlines_scored.parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    in_path = project_root() / args.input
    if not in_path.exists():
        log.error(f"Missing {in_path}. Chạy `python -m src.features.news_fetch` trước.")
        return 1
    df = pd.read_parquet(in_path)
    log.info(f"Loaded {len(df)} headlines")

    df_scored = score_news_zero_shot(df, batch_size=args.batch_size)

    out_path = project_root() / args.output
    write_parquet(df_scored, out_path)

    # Summary
    counts = df_scored["label"].value_counts()
    log.info(f"\nLabel distribution: {counts.to_dict()}")
    log.info(f"Mean signed_score: {df_scored['signed_score'].mean():.3f}")
    log.info(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
