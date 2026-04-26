"""News sentiment pipeline cho VN gold market.

Strategy (free tier):
1. Scrape headlines từ CafeF/VnExpress gold tag → cache parquet (data/external/news_headlines.parquet)
2. Score bằng PhoBERT-base-v2 finetuned (nếu có labeled data)
   hoặc mDeBERTa-v3 zero-shot NLI (default — không cần labeled data)
3. Aggregate daily score: mean, std, count → exog feature daily.

Trong W2, KHỐNG bắt buộc chạy scrape. Nếu chưa có news → trả về DataFrame zeros
(stub) để pipeline xuôi chạy. User có thể chạy `python -m src.features.sentiment scrape` sau.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import project_root, write_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================
# STUB: zero sentiment for dates
# ============================================================

def stub_sentiment(dates: pd.Series) -> pd.DataFrame:
    """Trả về DataFrame zero-sentiment cùng index dates.

    Dùng khi chưa có news data — model có thể chạy mà không crash.
    Cột: sentiment_mean, sentiment_std, sentiment_count, sentiment_pos_ratio
    """
    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "sentiment_mean": 0.0,
        "sentiment_std":  0.0,
        "sentiment_count": 0,
        "sentiment_pos_ratio": 0.5,
    })
    return df


# ============================================================
# AGGREGATE: from headlines DataFrame → daily stats
# ============================================================

def aggregate_daily(headlines_df: pd.DataFrame, date_col: str = "date", score_col: str = "score") -> pd.DataFrame:
    """Aggregate score per day. headlines_df cần cột {date, score, label?}.

    Output columns: Date, sentiment_mean, sentiment_std, sentiment_count, sentiment_pos_ratio
    """
    if headlines_df.empty:
        return pd.DataFrame(columns=["Date", "sentiment_mean", "sentiment_std",
                                     "sentiment_count", "sentiment_pos_ratio"])
    df = headlines_df.copy()
    df["Date"] = pd.to_datetime(df[date_col]).dt.normalize()
    grouped = df.groupby("Date").agg(
        sentiment_mean=(score_col, "mean"),
        sentiment_std=(score_col, "std"),
        sentiment_count=(score_col, "size"),
    ).reset_index()
    grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
    if "label" in df.columns:
        pos_count = df.groupby("Date")["label"].apply(lambda s: (s == "positive").mean())
        grouped["sentiment_pos_ratio"] = grouped["Date"].map(pos_count).fillna(0.5)
    else:
        # Fallback: sigmoid(score) approximation
        grouped["sentiment_pos_ratio"] = (grouped["sentiment_mean"] > 0).astype(float)
    return grouped


# ============================================================
# SCORE WITH HF MODEL (lazy — only if user runs `score` command)
# ============================================================

def score_headlines_zero_shot(
    headlines: list[str],
    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    candidate_labels: tuple[str, ...] = ("positive", "neutral", "negative"),
) -> list[dict]:
    """Zero-shot sentiment classification cho list headlines (CPU OK).

    Output: list of {label: str, score: float}
    """
    from transformers import pipeline  # lazy
    log.info(f"Loading zero-shot pipeline: {model_name}")
    classifier = pipeline("zero-shot-classification", model=model_name)
    results = []
    for text in headlines:
        out = classifier(text, candidate_labels=list(candidate_labels))
        # Convert to signed score: positive=+1*score, neutral=0, negative=-1*score
        top = out["labels"][0]
        s = out["scores"][0]
        signed = s if top == "positive" else (-s if top == "negative" else 0.0)
        results.append({"label": top, "score": signed})
    return results


# ============================================================
# CLI: build_or_stub
# ============================================================

def build_or_stub(
    dates: pd.Series,
    headlines_path: str = "data/external/news_headlines.parquet",
) -> pd.DataFrame:
    """Trả về sentiment daily features.

    Nếu file headlines tồn tại → score + aggregate.
    Nếu KHÔNG tồn tại → stub (zeros).
    """
    p = project_root() / headlines_path
    if not p.exists():
        log.warning(f"News headlines chưa có ({headlines_path}) — dùng STUB (zeros). "
                    f"Chạy `python -m src.features.sentiment scrape` để bật sentiment thật.")
        return stub_sentiment(dates)

    df = pd.read_parquet(p)
    if "score" not in df.columns:
        log.warning(f"{p.name} thiếu cột 'score' — chưa score; dùng STUB")
        return stub_sentiment(dates)
    return aggregate_daily(df)


def merge_sentiment_into_features(
    features_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Left-join sentiment_df vào features_df, ffill các giá trị thiếu."""
    out = features_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    sent = sentiment_df.copy()
    sent["Date"] = pd.to_datetime(sent["Date"])
    out = out.merge(sent, on="Date", how="left")
    sentiment_cols = [c for c in sent.columns if c != "Date"]
    out[sentiment_cols] = out[sentiment_cols].ffill().fillna(0.0)
    # Add lags cho sentiment
    for col in ("sentiment_mean", "sentiment_count"):
        if col in out.columns:
            for lag in [1, 3, 7]:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentiment pipeline (stub-first)")
    parser.add_argument("command", choices=["stub", "score"], default="stub", nargs="?")
    parser.add_argument("--input-features", default="data/processed/features_v2.parquet")
    parser.add_argument("--output-features", default="data/processed/features_v2_with_sentiment.parquet")
    args = parser.parse_args()

    from src.utils.io import read_parquet  # lazy
    feats = read_parquet(args.input_features)
    log.info(f"Loaded features: {feats.shape}")

    if args.command == "stub":
        sent = build_or_stub(feats["Date"])
        feats_out = merge_sentiment_into_features(feats, sent)
        write_parquet(feats_out, args.output_features)
        log.info(f"Saved features+sentiment STUB: {feats_out.shape}")
    elif args.command == "score":
        log.error("Score command chưa implement scrape — cần news_headlines.parquet existing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
