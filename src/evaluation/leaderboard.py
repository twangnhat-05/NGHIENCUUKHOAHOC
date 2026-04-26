"""Leaderboard: aggregate per-model per-fold per-horizon metrics → CSV + plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.io import project_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Convert list of records → wide DataFrame. Mỗi record:
        {model, horizon, fold_id, metric_dict}
    Output: long DataFrame.
    """
    rows = []
    for r in results:
        base = {"model": r["model"], "horizon": r["horizon"], "fold_id": r["fold_id"]}
        for k, v in r["metrics"].items():
            rows.append({**base, "metric": k, "value": v})
    return pd.DataFrame(rows)


def summary_per_model(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per model per horizon per metric."""
    return (long_df.groupby(["horizon", "model", "metric"])["value"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
            .sort_values(["horizon", "metric", "mean"]))


def save_leaderboard(
    long_df: pd.DataFrame,
    output_dir: str | Path = "reports/leaderboard",
    name: str = "leaderboard",
) -> Path:
    """Save long DataFrame + summary CSV và barplot."""
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(out_dir / f"{name}_long.csv", index=False, encoding="utf-8-sig")
    summary = summary_per_model(long_df)
    summary.to_csv(out_dir / f"{name}_summary.csv", index=False, encoding="utf-8-sig")

    # Plot per horizon: mean MAPE by model (lower better)
    for horizon in sorted(long_df["horizon"].unique()):
        for metric in ("MAPE", "RMSE", "MAE", "DA"):
            sub = summary[(summary["horizon"] == horizon) & (summary["metric"] == metric)]
            if sub.empty:
                continue
            sub_sorted = sub.sort_values("mean", ascending=(metric != "DA"))
            fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(sub_sorted) + 1)))
            colors = ["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(sub_sorted))]
            ax.barh(sub_sorted["model"], sub_sorted["mean"],
                    xerr=sub_sorted["std"], color=colors, alpha=0.85, capsize=3)
            ax.set_xlabel(f"{metric} (mean ± std across folds)")
            arrow = "↓ lower better" if metric != "DA" else "↑ higher better"
            ax.set_title(f"Horizon h={horizon}: {metric} {arrow}")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            plot_path = out_dir / f"{name}_h{horizon}_{metric}.png"
            plt.savefig(plot_path, dpi=130, bbox_inches="tight")
            plt.close()
            log.info(f"Saved {plot_path.name}")

    log.info(f"Leaderboard saved → {out_dir}")
    return out_dir
