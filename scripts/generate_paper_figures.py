"""Generate 300 DPI publication-quality figures for IEEE paper.

Output: reports/paper/ieee_en/figures/*.png

Figures:
  fig1_leaderboard_h1.png     - top-12 MAPE bar chart h=1
  fig2_leaderboard_h5_h20.png - side-by-side h=5 vs h=20 top-10
  fig3_ablation_cumulative.png- ablation feature-family cumulative MAPE
  fig4_conformal_coverage.png - split vs ACI coverage per horizon (mean across folds)
  fig5_conformal_per_fold.png - per-fold split coverage drop in fold 4 (regime shift)
  fig6_shap_top10.png         - SHAP top-10 features LightGBM h=1
  fig7_friedman_cd.png        - Friedman ranks bar (proxy critical-difference)

Usage: python -m scripts.generate_paper_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("reports/paper/ieee_en/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

LB_SUMMARY = "reports/leaderboard/combined_v2_summary.csv"
ABLATION_SUMMARY = "reports/ablation/ablation_summary.csv"
CONFORMAL_SUMMARY = "reports/figures/conformal_full_summary.csv"
CONFORMAL_TABLE = "reports/figures/conformal_full_table.csv"
SHAP_TOP = "reports/figures/shap_lightgbm_h1_top20.csv"
FRIEDMAN = "reports/leaderboard/friedman_test.csv"


def load_lb_mape(horizon: int) -> pd.DataFrame:
    df = pd.read_csv(LB_SUMMARY)
    sub = df[(df["horizon"] == horizon) & (df["metric"] == "MAPE")].copy()
    sub = sub.sort_values("mean").reset_index(drop=True)
    return sub


def fig1_leaderboard_h1():
    df = load_lb_mape(1).head(12)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = ["#2E7D32" if m in ("Ridge", "ElasticNet") else
              "#1565C0" if m == "Chronos-Bolt-Small" else
              "#FB8C00" if "Naive" in m else
              "#757575" for m in df["model"]]
    bars = ax.barh(df["model"][::-1], df["mean"][::-1],
                   xerr=df["std"][::-1], color=colors[::-1],
                   ecolor="black", capsize=2.5, alpha=0.9)
    ax.set_xlabel("MAPE (%) — mean ± std across 5 walk-forward folds")
    ax.set_title("Top-12 Models on Vietnamese SJC ($h=1$, $n_{folds}=5$)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, df["mean"][::-1]):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_leaderboard_h1.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig2_leaderboard_h5_h20():
    df5 = load_lb_mape(5).head(10)
    df20 = load_lb_mape(20).head(10)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax, df_, h in zip(axes, [df5, df20], [5, 20]):
        colors = ["#2E7D32" if m in ("Ridge", "ElasticNet") else
                  "#1565C0" if m == "Chronos-Bolt-Small" else
                  "#FB8C00" if "Naive" in m else
                  "#757575" for m in df_["model"]]
        ax.barh(df_["model"][::-1], df_["mean"][::-1],
                xerr=df_["std"][::-1], color=colors[::-1],
                ecolor="black", capsize=2.0, alpha=0.9)
        ax.set_xlabel("MAPE (%)")
        ax.set_title(f"$h={h}$")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
    fig.suptitle("Top-10 Models — Long-Horizon Forecasts", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_leaderboard_h5_h20.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig3_ablation_cumulative():
    df = pd.read_csv(ABLATION_SUMMARY)
    subset_order = ["S1_lag_only", "S2_plus_returns", "S3_plus_technical",
                    "S4_plus_macro", "S5_plus_calendar", "S6_full"]
    labels = ["lag", "+returns", "+technical", "+macro", "+calendar", "+sentiment"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=False)
    for ax, h in zip(axes, [1, 5, 20]):
        sub = df[df["horizon"] == h].copy()
        x = np.arange(len(subset_order))
        for model, color, marker in [("Ridge", "#1565C0", "o"),
                                      ("ElasticNet", "#2E7D32", "s")]:
            mm = sub[sub["model"] == model].set_index("subset").loc[subset_order]
            ax.errorbar(x, mm["mape_mean"], yerr=mm["mape_std"],
                        marker=marker, color=color, label=model,
                        capsize=2.5, linewidth=1.5, markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"$h={h}$")
        ax.set_ylabel("MAPE (%)" if h == 1 else "")
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        if h == 1:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Feature-Family Ablation (cumulative add)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_ablation_cumulative.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig4_conformal_coverage():
    df = pd.read_csv(CONFORMAL_SUMMARY)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    horizons = [1, 5, 20]
    models = ["ElasticNet", "Ridge", "LightGBM"]
    width = 0.22
    x = np.arange(len(horizons))
    # Left panel: coverage
    ax = axes[0]
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("horizon").loc[horizons]
        ax.bar(x + i * width - width, sub["split_coverage"],
               width, label=f"{model} — split", alpha=0.55,
               color=["#90A4AE", "#1565C0", "#FB8C00"][i])
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("horizon").loc[horizons]
        ax.bar(x + i * width - width + width * 0.45, sub["aci_coverage"],
               width * 0.5, label=f"{model} — ACI",
               color=["#37474F", "#0D47A1", "#E65100"][i])
    ax.axhline(90, linestyle="--", color="red", linewidth=0.8, label="target 90%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$h={h}$" for h in horizons])
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Split vs ACI Coverage")
    ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)
    ax.set_ylim(60, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    # Right panel: width
    ax = axes[1]
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("horizon").loc[horizons]
        ax.bar(x + i * width - width, sub["split_width"],
               width, label=f"{model} — split", alpha=0.55,
               color=["#90A4AE", "#1565C0", "#FB8C00"][i])
        ax.bar(x + i * width - width + width * 0.45, sub["aci_width"],
               width * 0.5, label=f"{model} — ACI",
               color=["#37474F", "#0D47A1", "#E65100"][i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"$h={h}$" for h in horizons])
    ax.set_ylabel("Average interval width (M VND)")
    ax.set_title("Interval Width")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.suptitle("Conformal Prediction Coverage and Width (target $\\alpha=0.10$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_conformal_coverage.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig5_conformal_per_fold():
    df = pd.read_csv(CONFORMAL_TABLE)
    sub = df[(df["model"] == "ElasticNet") & (df["horizon"] == 1)].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(sub))
    width = 0.4
    ax.bar(x - width / 2, sub["split_coverage"], width,
           label="Split conformal", color="#FB8C00", alpha=0.9)
    ax.bar(x + width / 2, sub["aci_coverage"], width,
           label="ACI ($\\gamma=0.005$)", color="#1565C0", alpha=0.9)
    ax.axhline(90, linestyle="--", color="red", linewidth=0.8, label="target 90%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(f)}" for f in sub["fold_id"]])
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Per-Fold Coverage — ElasticNet, $h=1$\nFold 3 corresponds to the 2024 SJC rally regime shift")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_conformal_per_fold.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig6_shap_top10():
    df = pd.read_csv(SHAP_TOP).head(10)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    cols = df.columns.tolist()
    feature_col = cols[0]
    val_col = cols[1] if len(cols) > 1 else None
    if val_col is None or not pd.api.types.is_numeric_dtype(df[val_col]):
        return
    df_sorted = df.sort_values(val_col)
    ax.barh(df_sorted[feature_col], df_sorted[val_col],
            color="#5D4037", alpha=0.9)
    ax.set_xlabel("Mean $|\\phi|$ (SHAP)")
    ax.set_title("LightGBM SHAP — Top-10 features ($h=1$)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    for i, (feat, v) in enumerate(zip(df_sorted[feature_col], df_sorted[val_col])):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_shap_top10.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig7_friedman():
    df = pd.read_csv(FRIEDMAN)
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    horizons = df["horizon"].astype(int).tolist()
    chi2 = df["friedman_stat"].values
    pvals = df["p_value"].values
    bars = ax.bar([f"$h={h}$" for h in horizons], chi2,
                  color=["#2E7D32", "#1565C0", "#FB8C00"], alpha=0.9)
    for bar, c, p in zip(bars, chi2, pvals):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 1.5,
                f"$\\chi^2={c:.1f}$\n$p={p:.1e}$", ha="center", fontsize=7)
    ax.set_ylabel("Friedman $\\chi^2$ statistic")
    ax.axhline(35.17, linestyle="--", color="red", linewidth=0.8,
               label="critical $\\chi^2_{23,0.05}=35.17$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Friedman Non-Parametric Test (24 models, 5 folds)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(chi2) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_friedman.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig8_finetune_summary():
    """Mean MAPE across 5 folds — Ridge vs Chronos zero-shot vs Chronos fine-tuned."""
    df = pd.read_csv("reports/leaderboard/chronos_finetuned_summary.csv")
    horizons = [1, 5, 20]
    models = ["Ridge", "Chronos-Bolt-ZeroShot", "Chronos-Bolt-FineTuned"]
    colors = {"Ridge": "#2E7D32", "Chronos-Bolt-ZeroShot": "#90A4AE",
              "Chronos-Bolt-FineTuned": "#1565C0"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.27
    x = np.arange(len(horizons))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("horizon").loc[horizons]
        bars = ax.bar(x + (i - 1) * width, sub["MAPE_mean"], width,
                      yerr=sub["MAPE_std"], color=colors[model],
                      ecolor="black", capsize=3.0,
                      label=model.replace("Chronos-Bolt-", "Chronos "))
        for bar, val in zip(bars, sub["MAPE_mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                    f"{val:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$h={h}$" for h in horizons])
    ax.set_ylabel("MAPE (%) — mean$\\pm$std across 5 folds")
    ax.set_title("Fine-tuning Chronos-Bolt-Small on SJC (5-fold benchmark)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_finetune_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig9_finetune_per_fold():
    """Per-fold MAPE — Chronos fine-tuned vs zero-shot at h=1, highlighting fold 3 (2024 rally)."""
    df = pd.read_csv("reports/leaderboard/chronos_finetuned_long.csv")
    sub = df[(df["horizon"] == 1) & (df["model"].str.startswith("Chronos"))].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    folds = sorted(sub["fold"].unique())
    width = 0.36
    x = np.arange(len(folds))
    zs = sub[sub["model"] == "Chronos-Bolt-ZeroShot"].set_index("fold").loc[folds, "MAPE"]
    ft = sub[sub["model"] == "Chronos-Bolt-FineTuned"].set_index("fold").loc[folds, "MAPE"]
    ax.bar(x - width / 2, zs.values, width, color="#90A4AE", label="Zero-shot")
    ax.bar(x + width / 2, ft.values, width, color="#1565C0", label="Fine-tuned")
    for i, (zv, fv) in enumerate(zip(zs.values, ft.values)):
        ax.text(x[i] - width / 2, zv + 0.2, f"{zv:.2f}", ha="center", fontsize=7)
        ax.text(x[i] + width / 2, fv + 0.2, f"{fv:.2f}", ha="center", fontsize=7,
                color="#0D47A1", fontweight="bold")
    # Highlight fold 3 (2024 rally)
    ax.axvspan(2.55, 3.45, alpha=0.10, color="red", zorder=0)
    ax.text(3, max(zs) * 0.95, "2024 rally\nregime shift", ha="center",
            fontsize=7, color="darkred", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("MAPE (%) at $h=1$")
    ax.set_title("Per-fold MAPE — Chronos zero-shot vs fine-tuned, $h{=}1$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_finetune_per_fold.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    fig1_leaderboard_h1()
    fig2_leaderboard_h5_h20()
    fig3_ablation_cumulative()
    fig4_conformal_coverage()
    fig5_conformal_per_fold()
    fig6_shap_top10()
    fig7_friedman()
    fig8_finetune_summary()
    fig9_finetune_per_fold()
    print(f"All figures written to {OUT_DIR.resolve()}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}: {p.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
