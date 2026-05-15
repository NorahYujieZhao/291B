#!/usr/bin/env python3
"""Plot evaluation summaries from ``results/*.csv`` for the report / Overleaf.

Writes three separate PDFs (open-set, leak-free op_recall, decoy vs observed FDR).

Examples::

    uv run python scripts/plot_evaluation_figures.py \\
        --results-dir other_branch/results \\
        --out-dir information/figures

Defaults assume you run from the repository root after ``python src/run.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Soft, bright palette (pastels, readable on white / print)
_BAR_EDGE = "#F5F5F4"
_BAR_EDGELW = 0.55
_REF_LINE = "#A8A29E"
_ERR_CAP = "#78716C"


def _apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D6D3D1",
            "axes.labelcolor": "#44403C",
            "axes.titlecolor": "#292524",
            "xtick.color": "#44403C",
            "ytick.color": "#44403C",
            "grid.color": "#E7E5E4",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
        }
    )


def _bar_color(model: str) -> str:
    m = model
    if "M1_SAAP" in m or m.startswith("M1"):
        return "#7EC9BE"  # seafoam
    if "M2_SAAP" in m or m.startswith("M2"):
        return "#9DD9CC"  # mint
    if "M3_" in m:
        return "#C4B5FD"  # soft violet
    if "M4_" in m:
        return "#93C5FD"  # sky
    if "M5_" in m:
        return "#FBC4A8"  # peach
    if "random" in m:
        return "#D6D3D1"
    if "majority" in m:
        return "#CBB8A7"
    if "cosine" in m:
        return "#F5E6A8"  # butter
    if "spearman" in m:
        return "#BFDBFE"  # ice blue
    return "#D4D4D8"


def plot_openset_true_fdr1(ident: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = ident.sort_values("n_correct_trueFDR1", ascending=True)
    n = len(df)
    fig_h = max(2.75, 0.32 * n + 1.05)
    fig, ax = plt.subplots(figsize=(4.5, fig_h), layout="constrained")
    labels = [m.replace("baseline:", "bl:") for m in df["model"]]
    y = (df["n_correct_trueFDR1"] / df["n_queries_with_true_match_in_db"]).values
    colors = [_bar_color(m) for m in df["model"]]
    ax.barh(labels, y, color=colors, edgecolor=_BAR_EDGE, linewidth=_BAR_EDGELW)
    ax.set_xlabel("Fraction correct @ true 1% FDR")
    ax.set_xlim(0, 1.05)
    ax.axvline(1.0, color=_REF_LINE, linestyle="--", linewidth=0.9, zorder=0)
    ax.set_title("Open-set identification")
    ax.grid(axis="x", alpha=0.85)
    ax.set_axisbelow(True)
    fig.savefig(out, format="pdf")
    plt.close(fig)


def plot_leakfree_op_recall(cv: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = cv[cv["op_recall_mean"].notna()].copy().sort_values("op_recall_mean", ascending=True)
    n = len(sub)
    fig_h = max(2.65, 0.32 * n + 1.05)
    fig, ax = plt.subplots(figsize=(4.5, fig_h), layout="constrained")
    labels = [m.replace("baseline:", "bl:") for m in sub["model"]]
    err = sub["op_recall_std"].fillna(0)
    colors = [_bar_color(m) for m in sub["model"]]
    ax.barh(
        labels,
        sub["op_recall_mean"],
        xerr=err,
        color=colors,
        edgecolor=_BAR_EDGE,
        linewidth=_BAR_EDGELW,
        ecolor=_ERR_CAP,
        capsize=2.0,
        error_kw={"elinewidth": 0.85, "capthick": 0.85},
    )
    ax.set_xlabel("Leak-free operating recall (mean ± std)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Train-threshold → test pairs")
    ax.grid(axis="x", alpha=0.85)
    ax.set_axisbelow(True)
    fig.savefig(out, format="pdf")
    plt.close(fig)


def plot_decoy_vs_observed_fdr(ident: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    want = ["M1_SAAP_Jaccard", "M2_SAAP_weighted", "M3_PCA_cosine", "M4_RandomForest", "M5_Contrastive"]
    d2 = ident[ident["model"].isin(want)].copy()
    d2["_order"] = d2["model"].map({m: i for i, m in enumerate(want)})
    d2 = d2.sort_values("_order")
    obs = d2["observed_FDR_decoyFDR1"].values * 100
    dec = d2["decoy_estimated_FDR_decoyFDR1"].values * 100
    x = range(len(d2))
    w = 0.36
    fig, ax = plt.subplots(figsize=(4.25, 2.85), layout="constrained")
    # Blue vs green (same softness as other eval figures; M4-sky + M1/M2-mint families)
    c_dec = "#7EB0E0"  # soft blue (decoy-estimated)
    c_obs = "#6BB89A"  # soft green (observed)
    ax.bar(
        [i - w / 2 for i in x],
        dec,
        width=w,
        label="Decoy-estimated FDR (%)",
        color=c_dec,
        edgecolor=_BAR_EDGE,
        linewidth=_BAR_EDGELW,
    )
    ax.bar(
        [i + w / 2 for i in x],
        obs,
        width=w,
        label="Observed FDR (%)",
        color=c_obs,
        edgecolor=_BAR_EDGE,
        linewidth=_BAR_EDGELW,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(["M1", "M2", "M3", "M4", "M5"])
    ax.set_ylabel("FDR (%) @ decoy-chosen 1% threshold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#E7E5E4", fontsize=7, loc="upper right")
    ax.set_title("Decoy vs observed FDR")
    ax.grid(axis="y", alpha=0.85)
    ax.set_axisbelow(True)
    fig.savefig(out, format="pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_repo_root() / "results",
        help="Directory containing identification_fdr.csv and cv_summary.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "information" / "figures",
        help="Where to write PDF figures",
    )
    args = p.parse_args()
    res = args.results_dir
    ident = pd.read_csv(res / "identification_fdr.csv")
    cv = pd.read_csv(res / "cv_summary.csv")

    out_dir = args.out_dir
    _apply_plot_style()

    outs = [
        out_dir / "eval_fig_openset_true_fdr1.pdf",
        out_dir / "eval_fig_leakfree_op_recall.pdf",
        out_dir / "eval_fig_decoy_vs_observed_fdr.pdf",
    ]
    plot_openset_true_fdr1(ident, outs[0])
    plot_leakfree_op_recall(cv, outs[1])
    plot_decoy_vs_observed_fdr(ident, outs[2])
    for path in outs:
        print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
