"""Class correlation and supporting figures for top important features.

Implements Stage-2-style assessment beyond feature ranking:
  * pair-level association with same-patient vs different-patient (the task "classes")
  * heatmaps of intensity / SAAP patterns across samples (ordered by patient)
  * histograms comparing pair-wise log-intensity separation by class

Typical use after ``feature_analysis.write_feature_analysis`` or a full ``run.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import feature_analysis as FA
from scipy import stats
from sklearn.metrics import roc_auc_score

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    _HAVE_MPL = True
except ImportError:  # pragma: no cover
    _HAVE_MPL = False


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_FIGURES = REPO_ROOT / "information" / "figures"

# Soft palette (aligned with plot_evaluation_figures.py)
_C_SAME = "#7EB0E0"
_C_DIFF = "#6BB89A"
_CMAP_INT = "#F5F5F4"
_HEAT_CMAP = "YlGnBu"


def _safe_log2_series(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    return np.log2(x.where(x > 0))


def _pair_indices_for_correlation(samples, sample_patient, seed=0, max_pairs=120_000):
    """All same-patient pairs plus a capped random subset of between-patient pairs."""
    ia, ib, lab = D.all_pairs(samples, sample_patient)
    pos = np.where(lab == 1)[0]
    neg = np.where(lab == 0)[0]
    rng = np.random.RandomState(seed)
    if len(pos) + len(neg) > max_pairs:
        n_pos_keep = min(len(pos), max_pairs // 3)
        n_neg_keep = min(len(neg), max_pairs - n_pos_keep)
        pos = rng.choice(pos, size=n_pos_keep, replace=False) if n_pos_keep < len(pos) else pos
        neg = rng.choice(neg, size=n_neg_keep, replace=False) if n_neg_keep < len(neg) else neg
        sel = np.concatenate([pos, neg])
        rng.shuffle(sel)
        ia, ib, lab = ia[sel], ib[sel], lab[sel]
    return ia, ib, lab


def _pair_metrics_binary(det: np.ndarray, ia, ib, lab):
    """Co-detection indicator: 1 iff both samples detect the SAAP."""
    score = ((det[ia] > 0) & (det[ib] > 0)).astype(float)
    y = lab.astype(int)
    if len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return None
    auc = roc_auc_score(y, score)
    r, p = stats.pointbiserialr(y, score)
    return {
        "n_pairs": int(len(y)),
        "pair_auroc": float(auc),
        "point_biserial_r": float(r),
        "point_biserial_p": float(p),
        "mean_score_same": float(score[y == 1].mean()),
        "mean_score_diff": float(score[y == 0].mean()),
    }


def _pair_metrics_intensity(log_vec: np.ndarray, ia, ib, lab):
    """Use negative absolute log-intensity difference as the pair similarity score."""
    la = log_vec[ia]
    lb = log_vec[ib]
    ok = np.isfinite(la) & np.isfinite(lb)
    if ok.sum() < 10:
        return None
    diff = np.abs(la[ok] - lb[ok])
    score = -diff
    y = lab[ok].astype(int)
    if len(np.unique(y)) < 2:
        return None
    auc = roc_auc_score(y, score)
    r, p = stats.pointbiserialr(y, score)
    same = lab[ok] == 1
    diff_mask = lab[ok] == 0
    return {
        "n_pairs": int(ok.sum()),
        "pair_auroc": float(auc),
        "point_biserial_r": float(r),
        "point_biserial_p": float(p),
        "mean_abs_log_diff_same": float(diff[same].mean()) if same.any() else np.nan,
        "mean_abs_log_diff_diff": float(diff[diff_mask].mean()) if diff_mask.any() else np.nan,
        "mean_score_same": float(score[same].mean()) if same.any() else np.nan,
        "mean_score_diff": float(score[diff_mask].mean()) if diff_mask.any() else np.nan,
    }


def compute_feature_class_correlation(
    ds: D.PeptidoformDataset,
    cfg,
    top_examples: pd.DataFrame,
    saap_ranked: pd.DataFrame,
    dbsnp_table=None,
    seed=0,
    max_pairs=120_000,
):
    """Correlate top RF / SAAP features with same-patient vs different-patient pairs."""
    reps = FA._build_current_representations(ds, cfg, dbsnp_table=dbsnp_table)
    samples = list(ds.samples)
    sample_patient = ds.sample_patient
    ia, ib, lab = _pair_indices_for_correlation(samples, sample_patient, seed=seed, max_pairs=max_pairs)

    log_mat = reps["log_matrix"]
    saap_det = reps["saap_detection"]

    saap_lookup = saap_ranked.set_index("peptidoform") if not saap_ranked.empty else pd.DataFrame()

    rows = []
    for _, row in top_examples.iterrows():
        pep = row["peptidoform"]
        source = row.get("source", "")
        rec = {
            "peptidoform": pep,
            "source": source,
            "display_rank": row.get("display_rank", np.nan),
            "class_definition": "same_patient_pair=1, different_patient_pair=0",
        }
        source_s = str(source)
        rec_out = {**rec}

        if "RF" in source_s and pep in log_mat.columns:
            log_vec = _safe_log2_series(log_mat[pep]).values
            m = _pair_metrics_intensity(log_vec, ia, ib, lab)
            rec_out["feature_type"] = "log_intensity"
            if m:
                rec_out.update(m)
        elif pep in saap_det.columns:
            det = saap_det[pep].values.astype(float)
            m = _pair_metrics_binary(det, ia, ib, lab)
            rec_out["feature_type"] = "saap_detection"
            if m:
                rec_out.update(m)
            if pep in saap_lookup.index:
                for col in (
                    "within_shared_rate",
                    "between_shared_rate",
                    "enrichment",
                    "weighted_score",
                    "rarity_weight",
                ):
                    rec_out[col] = saap_lookup.at[pep, col]
        elif pep in log_mat.columns:
            log_vec = _safe_log2_series(log_mat[pep]).values
            m = _pair_metrics_intensity(log_vec, ia, ib, lab)
            rec_out["feature_type"] = "log_intensity"
            if m:
                rec_out.update(m)
        else:
            continue

        rows.append(rec_out)

    out = pd.DataFrame(rows)
    if not out.empty and "pair_auroc" in out.columns:
        out = out.sort_values(["pair_auroc", "point_biserial_r"], ascending=[False, False])
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _sample_order_by_patient(ds: D.PeptidoformDataset):
    meta = pd.DataFrame({"sample": ds.samples, "patient": [ds.patient_of(s) for s in ds.samples]})
    meta = meta.sort_values(["patient", "sample"])
    return meta["sample"].tolist(), meta["patient"].tolist()


def _apply_mpl_style():
    if not _HAVE_MPL:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )


def _pick_rf_peptides_for_plots(
    rf_peptides: pd.DataFrame,
    top_examples: pd.DataFrame,
    n: int,
    min_detection_rate: float = 0.15,
):
    """RF top list with real abundance signal (skip missingness-only features)."""
    if rf_peptides is not None and not rf_peptides.empty:
        sub = rf_peptides.copy()
        if "selected_for_report" in sub.columns:
            sub = sub[sub["selected_for_report"].astype(bool)]
        if "detection_rate" in sub.columns:
            sub = sub[sub["detection_rate"].astype(float) >= min_detection_rate]
        if "abundance_importance" in sub.columns:
            sub = sub[sub["abundance_importance"].astype(float) > 0]
        sub = sub.sort_values("total_importance", ascending=False)
        if len(sub) >= n:
            return sub["peptidoform"].head(n).tolist()

    # Fallback: RF rows in combined table that were actually detected
    sub = top_examples[top_examples["source"].astype(str).str.contains("RF", na=False)].copy()
    if "detection_rate" in sub.columns:
        sub = sub[sub["detection_rate"].astype(float) >= min_detection_rate]
    sub = sub.sort_values("display_rank")
    return sub["peptidoform"].head(n).tolist()


def plot_intensity_heatmap(
    ds: D.PeptidoformDataset,
    cfg,
    peptides: list[str],
    out_path: Path,
    max_features: int = 10,
):
    """Heatmap: samples (by patient) × top log-intensity peptidoforms."""
    if not _HAVE_MPL:
        raise RuntimeError("matplotlib is required for feature assessment figures")

    reps = FA._build_current_representations(ds, cfg, dbsnp_table=None)
    log_mat = reps["log_matrix"]

    chosen = []
    for p in peptides:
        if p not in log_mat.columns:
            continue
        col = log_mat[p].astype(float)
        if (col > 0).sum() < max(10, int(0.05 * len(col))):
            continue
        chosen.append(p)
        if len(chosen) >= max_features:
            break
    if not chosen:
        return

    order, patients = _sample_order_by_patient(ds)
    mat = log_mat.loc[order, chosen].astype(float)
    mat = np.log2(mat.where(mat > 0))

    # Short column labels
    short = [p[:28] + ("…" if len(p) > 28 else "") for p in chosen]

    fig_w = max(6.0, 0.45 * len(chosen) + 2)
    fig_h = max(5.0, 0.12 * len(order) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    vals = mat.values.astype(float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if vmin >= vmax:
        vmin, vmax = finite.min(), finite.max()
    masked = np.ma.masked_invalid(vals)

    cmap = plt.get_cmap(_HEAT_CMAP).copy()
    cmap.set_bad(color="#E7E5E4")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(chosen)))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("Top RF peptidoforms (log₂ intensity)")
    ax.set_ylabel(f"Samples ordered by patient (n={len(order)})")

    # Patient boundaries
    prev = None
    for i, p in enumerate(patients):
        if prev is not None and p != prev:
            ax.axhline(i - 0.5, color="#78716C", linewidth=0.35, alpha=0.6)
        prev = p

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("log₂ intensity")
    ax.set_title("Feature intensities across patients")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_saap_heatmap(
    ds: D.PeptidoformDataset,
    cfg,
    peptides: list[str],
    out_path: Path,
    max_features: int = 10,
    dbsnp_table=None,
):
    if not _HAVE_MPL:
        raise RuntimeError("matplotlib is required for feature assessment figures")

    reps = FA._build_current_representations(ds, cfg, dbsnp_table=dbsnp_table)
    det = reps["saap_detection"]

    chosen = [p for p in peptides if p in det.columns][:max_features]
    if not chosen:
        return

    order, patients = _sample_order_by_patient(ds)
    mat = det.loc[order, chosen].astype(float).values

    short = [p[:28] + ("…" if len(p) > 28 else "") for p in chosen]
    cmap = ListedColormap(["#F5F5F4", "#7EC9BE"])

    fig_w = max(6.0, 0.45 * len(chosen) + 2)
    fig_h = max(5.0, 0.12 * len(order) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(chosen)))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("Top SAAP features (detected = 1)")
    ax.set_ylabel(f"Samples ordered by patient (n={len(order)})")

    prev = None
    for i, p in enumerate(patients):
        if prev is not None and p != prev:
            ax.axhline(i - 0.5, color="#78716C", linewidth=0.35, alpha=0.6)
        prev = p

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=[0, 1])
    cbar.ax.set_yticklabels(["absent", "detected"])
    ax.set_title("SAAP detection across patients")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_pair_logdiff_histograms(
    ds: D.PeptidoformDataset,
    cfg,
    peptides: list[str],
    out_path: Path,
    n_panels: int = 4,
    seed=0,
    max_pairs=80_000,
):
    """|log-intensity difference| for same-patient vs different-patient pairs."""
    if not _HAVE_MPL:
        raise RuntimeError("matplotlib is required for feature assessment figures")

    reps = FA._build_current_representations(ds, cfg, dbsnp_table=None)
    log_mat = reps["log_matrix"]
    samples = list(ds.samples)
    ia, ib, lab = _pair_indices_for_correlation(samples, ds.sample_patient, seed=seed, max_pairs=max_pairs)

    chosen = []
    for p in peptides:
        if p not in log_mat.columns:
            continue
        if (log_mat[p].astype(float) > 0).sum() < 30:
            continue
        chosen.append(p)
        if len(chosen) >= n_panels:
            break
    if not chosen:
        return

    n = len(chosen)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.6 * nrows), layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for ax, pep in zip(axes, chosen):
        log_vec = _safe_log2_series(log_mat[pep]).values
        la, lb = log_vec[ia], log_vec[ib]
        ok = np.isfinite(la) & np.isfinite(lb)
        diff = np.abs(la[ok] - lb[ok])
        same = lab[ok] == 1
        ax.hist(
            diff[same],
            bins=30,
            alpha=0.65,
            color=_C_SAME,
            density=True,
            label="same patient",
        )
        ax.hist(
            diff[~same],
            bins=30,
            alpha=0.55,
            color=_C_DIFF,
            density=True,
            label="different patient",
        )
        title = pep[:32] + ("…" if len(pep) > 32 else "")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("|Δ log₂ intensity|")
        ax.set_ylabel("density")
        ax.legend(fontsize=6, frameon=False)

    for ax in axes[len(chosen) :]:
        ax.set_visible(False)

    fig.suptitle("Pairwise intensity separation by class (top RF features)", fontsize=9)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_saap_codetect_histograms(
    ds: D.PeptidoformDataset,
    cfg,
    peptides: list[str],
    out_path: Path,
    n_panels: int = 4,
    seed=0,
    max_pairs=80_000,
    dbsnp_table=None,
):
    """Co-detection indicator (both detect) for same vs different patient pairs."""
    if not _HAVE_MPL:
        raise RuntimeError("matplotlib is required for feature assessment figures")

    reps = FA._build_current_representations(ds, cfg, dbsnp_table=dbsnp_table)
    det = reps["saap_detection"]
    samples = list(ds.samples)
    ia, ib, lab = _pair_indices_for_correlation(samples, ds.sample_patient, seed=seed, max_pairs=max_pairs)

    chosen = [p for p in peptides if p in det.columns][:n_panels]
    if not chosen:
        return

    ncols = 2
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.6 * nrows), layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for ax, pep in zip(axes, chosen):
        both = ((det[pep].values[ia] > 0) & (det[pep].values[ib] > 0)).astype(int)
        same = lab == 1
        # Bar: fraction co-detected
        cats = ["same\npatient", "different\npatient"]
        fracs = [both[same].mean(), both[~same].mean()]
        ax.bar(cats, fracs, color=[_C_SAME, _C_DIFF], edgecolor=_CMAP_INT, linewidth=0.6)
        ax.set_ylim(0, min(1.05, max(fracs) * 1.25 + 0.05))
        ax.set_ylabel("P(both detect)")
        title = pep[:32] + ("…" if len(pep) > 32 else "")
        ax.set_title(title, fontsize=8)

    for ax in axes[len(chosen) :]:
        ax.set_visible(False)

    fig.suptitle("SAAP co-detection rate by pair class (top SAAP features)", fontsize=9)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _pick_peptides_from_examples(top_examples: pd.DataFrame, source_filter: str, n: int):
    sub = top_examples.copy()
    if source_filter == "rf":
        sub = sub[sub["source"].astype(str).str.contains("RF", na=False)]
    elif source_filter == "saap":
        sub = sub[sub["source"].astype(str).str.contains("SAAP", na=False)]
    sub = sub.sort_values("display_rank")
    return sub["peptidoform"].head(n).tolist()


def write_feature_assessment(
    results_dir: os.PathLike,
    figures_dir: os.PathLike,
    ds: D.PeptidoformDataset,
    cfg,
    dbsnp_table=None,
    top_k: int = 25,
    heatmap_features: int = 10,
    hist_panels: int = 4,
    seed=0,
    verbose=True,
):
    """Write correlation table + supporting PDFs for the final report."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "top_peptide_examples": results_dir / "top_peptide_examples.csv",
        "saap_feature_importance": results_dir / "saap_feature_importance.csv",
        "rf_top_peptides": results_dir / "rf_top_peptides.csv",
    }
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run feature_analysis first (or full run.py)."
            )

    top_examples = pd.read_csv(paths["top_peptide_examples"])
    saap_ranked = pd.read_csv(paths["saap_feature_importance"])
    rf_peptides = pd.read_csv(paths["rf_top_peptides"]) if paths["rf_top_peptides"].exists() else pd.DataFrame()

    corr = compute_feature_class_correlation(
        ds, cfg, top_examples, saap_ranked, dbsnp_table=dbsnp_table, seed=seed
    )
    corr_path = results_dir / "feature_class_correlation.csv"
    corr.to_csv(corr_path, index=False)

    fig_paths = {}
    if _HAVE_MPL:
        _apply_mpl_style()
        rf_peps = _pick_rf_peptides_for_plots(rf_peptides, top_examples, heatmap_features)
        saap_peps = _pick_peptides_from_examples(top_examples, "saap", heatmap_features)

        p_int = figures_dir / "feature_heatmap_intensity.pdf"
        p_saap = figures_dir / "feature_heatmap_saap.pdf"
        p_hist = figures_dir / "feature_hist_logdiff_pairs.pdf"
        p_saap_h = figures_dir / "feature_hist_saap_codetect.pdf"

        if rf_peps:
            plot_intensity_heatmap(ds, cfg, rf_peps, p_int, max_features=heatmap_features)
            plot_pair_logdiff_histograms(ds, cfg, rf_peps, p_hist, n_panels=hist_panels, seed=seed)
            fig_paths["heatmap_intensity"] = p_int
            fig_paths["hist_logdiff"] = p_hist
        elif verbose:
            print(
                "[feature-assess] skipped intensity heatmap/histograms: "
                "no RF peptidoforms with sufficient detection in the log matrix"
            )
        if saap_peps:
            plot_saap_heatmap(ds, cfg, saap_peps, p_saap, max_features=heatmap_features, dbsnp_table=dbsnp_table)
            plot_saap_codetect_histograms(
                ds, cfg, saap_peps, p_saap_h, n_panels=hist_panels, seed=seed, dbsnp_table=dbsnp_table
            )
            fig_paths["heatmap_saap"] = p_saap
            fig_paths["hist_saap_codetect"] = p_saap_h

    summary = {
        "n_features_correlated": int(len(corr)),
        "top_k_from_examples": int(top_k),
        "class_definition": "same_patient_pair=1, different_patient_pair=0",
        "correlation_csv": str(corr_path),
        "figures": {k: str(v) for k, v in fig_paths.items()},
    }
    summary_path = results_dir / "feature_assessment_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print(f"[feature-assess] wrote {corr_path}")
        for k, p in fig_paths.items():
            print(f"[feature-assess]   {k}: {p}")
        if not _HAVE_MPL:
            print("[feature-assess] matplotlib not available; skipped figures")

    return {"correlation": corr, "paths": {"correlation": corr_path, "summary": summary_path, **fig_paths}}


def main(argv=None):
    import argparse
    import evaluate as E

    parser = argparse.ArgumentParser(description="Feature-class correlation and supporting figures.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-dbsnp", action="store_true")
    args = parser.parse_args(argv)

    cfg = dict(FA.DEFAULT_CFG)
    cfg["seed"] = args.seed

    path = args.data_dir / "weightloss_peptidoforms.tsv"
    ds = D.load_peptidoform_table(path, name="weightloss", verbose=True)
    dbsnp = None
    if not args.no_dbsnp:
        dbsnp_path = args.data_dir / "SAAP_frequencies_dbSNP_2021.tsv"
        if dbsnp_path.exists():
            dbsnp = E.load_dbsnp_frequencies(str(dbsnp_path), verbose=True)

    write_feature_assessment(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        ds=ds,
        cfg=cfg,
        dbsnp_table=dbsnp,
        top_k=args.top_k,
        seed=args.seed,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
