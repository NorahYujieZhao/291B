"""Generate final-report peptide/variant assessment artifacts.

This helper script takes the ranked SAAP feature table and the raw weight-loss
peptidoform intensity table, then exports the top-variant evidence needed for
the final report:

* spectrum-identification metadata and manual screenshot checklists;
* variant/unmodified missingness, intensity, and within-patient recurrence
  summaries;
* per-sample and per-patient detection tables;
* report-ready SVG and PNG figures for intensity histograms and
  patient/timepoint detection heatmaps.

The script writes all outputs under ``results/final_assessment/`` by default.
It does not perform the manual Lorikeet/MassIVE-KB spectrum inspection; it
organizes the tabular evidence and figure outputs that accompany those
screenshots.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import data as D


META_COLUMNS = {
    "rowid",
    "ccms_row_id",
    "Peptidoform",
    "Peptidoform ID",
    "Unmod peptidoform",
    "Total",
    "Total- Unmodified sequence",
    "Peptidoforms- Unmodified sequence",
    "Proteins",
    "Mass",
    "Charge",
    "Num Mods",
    "All Mods",
    "Is Decoy",
    "Lorikeet input",
    "Orig cluster FDR",
    "Pep Prefix",
    "Annotation",
    "Annotation without position",
    "Known",
    "Num mod frags",
    "PValue",
    "% Explained",
    "Rep cluster task",
    "Rep cluster user",
    "Rep cluster index",
    "Num tasks",
    "Rep spectrum filename",
    "Rep spectrum scan",
    "Outlier groups",
    "Outlier group ratio",
    "Outlier groups- unmod",
    "Outlier group ratio- unmod",
    "Unmod_Peptidoform",
}


def strip_dyn(name: str) -> str:
    return re.sub(r"^_dyn_#", "", str(name))


def is_unmod_sample_col(name: str) -> bool:
    return strip_dyn(name).endswith("_unmod")


def is_variant_sample_col(name: str) -> bool:
    if name in META_COLUMNS:
        return False
    s = strip_dyn(name)
    if s.endswith("_unmod"):
        return False
    return s.startswith("Patient_")


def paired_unmod_col(variant_col: str) -> str:
    return f"{variant_col}_unmod"


def to_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return out.where(out != 0.0, np.nan)


def log2_or_nan(values: pd.Series) -> pd.Series:
    values = to_numeric(values)
    return np.log2(values.where(values > 0))


def patient_of(sample_name: str) -> str:
    return strip_dyn(sample_name).split(".")[0]


def timepoint_of(sample_name: str):
    m = re.search(r"Timepoint_(\d+)", strip_dyn(sample_name))
    return int(m.group(1)) if m else np.nan


def try_paired_wilcoxon(x: pd.Series, y: pd.Series):
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < 3:
        return np.nan, int(len(both))
    try:
        from scipy.stats import wilcoxon

        stat = wilcoxon(both.iloc[:, 0], both.iloc[:, 1], zero_method="wilcox")
        return float(stat.pvalue), int(len(both))
    except Exception:
        return np.nan, int(len(both))


def summarize_one(row: pd.Series, rank_row: pd.Series, sample_cols: list[str]):
    variant_raw = to_numeric(row[sample_cols])
    unmod_cols = [paired_unmod_col(c) for c in sample_cols]
    existing_unmod_cols = [c for c in unmod_cols if c in row.index]
    unmod_raw = to_numeric(row[existing_unmod_cols])
    unmod_raw.index = [c[: -len("_unmod")] for c in existing_unmod_cols]
    variant_raw.index = sample_cols

    common = sorted(set(variant_raw.index) & set(unmod_raw.index))
    variant_log = np.log2(variant_raw.loc[common].where(variant_raw.loc[common] > 0))
    unmod_log = np.log2(unmod_raw.loc[common].where(unmod_raw.loc[common] > 0))
    paired_p, paired_n = try_paired_wilcoxon(variant_log, unmod_log)

    variant_detected = variant_raw.notna()
    unmod_detected = unmod_raw.notna()
    patients = pd.Series([patient_of(c) for c in sample_cols], index=sample_cols)

    patient_table = pd.DataFrame(
        {
            "sample": sample_cols,
            "patient": patients,
            "timepoint": [timepoint_of(c) for c in sample_cols],
            "variant_detected": variant_detected.astype(int),
            "variant_log2": np.log2(variant_raw.where(variant_raw > 0)),
            "unmodified_detected": unmod_detected.reindex(sample_cols).fillna(False).astype(int),
            "unmodified_log2": np.log2(unmod_raw.reindex(sample_cols).where(unmod_raw.reindex(sample_cols) > 0)),
        }
    )
    grouped = patient_table.groupby("patient", sort=True)
    by_patient = grouped.agg(
        n_samples=("variant_detected", "size"),
        n_variant_detected=("variant_detected", "sum"),
        n_unmodified_detected=("unmodified_detected", "sum"),
        median_variant_log2=("variant_log2", "median"),
        median_unmodified_log2=("unmodified_log2", "median"),
    ).reset_index()
    by_patient["variant_detection_fraction"] = (
        by_patient["n_variant_detected"] / by_patient["n_samples"]
    )
    by_patient["unmodified_detection_fraction"] = (
        by_patient["n_unmodified_detected"] / by_patient["n_samples"]
    )

    detected_patients = by_patient[by_patient["n_variant_detected"] > 0]
    recurrence_mean = detected_patients["variant_detection_fraction"].mean()
    recurrence_median = detected_patients["variant_detection_fraction"].median()

    summary = {
        "rank": int(rank_row.get("rank", np.nan)),
        "peptidoform": row.get("Peptidoform", ""),
        "unmod_peptidoform": row.get("Unmod peptidoform", row.get("Unmod_Peptidoform", "")),
        "annotation": row.get("Annotation", ""),
        "annotation_wo_pos": row.get("Annotation without position", ""),
        "proteins": row.get("Proteins", ""),
        "candidate_proteins": ";".join(D.all_uniprot_accessions(row.get("Proteins", ""))),
        "n_candidate_proteins": len(D.all_uniprot_accessions(row.get("Proteins", ""))),
        "unique_protein_mapping": len(D.all_uniprot_accessions(row.get("Proteins", ""))) == 1,
        "charge": row.get("Charge", ""),
        "pvalue": pd.to_numeric(row.get("PValue", np.nan), errors="coerce"),
        "percent_explained": pd.to_numeric(row.get("% Explained", np.nan), errors="coerce"),
        "orig_cluster_fdr": pd.to_numeric(row.get("Orig cluster FDR", np.nan), errors="coerce"),
        "rep_cluster_task": row.get("Rep cluster task", ""),
        "rep_spectrum_filename": row.get("Rep spectrum filename", ""),
        "rep_spectrum_scan": row.get("Rep spectrum scan", ""),
        "lorikeet_input_present": pd.notna(row.get("Lorikeet input", np.nan)),
        "n_samples": len(sample_cols),
        "n_variant_detected_samples": int(variant_detected.sum()),
        "variant_missing_rate": float(1.0 - variant_detected.mean()),
        "n_unmodified_detected_samples": int(unmod_detected.sum()),
        "unmodified_missing_rate": float(1.0 - unmod_detected.mean()) if len(unmod_detected) else np.nan,
        "n_variant_detected_patients": int((by_patient["n_variant_detected"] > 0).sum()),
        "n_unmodified_detected_patients": int((by_patient["n_unmodified_detected"] > 0).sum()),
        "mean_variant_recurrence_within_detected_patients": float(recurrence_mean)
        if not math.isnan(recurrence_mean)
        else np.nan,
        "median_variant_recurrence_within_detected_patients": float(recurrence_median)
        if not math.isnan(recurrence_median)
        else np.nan,
        "median_variant_log2_intensity": float(np.nanmedian(np.log2(variant_raw.where(variant_raw > 0)))),
        "median_unmodified_log2_intensity": float(np.nanmedian(np.log2(unmod_raw.where(unmod_raw > 0))))
        if len(unmod_raw)
        else np.nan,
        "paired_variant_vs_unmodified_wilcoxon_p": paired_p,
        "paired_variant_vs_unmodified_n": paired_n,
        "saap_within_shared_rate": rank_row.get("within_shared_rate", np.nan),
        "saap_between_shared_rate": rank_row.get("between_shared_rate", np.nan),
        "saap_enrichment": rank_row.get("enrichment", np.nan),
        "saap_weighted_score": rank_row.get("weighted_score", np.nan),
        "dbsnp_frequency": rank_row.get("dbsnp_frequency", np.nan),
    }

    by_patient.insert(0, "peptidoform", row.get("Peptidoform", ""))
    patient_table.insert(0, "peptidoform", row.get("Peptidoform", ""))
    return summary, by_patient, patient_table


def _linmap(value, src_min, src_max, dst_min, dst_max):
    if src_max <= src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (float(value) - src_min) * (dst_max - dst_min) / (src_max - src_min)


def write_histogram_svg(out_path: Path, title: str, variant_vals, unmod_vals):
    variant_vals = np.asarray([v for v in variant_vals if pd.notna(v)], dtype=float)
    unmod_vals = np.asarray([v for v in unmod_vals if pd.notna(v)], dtype=float)
    all_vals = np.concatenate([variant_vals, unmod_vals]) if len(unmod_vals) else variant_vals
    if len(all_vals) == 0:
        return
    lo, hi = float(np.floor(all_vals.min())), float(np.ceil(all_vals.max()))
    if hi <= lo:
        hi = lo + 1
    bins = np.linspace(lo, hi, 13)
    v_counts, _ = np.histogram(variant_vals, bins=bins)
    u_counts, _ = np.histogram(unmod_vals, bins=bins)
    max_count = max(int(v_counts.max(initial=0)), int(u_counts.max(initial=0)), 1)

    width, height = 820, 420
    left, right, top, bottom = 70, 30, 52, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = plot_w / (len(bins) - 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" font-weight="700">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<text x="{left + plot_w / 2 - 80}" y="{height - 22}" font-family="Arial" font-size="13">log2 intensity</text>',
        f'<text x="14" y="{top + plot_h / 2}" transform="rotate(-90 14 {top + plot_h / 2})" font-family="Arial" font-size="13">sample count</text>',
        f'<rect x="{left + 520}" y="20" width="16" height="10" fill="#2f6fbb" opacity="0.65"/>',
        f'<text x="{left + 542}" y="30" font-family="Arial" font-size="12">variant</text>',
        f'<rect x="{left + 610}" y="20" width="16" height="10" fill="#c75b39" opacity="0.55"/>',
        f'<text x="{left + 632}" y="30" font-family="Arial" font-size="12">unmodified</text>',
    ]
    for i in range(len(bins) - 1):
        x = left + i * bar_w
        vh = _linmap(v_counts[i], 0, max_count, 0, plot_h)
        uh = _linmap(u_counts[i], 0, max_count, 0, plot_h)
        parts.append(
            f'<rect x="{x + 2:.1f}" y="{top + plot_h - vh:.1f}" width="{bar_w / 2 - 3:.1f}" height="{vh:.1f}" fill="#2f6fbb" opacity="0.65"/>'
        )
        parts.append(
            f'<rect x="{x + bar_w / 2 + 1:.1f}" y="{top + plot_h - uh:.1f}" width="{bar_w / 2 - 3:.1f}" height="{uh:.1f}" fill="#c75b39" opacity="0.55"/>'
        )
        if i % 2 == 0:
            parts.append(
                f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-family="Arial" font-size="10">{bins[i]:.0f}</text>'
            )
    for frac in (0, 0.25, 0.5, 0.75, 1):
        y = top + plot_h - frac * plot_h
        val = int(round(frac * max_count))
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{left - 36}" y="{y + 4:.1f}" font-family="Arial" font-size="10">{val}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_detection_svg(out_path: Path, title: str, sample_df: pd.DataFrame):
    sample_df = sample_df.sort_values(["patient", "timepoint"])
    patients = sample_df["patient"].drop_duplicates().tolist()
    cell, gap = 13, 2
    left, top = 70, 64
    max_tp = int(pd.to_numeric(sample_df["timepoint"], errors="coerce").max())
    plot_w = len(patients) * (cell + gap)
    plot_h = max_tp * (cell + gap)
    width = max(1120, left + plot_w + 40)
    height = top + plot_h + 170
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="Arial" font-size="18" font-weight="700">{escape(title)}</text>',
        f'<rect x="{width - 330}" y="16" width="12" height="12" fill="#2f6fbb"/>',
        f'<text x="{width - 312}" y="27" font-family="Arial" font-size="12">variant detected</text>',
        f'<rect x="{width - 180}" y="16" width="12" height="12" fill="#e5e7eb" stroke="#bbb"/>',
        f'<text x="{width - 162}" y="27" font-family="Arial" font-size="12">not detected / missing</text>',
    ]
    for tp in range(1, max_tp + 1):
        y = top + (tp - 1) * (cell + gap)
        parts.append(f'<text x="28" y="{y + 10}" font-family="Arial" font-size="11">T{tp}</text>')
    for c, patient in enumerate(patients):
        x = left + c * (cell + gap)
        label_y = top + plot_h + 18
        parts.append(
            f'<text x="{x + 3}" y="{label_y}" transform="rotate(65 {x + 3} {label_y})" font-family="Arial" font-size="8">{escape(patient)}</text>'
        )
        rows = sample_df[sample_df["patient"] == patient]
        for _, row in rows.iterrows():
            tp = int(row["timepoint"]) if pd.notna(row["timepoint"]) else 1
            y = top + (tp - 1) * (cell + gap)
            fill = "#2f6fbb" if int(row["variant_detected"]) else "#e5e7eb"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#b8bec8" stroke-width="0.5"/>'
            )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_da_markdown(out_dir: Path, summary: pd.DataFrame):
    lines = [
        "# Peptide/Variant-Level Differential Abundance Notes",
        "",
        "These notes are generated from the raw weight-loss peptidoform intensity table. The appropriate interpretation for this identifiability project is detection-level, identity-associated signal rather than a disease/control abundance claim.",
        "",
    ]
    for _, row in summary.iterrows():
        rank = int(row["rank"])
        lines.extend(
            [
                f"## Rank {rank}: `{row['peptidoform']}`",
                "",
                f"- Variant missing rate: {row['variant_missing_rate']:.3f} ({int(row['n_variant_detected_samples'])}/{int(row['n_samples'])} samples detected).",
                f"- Unmodified missing rate: {row['unmodified_missing_rate']:.3f} ({int(row['n_unmodified_detected_samples'])}/{int(row['n_samples'])} samples detected).",
                f"- Variant detected in {int(row['n_variant_detected_patients'])} patients; among patients where it is detected, median within-patient recurrence is {row['median_variant_recurrence_within_detected_patients']:.3f}.",
                f"- Median log2 intensity: variant {row['median_variant_log2_intensity']:.2f}, unmodified {row['median_unmodified_log2_intensity']:.2f}.",
                f"- Paired Wilcoxon test comparing variant and unmodified log2 intensity among samples where both are observed: p = {row['paired_variant_vs_unmodified_wilcoxon_p']:.3e}, n = {int(row['paired_variant_vs_unmodified_n'])}.",
                f"- Same-patient shared rate vs different-patient shared rate: {row['saap_within_shared_rate']:.3f} vs {row['saap_between_shared_rate']:.3f}.",
                "",
                "Report interpretation: use this feature as an identity-associated variant detection marker. Because the variant has non-trivial missingness, avoid claiming a conventional abundance differential unless the text explicitly conditions on observed intensities.",
                "",
                f"Suggested figures: `rank{rank}_intensity_histogram.svg` and `rank{rank}_variant_detection_by_patient.svg`.",
                "",
            ]
        )
    (out_dir / "peptide_variant_differential_abundance_notes.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _find_svg_renderer():
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("chromium"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _svg_size(svg_path: Path):
    first = svg_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    w = re.search(r'width="([0-9]+)"', first)
    h = re.search(r'height="([0-9]+)"', first)
    return (int(w.group(1)) if w else 1200, int(h.group(1)) if h else 700)


def render_svgs_to_png(out_dir: Path, upload_dir: Path | None = None):
    renderer = _find_svg_renderer()
    if renderer is None:
        print("[assessment] PNG rendering skipped: no Edge/Chrome executable found")
        return []

    profile_dir = out_dir / "edge-svg-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for svg_path in sorted(out_dir.glob("rank*_*.svg")):
        png_path = svg_path.with_suffix(".png")
        width, height = _svg_size(svg_path)
        cmd = [
            renderer,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--user-data-dir={profile_dir}",
            f"--screenshot={png_path}",
            f"--window-size={width},{height}",
            svg_path.resolve().as_uri(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        written.append(png_path)
        if upload_dir is not None:
            upload_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png_path, upload_dir / png_path.name)
    return written


def write_markdown(out_dir: Path, summary: pd.DataFrame):
    lines = [
        "# Peptide/Variant Assessment Helper Output",
        "",
        "Use this file as a checklist for the manual spectrum-validation screenshots.",
        "",
        "For each top variant, capture:",
        "",
        "1. ProteoSAFe/Lorikeet annotated spectrum for the variant PSM.",
        "2. The same page showing metadata: PValue, % Explained, Orig cluster FDR, charge, scan.",
        "3. MassIVE-KB search result or related-spectrum page for the unmodified peptide.",
        "4. Mirror/shifted-cosine comparison between variant and unmodified/reference spectrum.",
        "5. A short note on whether the mass offset is localized to the reported amino acid.",
        "",
        "## Top Variants",
        "",
    ]
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"### Rank {int(row['rank'])}: `{row['peptidoform']}`",
                "",
                f"- Unmodified: `{row['unmod_peptidoform']}`",
                f"- Annotation: `{row['annotation']}`",
                f"- Proteins: `{row['proteins']}`",
                f"- Unique protein mapping: `{row['unique_protein_mapping']}`",
                f"- PValue: `{row['pvalue']}`",
                f"- % Explained: `{row['percent_explained']}`",
                f"- Orig cluster FDR: `{row['orig_cluster_fdr']}`",
                f"- Variant missing rate: `{row['variant_missing_rate']:.3f}`",
                f"- Unmodified missing rate: `{row['unmodified_missing_rate']:.3f}`",
                f"- Within/between shared rates: `{row['saap_within_shared_rate']:.3f}` / `{row['saap_between_shared_rate']:.3f}`",
                "",
                "Screenshot filenames to place in `image/peptide/`:",
                "",
                f"- `rank{int(row['rank'])}_variant_lorikeet.png`",
                f"- `rank{int(row['rank'])}_massivekb_search.png`",
                f"- `rank{int(row['rank'])}_mirror_comparison.png`",
                "",
            ]
        )
    (out_dir / "peptide_variant_assessment_checklist.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export top SAAP peptide/variant assessment tables for the final report."
    )
    parser.add_argument("--data", default=REPO_ROOT / "data" / "weightloss_peptidoforms.tsv")
    parser.add_argument("--saap", default=REPO_ROOT / "results" / "saap_feature_importance.csv")
    parser.add_argument("--out-dir", default=REPO_ROOT / "results" / "final_assessment")
    parser.add_argument("--upload-image-dir", default=None, help="optional directory to copy rendered PNG files into")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--no-png", action="store_true", help="skip SVG-to-PNG rendering")
    args = parser.parse_args()

    data_path = Path(args.data)
    saap_path = Path(args.saap)
    out_dir = Path(args.out_dir)
    upload_image_dir = Path(args.upload_image_dir) if args.upload_image_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    saap = pd.read_csv(saap_path).head(args.top_n)
    wanted = set(saap["peptidoform"].astype(str))

    header = pd.read_csv(data_path, sep="\t", nrows=0).columns.tolist()
    sample_cols = [c for c in header if is_variant_sample_col(c)]
    unmod_cols = [paired_unmod_col(c) for c in sample_cols if paired_unmod_col(c) in header]
    usecols = [c for c in META_COLUMNS if c in header] + sample_cols + unmod_cols
    raw = pd.read_csv(data_path, sep="\t", dtype=str, usecols=usecols, low_memory=False)
    raw = raw[raw["Peptidoform"].astype(str).isin(wanted)].copy()

    summaries = []
    patient_tables = []
    sample_tables = []
    for _, rank_row in saap.iterrows():
        pep = str(rank_row["peptidoform"])
        matches = raw[raw["Peptidoform"].astype(str) == pep]
        if matches.empty:
            continue
        summary, by_patient, sample_table = summarize_one(matches.iloc[0], rank_row, sample_cols)
        summaries.append(summary)
        patient_tables.append(by_patient)
        sample_tables.append(sample_table)

    summary_df = pd.DataFrame(summaries).sort_values("rank")
    by_patient_df = pd.concat(patient_tables, ignore_index=True) if patient_tables else pd.DataFrame()
    sample_df = pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame()

    summary_df.to_csv(out_dir / "top_variant_assessment_summary.csv", index=False)
    by_patient_df.to_csv(out_dir / "top_variant_by_patient_detection.csv", index=False)
    sample_df.to_csv(out_dir / "top_variant_sample_intensities.csv", index=False)
    raw.to_csv(out_dir / "top_variant_raw_rows.csv", index=False)
    write_markdown(out_dir, summary_df)
    write_da_markdown(out_dir, summary_df)

    for _, row in summary_df.iterrows():
        rank = int(row["rank"])
        pep = row["peptidoform"]
        one = sample_df[sample_df["peptidoform"] == pep].copy()
        write_histogram_svg(
            out_dir / f"rank{rank}_intensity_histogram.svg",
            f"Rank {rank} variant vs unmodified log2 intensity",
            one["variant_log2"],
            one["unmodified_log2"],
        )
        write_detection_svg(
            out_dir / f"rank{rank}_variant_detection_by_patient.svg",
            f"Rank {rank} variant detection by patient/timepoint",
            one,
        )
    png_paths = [] if args.no_png else render_svgs_to_png(out_dir, upload_image_dir)

    print(f"[assessment] wrote {out_dir / 'top_variant_assessment_summary.csv'}")
    print(f"[assessment] wrote {out_dir / 'top_variant_by_patient_detection.csv'}")
    print(f"[assessment] wrote {out_dir / 'top_variant_sample_intensities.csv'}")
    print(f"[assessment] wrote {out_dir / 'top_variant_raw_rows.csv'}")
    print(f"[assessment] wrote {out_dir / 'peptide_variant_assessment_checklist.md'}")
    print(f"[assessment] wrote {out_dir / 'peptide_variant_differential_abundance_notes.md'}")
    for path in png_paths:
        print(f"[assessment] wrote {path}")


if __name__ == "__main__":
    main()
