from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import data as D


META_COLUMNS = [
    "rowid",
    "ccms_row_id",
    "Peptidoform",
    "Peptidoform ID",
    "Unmod peptidoform",
    "Peptidoforms- Unmodified sequence",
    "Proteins",
    "Mass",
    "Charge",
    "Num Mods",
    "All Mods",
    "Is Decoy",
    "Orig cluster FDR",
    "Annotation",
    "Annotation without position",
    "Known",
    "Num mod frags",
    "PValue",
    "% Explained",
    "Rep cluster task",
    "Rep spectrum filename",
    "Rep spectrum scan",
]


def strip_dyn(name: str) -> str:
    return re.sub(r"^_dyn_#", "", str(name))


def is_variant_sample_col(name: str) -> bool:
    if name in META_COLUMNS:
        return False
    s = strip_dyn(name)
    return s.startswith("Patient_") and not s.endswith("_unmod")


def to_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return out.where(out != 0.0, np.nan)


def patient_of(sample_col: str) -> str:
    return strip_dyn(sample_col).split(".")[0]


def try_mannwhitney(x, y):
    x = np.asarray([v for v in x if pd.notna(v)], dtype=float)
    y = np.asarray([v for v in y if pd.notna(v)], dtype=float)
    if len(x) < 3 or len(y) < 3:
        return np.nan
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    m = len(valid)
    q = valid * m / np.arange(1, m + 1)
    q = q.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out.loc[q.index] = q
    return out


def load_raw(data_path: Path):
    header = pd.read_csv(data_path, sep="\t", nrows=0).columns.tolist()
    sample_cols = [c for c in header if is_variant_sample_col(c)]
    usecols = [c for c in META_COLUMNS if c in header] + sample_cols
    raw = pd.read_csv(data_path, sep="\t", dtype=str, usecols=usecols, low_memory=False)
    return raw, sample_cols


def target_carrier_patients(raw: pd.DataFrame, sample_cols: list[str], peptidoform: str):
    row = raw[raw["Peptidoform"].astype(str) == peptidoform]
    if row.empty:
        return set()
    values = to_numeric(row.iloc[0][sample_cols])
    return {patient_of(c) for c, v in values.items() if pd.notna(v)}


def protein_rows(raw: pd.DataFrame, accession: str):
    mask = raw["Proteins"].map(lambda x: accession in D.all_uniprot_accessions(x))
    return raw[mask].copy()


def summarize_protein_peptides(raw: pd.DataFrame, sample_cols: list[str], accession: str):
    rows = protein_rows(raw, accession)
    summaries = []
    for _, row in rows.iterrows():
        values = to_numeric(row[sample_cols])
        log_values = np.log2(values.where(values > 0))
        proteins = D.all_uniprot_accessions(row.get("Proteins", ""))
        detected_patients = {patient_of(c) for c, v in values.items() if pd.notna(v)}
        summaries.append(
            {
                "protein": accession,
                "peptidoform": row.get("Peptidoform", ""),
                "unmod_peptidoform": row.get("Unmod peptidoform", ""),
                "annotation": row.get("Annotation", ""),
                "annotation_wo_pos": row.get("Annotation without position", ""),
                "candidate_proteins": ";".join(proteins),
                "n_candidate_proteins": len(proteins),
                "unique_to_this_protein": len(proteins) == 1 and proteins[0] == accession,
                "pvalue": pd.to_numeric(row.get("PValue", np.nan), errors="coerce"),
                "percent_explained": pd.to_numeric(row.get("% Explained", np.nan), errors="coerce"),
                "orig_cluster_fdr": pd.to_numeric(row.get("Orig cluster FDR", np.nan), errors="coerce"),
                "charge": row.get("Charge", ""),
                "n_detected_samples": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "n_detected_patients": len(detected_patients),
                "median_log2_intensity": float(np.nanmedian(log_values)) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(summaries)


def da_against_variant_carriers(
    raw: pd.DataFrame,
    sample_cols: list[str],
    accession: str,
    target_peptidoform: str,
):
    carriers = target_carrier_patients(raw, sample_cols, target_peptidoform)
    patients = pd.Series({c: patient_of(c) for c in sample_cols})
    carrier_cols = [c for c in sample_cols if patients[c] in carriers]
    noncarrier_cols = [c for c in sample_cols if patients[c] not in carriers]

    rows = protein_rows(raw, accession)
    out = []
    for _, row in rows.iterrows():
        values = to_numeric(row[sample_cols])
        log_values = np.log2(values.where(values > 0))
        carrier_vals = log_values.reindex(carrier_cols).dropna()
        noncarrier_vals = log_values.reindex(noncarrier_cols).dropna()
        proteins = D.all_uniprot_accessions(row.get("Proteins", ""))
        delta = (
            float(np.nanmedian(carrier_vals) - np.nanmedian(noncarrier_vals))
            if len(carrier_vals) and len(noncarrier_vals)
            else np.nan
        )
        out.append(
            {
                "protein": accession,
                "target_peptidoform": target_peptidoform,
                "peptidoform": row.get("Peptidoform", ""),
                "unmod_peptidoform": row.get("Unmod peptidoform", ""),
                "annotation": row.get("Annotation", ""),
                "unique_to_this_protein": len(proteins) == 1 and proteins[0] == accession,
                "n_carrier_patients": len(carriers),
                "n_carrier_samples_detected": int(values.reindex(carrier_cols).notna().sum()),
                "n_noncarrier_samples_detected": int(values.reindex(noncarrier_cols).notna().sum()),
                "carrier_missing_rate": float(values.reindex(carrier_cols).isna().mean()) if carrier_cols else np.nan,
                "noncarrier_missing_rate": float(values.reindex(noncarrier_cols).isna().mean()) if noncarrier_cols else np.nan,
                "median_log2_carrier": float(np.nanmedian(carrier_vals)) if len(carrier_vals) else np.nan,
                "median_log2_noncarrier": float(np.nanmedian(noncarrier_vals)) if len(noncarrier_vals) else np.nan,
                "median_log2_delta_carrier_minus_noncarrier": delta,
                "mannwhitney_p": try_mannwhitney(carrier_vals, noncarrier_vals),
            }
        )
    da = pd.DataFrame(out)
    if not da.empty:
        da["mannwhitney_q_bh"] = benjamini_hochberg(da["mannwhitney_p"])
        da["significant_q05"] = da["mannwhitney_q_bh"] < 0.05
        da["same_direction_as_target"] = np.sign(da["median_log2_delta_carrier_minus_noncarrier"]) == np.sign(
            da.loc[da["peptidoform"] == target_peptidoform, "median_log2_delta_carrier_minus_noncarrier"].iloc[0]
            if (da["peptidoform"] == target_peptidoform).any()
            else np.nan
        )
    return da


def greedy_parsimony_mapping(mapping: pd.DataFrame, protein_summary: pd.DataFrame) -> pd.DataFrame:
    """Greedy set-cover approximation for peptide-to-protein parsimony.

    The universe is the inspected target peptidoforms. Candidate proteins cover the
    inspected peptides they can explain. Ties are resolved by evidence in the full
    table: first uniquely mapped peptidoform count, then total mapped peptidoform
    count, then accession for deterministic output.
    """
    support = (
        protein_summary.groupby("protein", as_index=False)
        .agg(
            n_mapped_peptidoforms=("peptidoform", "nunique"),
            n_unique_peptidoforms=("unique_to_this_protein", "sum"),
        )
    )
    support["n_unique_peptidoforms"] = support["n_unique_peptidoforms"].astype(int)
    support_by_protein = support.set_index("protein").to_dict("index")

    protein_to_peptides = (
        mapping.groupby("protein")["target_peptidoform"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )
    remaining = set(mapping["target_peptidoform"].astype(str))
    rows = []
    step = 1
    while remaining:
        candidates = []
        for protein, peptides in protein_to_peptides.items():
            newly_covered = peptides & remaining
            if not newly_covered:
                continue
            supp = support_by_protein.get(protein, {})
            candidates.append(
                (
                    len(newly_covered),
                    int(supp.get("n_unique_peptidoforms", 0)),
                    int(supp.get("n_mapped_peptidoforms", 0)),
                    protein,
                    newly_covered,
                )
            )
        if not candidates:
            raise RuntimeError(f"could not cover remaining target peptides: {sorted(remaining)}")
        candidates.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
        coverage, n_unique, n_mapped, protein, newly_covered = candidates[0]
        rows.append(
            {
                "step": step,
                "selected_protein": protein,
                "newly_covered_peptides": ";".join(sorted(newly_covered)),
                "n_newly_covered": coverage,
                "n_mapped_peptidoforms": n_mapped,
                "n_unique_peptidoforms": n_unique,
                "remaining_after_step": len(remaining - newly_covered),
            }
        )
        remaining -= newly_covered
        step += 1
    return pd.DataFrame(rows)


def write_markdown(
    out_dir: Path,
    targets: pd.DataFrame,
    protein_summary: pd.DataFrame,
    da: pd.DataFrame,
    parsimony: pd.DataFrame,
):
    lines = [
        "# Protein Identification and Protein-Level Differential Abundance Notes",
        "",
        "Protein-level assessment uses the top variant peptides as anchors. For each associated protein, the analysis lists all peptidoforms mapping to that protein, separates unique from non-unique peptide mappings, and tests whether other uniquely mapped peptides show the same carrier-vs-noncarrier abundance pattern as the target variant.",
        "",
    ]
    for _, target in targets.iterrows():
        pep = target["peptidoform"]
        proteins = [p for p in str(target.get("candidate_proteins", "")).split(";") if p]
        lines.extend([f"## Target `{pep}`", ""])
        lines.append(f"Target candidate proteins: `{'; '.join(proteins)}`.")
        if len(proteins) != 1:
            lines.append(
                "Protein identification interpretation: this peptide is not uniquely mapped, so it supports peptide/variant-level identification but is weak evidence for a single protein-level claim."
            )
            lines.append("")
        for protein in proteins:
            ps = protein_summary[protein_summary["protein"] == protein]
            ds = da[(da["protein"] == protein) & (da["target_peptidoform"] == pep)]
            unique_ps = ps[ps["unique_to_this_protein"]]
            tested_unique = ds[ds["unique_to_this_protein"] & ds["mannwhitney_q_bh"].notna()]
            sig_unique = tested_unique[tested_unique["significant_q05"]]
            same_dir_sig = sig_unique[sig_unique["same_direction_as_target"]]
            majority = len(same_dir_sig) > 0.5 * len(tested_unique) if len(tested_unique) else False
            lines.extend(
                [
                    f"### Protein `{protein}`",
                    "",
                    f"- Peptidoforms mapping to protein: {len(ps)}.",
                    f"- Uniquely mapped peptidoforms: {len(unique_ps)}.",
                    f"- Uniquely mapped peptides tested for carrier-vs-noncarrier abundance: {len(tested_unique)}.",
                    f"- Significant unique peptides at BH q<0.05 in the target direction: {len(same_dir_sig)}.",
                    f"- Majority-rule protein-level DA call: {'protein-level effect supported' if majority else 'not supported; interpret as peptide/variant-level or mapping-limited evidence'}.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Greedy parsimony mapping",
            "",
            "The inspected-peptide universe was covered by the following greedy set-cover approximation. At each step, the selected protein covered the largest number of currently uncovered inspected peptides; ties were resolved by the number of uniquely mapped peptidoforms and then total mapped peptidoforms in the full result table.",
            "",
        ]
    )
    for _, row in parsimony.iterrows():
        lines.append(
            f"- Step {int(row['step'])}: selected `{row['selected_protein']}` covering `{row['newly_covered_peptides']}` "
            f"({int(row['n_unique_peptidoforms'])} unique peptidoforms, {int(row['n_mapped_peptidoforms'])} mapped peptidoforms)."
        )
    (out_dir / "protein_assessment_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Protein identification and protein-level DA helper.")
    parser.add_argument("--data", default=REPO_ROOT / "data" / "weightloss_peptidoforms.tsv")
    parser.add_argument(
        "--targets",
        default=REPO_ROOT / "results" / "final_assessment" / "top_variant_assessment_summary.csv",
    )
    parser.add_argument("--out-dir", default=REPO_ROOT / "results" / "final_assessment")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, sample_cols = load_raw(Path(args.data))
    targets = pd.read_csv(args.targets)

    protein_parts = []
    da_parts = []
    rows = []
    for _, target in targets.iterrows():
        pep = target["peptidoform"]
        proteins = [p for p in str(target.get("candidate_proteins", "")).split(";") if p]
        for protein in proteins:
            rows.append(
                {
                    "target_rank": target["rank"],
                    "target_peptidoform": pep,
                    "target_annotation": target["annotation"],
                    "protein": protein,
                    "target_unique_mapping": len(proteins) == 1,
                }
            )
            protein_parts.append(summarize_protein_peptides(raw, sample_cols, protein))
            da_parts.append(da_against_variant_carriers(raw, sample_cols, protein, pep))

    mapping = pd.DataFrame(rows)
    protein_summary = pd.concat(protein_parts, ignore_index=True) if protein_parts else pd.DataFrame()
    da = pd.concat(da_parts, ignore_index=True) if da_parts else pd.DataFrame()
    parsimony = greedy_parsimony_mapping(mapping, protein_summary)

    mapping.to_csv(out_dir / "target_peptide_protein_mapping.csv", index=False)
    protein_summary.to_csv(out_dir / "protein_peptide_support.csv", index=False)
    da.to_csv(out_dir / "protein_level_da_majority_rule.csv", index=False)
    parsimony.to_csv(out_dir / "greedy_parsimony_mapping.csv", index=False)
    write_markdown(out_dir, targets, protein_summary, da, parsimony)

    print(f"[protein] wrote {out_dir / 'target_peptide_protein_mapping.csv'}")
    print(f"[protein] wrote {out_dir / 'protein_peptide_support.csv'}")
    print(f"[protein] wrote {out_dir / 'protein_level_da_majority_rule.csv'}")
    print(f"[protein] wrote {out_dir / 'greedy_parsimony_mapping.csv'}")
    print(f"[protein] wrote {out_dir / 'protein_assessment_notes.md'}")


if __name__ == "__main__":
    main()
