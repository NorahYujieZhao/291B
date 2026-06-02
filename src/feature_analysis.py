from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import evaluate as E
import models as M


DEFAULT_CFG = {
    "seed": 0,
    "n_splits": 5,
    "recall_ks": [1, 5, 10],
    "saap_pvalue_min": 1.0,
    "saap_min_detection_rate": 0.1,
    "weighted_eps": 1e-4,
    "max_missing": 0.5,
    "min_variance_quantile": 0.1,
    "rf_top_k_features": 1000,
    "impute": "min",
    "pca_components": 50,
    "neg_per_pos": 5,
    "rf_trees": 300,
    "rf_max_depth": None,
    "rf_min_leaf": 2,
    "emb_dim": 64,
    "emb_hidden": 128,
    "emb_dropout": 0.3,
    "emb_lr": 1e-3,
    "emb_weight_decay": 1e-4,
    "emb_margin": 1.0,
    "emb_epochs": 80,
    "op_fdr": 0.01,
    "fdr_levels": [0.01, 0.05],
    "max_features": None,
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def _n_choose_2(x):
    x = np.asarray(x, dtype=np.int64)
    return x * (x - 1) // 2


def _base_feature_name(name: str) -> str:
    name = str(name)
    return name[:-10] if name.endswith("__missing") else name


def _safe_log2_frame(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.astype(float).values
    arr = np.where(arr > 0, arr, np.nan)
    out = np.log2(arr)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _feature_metadata(ds: D.PeptidoformDataset, feature_names: Iterable[str]) -> pd.DataFrame:
    feature_names = list(dict.fromkeys(feature_names))
    meta = ds.feature_meta.reindex(feature_names).copy()
    meta = meta.drop(columns=["peptidoform"], errors="ignore")
    meta.index.name = "peptidoform"
    meta["primary_protein"] = meta["proteins"].map(D.first_uniprot_accession)
    meta["candidate_proteins"] = meta["proteins"].map(
        lambda x: ";".join(D.all_uniprot_accessions(x)) if x is not None else ""
    )
    meta["n_candidate_proteins"] = meta["proteins"].map(
        lambda x: len(D.all_uniprot_accessions(x))
    )
    meta["unique_protein_mapping"] = meta["n_candidate_proteins"] == 1
    return meta


def _feature_stats(ds: D.PeptidoformDataset, feature_names: Iterable[str]) -> pd.DataFrame:
    feature_names = list(dict.fromkeys(feature_names))
    X = ds.intensity.reindex(columns=feature_names)
    det = X.notna()
    logX = _safe_log2_frame(X)

    patient_keys = pd.Index([ds.patient_of(s) for s in X.index], name="patient")
    patient_detected = det.groupby(patient_keys).any()

    stats = pd.DataFrame(index=feature_names)
    stats["n_detected_samples"] = det.sum(axis=0).astype(int)
    stats["detection_rate"] = det.mean(axis=0).astype(float)
    stats["missing_rate"] = X.isna().mean(axis=0).astype(float)
    stats["n_patients_detected"] = patient_detected.sum(axis=0).reindex(feature_names).astype(int)
    stats["patient_detection_rate"] = (
        patient_detected.mean(axis=0).reindex(feature_names).astype(float)
    )
    stats["median_log_intensity"] = logX.median(axis=0, skipna=True)
    stats["mean_log_intensity"] = logX.mean(axis=0, skipna=True)
    return stats


def _saap_feature_frequencies(ds: D.PeptidoformDataset, saap_features, dbsnp_table):
    out = np.full(len(saap_features), np.nan, dtype=float)
    fm = ds.feature_meta
    for i, pep in enumerate(saap_features):
        prob, matched = E._saap_match_probability(
            fm.at[pep, "annotation_wo_pos"],
            fm.at[pep, "proteins"],
            dbsnp_table,
        )
        if matched:
            out[i] = prob
    return out


def _build_current_representations(ds: D.PeptidoformDataset, cfg, dbsnp_table=None):
    train_samples = list(ds.samples)

    det_full, _ = D.build_saap_detection(
        ds,
        min_detection_rate=0.0,
        max_pvalue_neglog10=cfg["saap_pvalue_min"],
    )
    train_rate = det_full.loc[train_samples].mean(axis=0)
    keep = train_rate.index[(train_rate >= cfg["saap_min_detection_rate"]).values]
    if len(keep) == 0:
        keep = train_rate.sort_values(ascending=False).index[: max(20, int(0.05 * len(train_rate)))]

    saap_features = list(keep)
    saap_detection = det_full[saap_features]
    saap_pvalue = ds.feature_meta.reindex(saap_features)["pvalue"].values
    saap_freq = None
    if dbsnp_table is not None:
        saap_freq = _saap_feature_frequencies(ds, saap_features, dbsnp_table)

    pre = D.IntensityPreprocessor(
        max_missing=cfg["max_missing"],
        min_variance_quantile=cfg["min_variance_quantile"],
        top_k_by_variance=cfg["rf_top_k_features"],
        impute=cfg["impute"],
        log_transform=True,
        l2_normalise=False,
        include_missing_indicators=False,
    ).fit(ds.intensity.loc[train_samples])

    log_features = pre.output_feature_names()
    log_matrix = pd.DataFrame(
        pre.transform(ds.intensity),
        index=ds.samples,
        columns=log_features,
    )

    pre_missing = D.IntensityPreprocessor(
        max_missing=cfg["max_missing"],
        min_variance_quantile=cfg["min_variance_quantile"],
        top_k_by_variance=cfg["rf_top_k_features"],
        impute=cfg["impute"],
        log_transform=True,
        l2_normalise=False,
        include_missing_indicators=True,
    ).fit(ds.intensity.loc[train_samples])

    log_missing_features = pre_missing.output_feature_names()
    log_missing_matrix = pd.DataFrame(
        pre_missing.transform(ds.intensity),
        index=ds.samples,
        columns=log_missing_features,
    )

    return {
        "saap_features": saap_features,
        "saap_detection": saap_detection,
        "saap_pvalue": saap_pvalue,
        "saap_freq": saap_freq,
        "pre": pre,
        "pre_missing": pre_missing,
        "log_features": log_features,
        "log_matrix": log_matrix,
        "log_missing_features": log_missing_features,
        "log_missing_matrix": log_missing_matrix,
    }


def analyze_random_forest_features(ds: D.PeptidoformDataset, cfg, top_k=25):
    reps = _build_current_representations(ds, cfg, dbsnp_table=None)

    model = M.RandomForestPairs(
        n_estimators=cfg["rf_trees"],
        max_depth=cfg["rf_max_depth"],
        min_samples_leaf=cfg["rf_min_leaf"],
        neg_per_pos=cfg["neg_per_pos"],
        seed=cfg["seed"],
    )
    X = reps["log_missing_matrix"].values
    patient_ids = [ds.patient_of(s) for s in ds.samples]
    model.fit(X, patient_ids)

    raw = pd.DataFrame(
        {
            "feature_name": reps["log_missing_features"],
            "importance": np.asarray(model.feature_importances_, dtype=float),
        }
    )
    raw["feature_type"] = np.where(
        raw["feature_name"].str.endswith("__missing"),
        "missingness",
        "abundance",
    )
    raw["base_peptidoform"] = raw["feature_name"].map(_base_feature_name)

    base_features = list(dict.fromkeys(raw["base_peptidoform"]))
    meta = _feature_metadata(ds, base_features)
    stats = _feature_stats(ds, base_features)

    raw = raw.join(meta, on="base_peptidoform")
    raw = raw.join(stats, on="base_peptidoform")
    raw = raw.sort_values(["importance", "feature_type"], ascending=[False, True]).reset_index(drop=True)
    raw.insert(0, "rank", np.arange(1, len(raw) + 1))

    pep = (
        raw.groupby(["base_peptidoform", "feature_type"], as_index=False)["importance"]
        .sum()
        .pivot(index="base_peptidoform", columns="feature_type", values="importance")
        .fillna(0.0)
    )
    for col in ("abundance", "missingness"):
        if col not in pep.columns:
            pep[col] = 0.0
    pep = pep.rename(
        columns={
            "abundance": "abundance_importance",
            "missingness": "missingness_importance",
        }
    )
    pep["total_importance"] = pep["abundance_importance"] + pep["missingness_importance"]
    pep.index.name = "peptidoform"

    pep = pep.join(meta.rename_axis("peptidoform"))
    pep = pep.join(stats.rename_axis("peptidoform"))
    pep = pep.drop(columns=["peptidoform"], errors="ignore")
    pep = pep.sort_values(
        ["total_importance", "abundance_importance", "missingness_importance"],
        ascending=[False, False, False],
    ).reset_index()
    pep.insert(0, "rank", np.arange(1, len(pep) + 1))
    pep["selected_for_report"] = pep["rank"] <= int(top_k)

    return raw, pep


def _saap_shared_rates(saap_detection: pd.DataFrame, sample_patient: dict):
    det = saap_detection.astype(bool)
    patient_order = [sample_patient[s] for s in det.index]

    n_total = det.shape[0]
    total_pair_denom = int(_n_choose_2(n_total))
    total_detected = det.sum(axis=0).values.astype(np.int64)
    total_shared = _n_choose_2(total_detected)

    within_shared = np.zeros(det.shape[1], dtype=np.int64)
    within_pair_denom = 0

    patient_to_rows = {}
    for row_idx, patient in enumerate(patient_order):
        patient_to_rows.setdefault(patient, []).append(row_idx)

    X = det.values.astype(np.int64)
    for rows in patient_to_rows.values():
        n_p = len(rows)
        within_pair_denom += int(_n_choose_2(n_p))
        if n_p < 2:
            continue
        k_p = X[rows, :].sum(axis=0).astype(np.int64)
        within_shared += _n_choose_2(k_p)

    between_pair_denom = total_pair_denom - within_pair_denom
    between_shared = total_shared - within_shared

    within_rate = (
        within_shared.astype(float) / within_pair_denom if within_pair_denom > 0 else np.full(det.shape[1], np.nan)
    )
    between_rate = (
        between_shared.astype(float) / between_pair_denom if between_pair_denom > 0 else np.full(det.shape[1], np.nan)
    )

    return within_rate, between_rate, within_pair_denom, between_pair_denom


def analyze_saap_features(ds: D.PeptidoformDataset, cfg, dbsnp_table=None, top_k=25):
    reps = _build_current_representations(ds, cfg, dbsnp_table=dbsnp_table)
    det = reps["saap_detection"].copy()

    within_rate, between_rate, n_within_pairs, n_between_pairs = _saap_shared_rates(
        det,
        ds.sample_patient,
    )

    meta = _feature_metadata(ds, det.columns)
    stats = _feature_stats(ds, det.columns)

    detection_rate = det.mean(axis=0).values.astype(float)
    if reps["saap_freq"] is not None:
        dbsnp_freq = np.asarray(reps["saap_freq"], dtype=float)
        dbsnp_freq_available = ~np.isnan(dbsnp_freq)
        effective_freq = np.where(dbsnp_freq_available, dbsnp_freq, detection_rate)
    else:
        dbsnp_freq = np.full(det.shape[1], np.nan, dtype=float)
        dbsnp_freq_available = np.zeros(det.shape[1], dtype=bool)
        effective_freq = detection_rate

    rarity_weight = -np.log10(np.clip(effective_freq, 0, None) + cfg["weighted_eps"])

    pv = pd.to_numeric(meta["pvalue"], errors="coerce").fillna(0.0).clip(0.0, 20.0).values
    confidence_weight = 0.25 + 0.75 * (pv / 20.0)

    enrichment = within_rate - between_rate
    shared_ratio = np.divide(
        within_rate,
        np.clip(between_rate, 1e-12, None),
        out=np.full_like(within_rate, np.nan, dtype=float),
        where=~np.isnan(within_rate) & ~np.isnan(between_rate),
    )
    weighted_score = enrichment * rarity_weight * confidence_weight

    out = pd.DataFrame(
        {
            "peptidoform": det.columns,
            "within_shared_rate": within_rate,
            "between_shared_rate": between_rate,
            "enrichment": enrichment,
            "shared_ratio": shared_ratio,
            "rarity_weight": rarity_weight,
            "confidence_weight": confidence_weight,
            "weighted_score": weighted_score,
            "dbsnp_frequency": dbsnp_freq,
            "effective_frequency": effective_freq,
            "dbsnp_freq_available": dbsnp_freq_available,
            "n_within_pairs_total": n_within_pairs,
            "n_between_pairs_total": n_between_pairs,
        }
    ).set_index("peptidoform")

    out = out.join(meta.rename_axis("peptidoform"))
    out = out.join(stats.rename_axis("peptidoform"))
    out = out.drop(columns=["peptidoform"], errors="ignore")
    out = out.sort_values(
        ["weighted_score", "enrichment", "rarity_weight", "detection_rate"],
        ascending=[False, False, False, False],
    ).reset_index()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["selected_for_report"] = out["rank"] <= int(top_k)

    return out


def build_top_peptide_examples(
    ds: D.PeptidoformDataset,
    rf_peptides: pd.DataFrame,
    saap_features: pd.DataFrame,
    top_k=15,
):
    rf_keep = rf_peptides.nsmallest(int(top_k), "rank").copy()
    saap_keep = saap_features.nsmallest(int(top_k), "rank").copy()

    rf_cols = [
        "peptidoform",
        "rank",
        "total_importance",
        "abundance_importance",
        "missingness_importance",
    ]
    saap_cols = [
        "peptidoform",
        "rank",
        "weighted_score",
        "within_shared_rate",
        "between_shared_rate",
        "enrichment",
        "rarity_weight",
        "dbsnp_frequency",
        "dbsnp_freq_available",
    ]

    rf_tmp = rf_keep[rf_cols].rename(
        columns={
            "rank": "rf_rank",
            "total_importance": "rf_total_importance",
            "abundance_importance": "rf_abundance_importance",
            "missingness_importance": "rf_missingness_importance",
        }
    ).set_index("peptidoform")

    saap_tmp = saap_keep[saap_cols].rename(
        columns={
            "rank": "saap_rank",
            "weighted_score": "saap_weighted_score",
        }
    ).set_index("peptidoform")

    all_peptides = sorted(set(rf_tmp.index) | set(saap_tmp.index))
    meta = _feature_metadata(ds, all_peptides)
    stats = _feature_stats(ds, all_peptides)

    out = pd.DataFrame(index=all_peptides)
    out.index.name = "peptidoform"
    out = out.join(meta.rename_axis("peptidoform"))
    out = out.join(stats.rename_axis("peptidoform"))
    out = out.join(rf_tmp)
    out = out.join(saap_tmp)
    out = out.drop(columns=["peptidoform"], errors="ignore")

    def _source(row):
        has_rf = pd.notna(row.get("rf_rank"))
        has_saap = pd.notna(row.get("saap_rank"))
        if has_rf and has_saap:
            return "RF+SAAP"
        if has_rf:
            return "RF"
        if has_saap:
            return "SAAP"
        return "other"

    out["source"] = out.apply(_source, axis=1)

    combined_rank = []
    for _, row in out.iterrows():
        vals = [v for v in [row.get("rf_rank"), row.get("saap_rank")] if pd.notna(v)]
        combined_rank.append(min(vals) if vals else np.nan)
    out["combined_rank"] = combined_rank

    out = out.sort_values(
        ["combined_rank", "source", "rf_total_importance", "saap_weighted_score"],
        ascending=[True, True, False, False],
    ).reset_index()
    out.insert(0, "display_rank", np.arange(1, len(out) + 1))

    return out


def write_feature_analysis(results_dir, ds, cfg, dbsnp_table=None, top_k=25, verbose=True):
    os.makedirs(results_dir, exist_ok=True)

    rf_raw, rf_peptides = analyze_random_forest_features(ds, cfg, top_k=top_k)
    saap_df = analyze_saap_features(ds, cfg, dbsnp_table=dbsnp_table, top_k=top_k)
    top_examples = build_top_peptide_examples(ds, rf_peptides, saap_df, top_k=top_k)

    paths = {
        "rf_feature_importance_raw": os.path.join(results_dir, "rf_feature_importance_raw.csv"),
        "rf_top_peptides": os.path.join(results_dir, "rf_top_peptides.csv"),
        "saap_feature_importance": os.path.join(results_dir, "saap_feature_importance.csv"),
        "top_peptide_examples": os.path.join(results_dir, "top_peptide_examples.csv"),
        "feature_analysis_summary": os.path.join(results_dir, "feature_analysis_summary.json"),
    }

    rf_raw.to_csv(paths["rf_feature_importance_raw"], index=False)
    rf_peptides.to_csv(paths["rf_top_peptides"], index=False)
    saap_df.to_csv(paths["saap_feature_importance"], index=False)
    top_examples.to_csv(paths["top_peptide_examples"], index=False)

    summary = {
        "n_rf_features": int(len(rf_raw)),
        "n_rf_peptides": int(len(rf_peptides)),
        "n_saap_features_ranked": int(len(saap_df)),
        "n_top_examples": int(len(top_examples)),
        "top_k_requested": int(top_k),
        "top_rf_peptide": (
            None if rf_peptides.empty else str(rf_peptides.iloc[0]["peptidoform"])
        ),
        "top_saap_feature": (
            None if saap_df.empty else str(saap_df.iloc[0]["peptidoform"])
        ),
    }

    with open(paths["feature_analysis_summary"], "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print("[feature] wrote feature-analysis outputs:")
        for key, path in paths.items():
            print(f"[feature]   {key}: {path}")

    return {
        "rf_feature_importance_raw": rf_raw,
        "rf_top_peptides": rf_peptides,
        "saap_feature_importance": saap_df,
        "top_peptide_examples": top_examples,
        "summary": summary,
        "paths": paths,
    }


def _load_weightloss_dataset(data_dir, max_features=None, verbose=True):
    path = os.path.join(data_dir, "weightloss_peptidoforms.tsv")
    return D.load_peptidoform_table(
        path,
        name="weightloss",
        max_features=max_features,
        verbose=verbose,
    )


def _maybe_load_dbsnp(data_dir, verbose=True):
    path = os.path.join(data_dir, "SAAP_frequencies_dbSNP_2021.tsv")
    if not os.path.exists(path):
        return None
    return E.load_dbsnp_frequencies(path, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="Post-hoc feature analysis for the Sample Identifiability project.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--no-dbsnp", action="store_true")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg["seed"] = args.seed
    cfg["max_features"] = args.max_features

    ds = _load_weightloss_dataset(args.data_dir, max_features=args.max_features, verbose=True)
    dbsnp_table = None if args.no_dbsnp else _maybe_load_dbsnp(args.data_dir, verbose=True)

    write_feature_analysis(
        results_dir=args.results_dir,
        ds=ds,
        cfg=cfg,
        dbsnp_table=dbsnp_table,
        top_k=args.top_k,
        verbose=True,
    )


if __name__ == "__main__":
    main()