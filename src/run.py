"""End-to-end pipeline for the Sample Identifiability project (Project 2).

Steps:
  1. ensure the datasets are present in ``data/`` (downloading them if needed);
  2. load the MSV000080596 plasma weight-loss peptidoform table and build the two
     sample representations (SAAP detection vectors / filtered log-intensity matrix);
  3. evaluate the five models from the project plan under patient-level k-fold
     cross-validation, alongside the reference baselines, reporting AUROC / AUPRC /
     TPR@low-FPR and Recall@k / mAP;
  4. test cross-dataset robustness against the external MSV000085507 COVID-19 sera cohort
     (no weight-loss sample should match a COVID sample);
  5. compute the worldwide-identifiability of each weight-loss sample from its detected
     single-amino-acid polymorphisms against dbSNP population frequencies.

Usage::

    python src/run.py                 # full run (downloads ~250 MB the first time)
    python src/run.py --quick         # smaller / faster configuration for a smoke test
    python src/run.py --no-covid --no-worldwide   # skip the optional analyses

Results are printed and also written to ``results/`` as CSV / JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D                      # noqa: E402
import models as M                    # noqa: E402
import evaluate as E                  # noqa: E402
import download_data as DL            # noqa: E402
import feature_analysis as FA
import feature_assessment as FAssess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# --------------------------------------------------------------------------------------
# Per-fold representation building (all fitting done on training samples only)
# --------------------------------------------------------------------------------------

class FoldRepresentations:
    """Builds and caches the model input matrices for one CV fold / split."""

    def __init__(self, ds: D.PeptidoformDataset, train_samples, cfg):
        self.ds = ds
        self.train_samples = list(train_samples)
        self.cfg = cfg

        # ---- Representation A: SAAP detection -------------------------------------
        det_full, _ = D.build_saap_detection(
            ds, min_detection_rate=0.0, max_pvalue_neglog10=cfg["saap_pvalue_min"]
        )
        train_rate = det_full.loc[self.train_samples].mean(axis=0)
        keep = train_rate.index[(train_rate >= cfg["saap_min_detection_rate"]).values]
        if len(keep) == 0:  # threshold too aggressive for this fold -- relax it
            keep = train_rate.sort_values(ascending=False).index[: max(20, int(0.05 * len(train_rate)))]
        self.saap_features = list(keep)
        self.saap_detection = det_full[self.saap_features]
        # per-feature info for Model 2
        fm = ds.feature_meta.reindex(self.saap_features)
        self.saap_pvalue = fm["pvalue"].values
        self.saap_freq = None  # filled in by run() if a dbSNP table is available

        # ---- Representation B1: filtered / imputed log-intensity -------------------
        # Used by the raw baselines and as the source matrix for PCA.
        self.pre = D.IntensityPreprocessor(
            max_missing=cfg["max_missing"],
            min_variance_quantile=cfg["min_variance_quantile"],
            top_k_by_variance=cfg["rf_top_k_features"],
            impute=cfg["impute"],
            log_transform=True,
            l2_normalise=False,
            include_missing_indicators=False,
        ).fit(ds.intensity.loc[self.train_samples])
        self.log_features = self.pre.output_feature_names()
        X_all = self.pre.transform(ds.intensity)  # rows aligned to ds.samples
        self.log_index = list(ds.samples)
        self.log_matrix = pd.DataFrame(X_all, index=self.log_index, columns=self.log_features)

        # ---- Representation B2: missingness-aware log-intensity --------------------
        # Used by Model 4 (RandomForestPairs).
        self.pre_missing = D.IntensityPreprocessor(
            max_missing=cfg["max_missing"],
            min_variance_quantile=cfg["min_variance_quantile"],
            top_k_by_variance=cfg["rf_top_k_features"],
            impute=cfg["impute"],
            log_transform=True,
            l2_normalise=False,
            include_missing_indicators=True,
        ).fit(ds.intensity.loc[self.train_samples])
        self.log_missing_features = self.pre_missing.output_feature_names()
        X_all_missing = self.pre_missing.transform(ds.intensity)
        self.log_missing_matrix = pd.DataFrame(
            X_all_missing,
            index=self.log_index,
            columns=self.log_missing_features,
        )

        # ---- PCA on the (centred) log-intensity matrix, fit on training samples ----
        # PCA stays on the original intensity-only representation, not the augmented one.
        from sklearn.decomposition import PCA
        n_comp = min(cfg["pca_components"], len(self.train_samples) - 1, len(self.log_features))
        n_comp = max(2, n_comp)
        self.pca = PCA(n_components=n_comp, random_state=cfg["seed"]).fit(
            self.log_matrix.loc[self.train_samples].values
        )
        Z_all = self.pca.transform(self.log_matrix.values)
        self.pca_matrix = pd.DataFrame(
            Z_all,
            index=self.log_index,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )
    
        print(
            f"[repr] log={self.log_matrix.shape} "
            f"log_missing={self.log_missing_matrix.shape} "
            f"pca={self.pca_matrix.shape}"
        )

    # ---- accessors used by the model loop ----------------------------------------

    def matrix_for(self, representation):
        if representation == "saap":
            return self.saap_detection
        if representation == "intensity_pca":
            return self.pca_matrix
        if representation == "intensity_log":
            return self.log_matrix
        if representation == "intensity_log_missing":
            return self.log_missing_matrix
        raise ValueError(representation)

    def submatrix(self, representation, samples):
        return self.matrix_for(representation).loc[list(samples)].values

# --------------------------------------------------------------------------------------
# Model construction with configurable hyperparameters
# --------------------------------------------------------------------------------------

def build_models(cfg, saap_freq=None, saap_pvalue=None):
    return [
        M.JaccardSAAP(),
        M.WeightedSAAP(feature_freq=saap_freq, feature_pvalue=saap_pvalue, eps=cfg["weighted_eps"]),
        M.SimilarityPCA(metric="cosine"),
        M.RandomForestPairs(n_estimators=cfg["rf_trees"], max_depth=cfg["rf_max_depth"],
                            min_samples_leaf=cfg["rf_min_leaf"], neg_per_pos=cfg["neg_per_pos"],
                            seed=cfg["seed"]),
        M.ContrastiveEmbedding(emb_dim=cfg["emb_dim"], hidden=cfg["emb_hidden"], dropout=cfg["emb_dropout"],
                               lr=cfg["emb_lr"], weight_decay=cfg["emb_weight_decay"], margin=cfg["emb_margin"],
                               epochs=cfg["emb_epochs"], neg_per_pos=cfg["neg_per_pos"], seed=cfg["seed"]),
    ]


BASELINES = ["majority_different", "random", "raw_cosine", "raw_spearman"]


# --------------------------------------------------------------------------------------
# Cross-validated evaluation
# --------------------------------------------------------------------------------------

def _same_patient_pair_records(ds, samples, ia, ib, lab, scores):
    """For longitudinal analysis: yield ``(time_gap, score)`` for every same-patient pair."""
    out = []
    for k in range(len(lab)):
        if lab[k] != 1:
            continue
        ti = ds.timepoint_of(samples[ia[k]])
        tj = ds.timepoint_of(samples[ib[k]])
        gap = abs(ti - tj) if (ti is not None and tj is not None) else None
        out.append((gap, float(scores[k])))
    return out


def cross_validate(ds: D.PeptidoformDataset, cfg, dbsnp_table=None):
    folds = list(D.patient_kfold(ds.patients, n_splits=cfg["n_splits"], seed=cfg["seed"]))
    per_fold_rows = []     # one row per (fold, model) with test metrics
    per_fold_train = []    # train metrics (for the overfitting check)
    longitudinal = {}      # model -> list of (time_gap, same-patient score) over test folds
    op_thresholds = {}     # model -> list of leak-free thresholds (true-FDR 1% on train pairs)
    for fold_idx, (train_patients, test_patients) in enumerate(folds):
        train_samples = [s for s in ds.samples if ds.patient_of(s) in set(train_patients)]
        test_samples = [s for s in ds.samples if ds.patient_of(s) in set(test_patients)]
        print(f"\n[cv] fold {fold_idx+1}/{cfg['n_splits']}: "
              f"{len(train_patients)} train patients ({len(train_samples)} samples), "
              f"{len(test_patients)} test patients ({len(test_samples)} samples)")
        reps = FoldRepresentations(ds, train_samples, cfg)
        if dbsnp_table is not None:
            reps.saap_freq = _saap_feature_frequencies(ds, reps.saap_features, dbsnp_table)

        # pair index sets (positions are into train_samples / test_samples respectively)
        ia_te, ib_te, lab_te = D.all_pairs(test_samples, ds.sample_patient)
        ia_tr, ib_tr, lab_tr = D.all_pairs(train_samples, ds.sample_patient)
        pid_te = [ds.patient_of(s) for s in test_samples]
        if lab_te.sum() == 0:
            print("[cv]   (fold has no within-patient test pairs -- skipping retrieval metrics)")

        models = build_models(cfg, saap_freq=reps.saap_freq, saap_pvalue=reps.saap_pvalue)
        for model in models:
            t0 = time.time()
            Xtr = reps.submatrix(model.representation, train_samples)
            Xte = reps.submatrix(model.representation, test_samples)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, [ds.patient_of(s) for s in train_samples])
                s_te = model.score_pairs(Xte, ia_te, ib_te)
                s_tr = model.score_pairs(Xtr, ia_tr, ib_tr)
            m_te = E.pairwise_metrics(lab_te, s_te)
            if lab_te.sum() > 0:
                m_te.update(E.retrieval_metrics(pid_te, ia_te, ib_te, s_te, ks=tuple(cfg["recall_ks"])))
            m_tr = E.pairwise_metrics(lab_tr, s_tr)
            # leak-free operating point: threshold chosen on TRAIN pairs at true-FDR 1%,
            # then precision / recall / F1 measured on the held-out TEST pairs.
            thr, _, _ = E.threshold_at_fdr(s_tr, lab_tr == 0, cfg["op_fdr"])
            op = E.operating_point_metrics(lab_te, s_te, thr)
            op_thresholds.setdefault(model.name, []).append(thr)
            row = {"fold": fold_idx, "model": model.name, "representation": model.representation,
                   "seconds": round(time.time() - t0, 2), **m_te,
                   "op_threshold": op["threshold"], "op_precision": op["precision"],
                   "op_recall": op["recall"], "op_F1": op["F1"], "op_observed_FDR": op["observed_FDR"]}
            if isinstance(model, M.RandomForestPairs) and model.oob_score_ is not None:
                row["rf_oob_accuracy"] = model.oob_score_
            per_fold_rows.append(row)
            per_fold_train.append({"fold": fold_idx, "model": model.name,
                                   "train_AUROC": m_tr.get("AUROC"), "test_AUROC": m_te.get("AUROC")})
            longitudinal.setdefault(model.name, []).extend(
                _same_patient_pair_records(ds, test_samples, ia_te, ib_te, lab_te, s_te))
            print(f"[cv]   {model.name:<18} AUROC={m_te.get('AUROC', float('nan')):.4f} "
                  f"AUPRC={m_te.get('AUPRC', float('nan')):.4f} "
                  f"TPR@1%FPR={m_te.get('TPR@FPR=0.01', float('nan')):.3f} "
                  f"Recall@1={m_te.get('Recall@1', float('nan')):.3f} "
                  f"mAP={m_te.get('mAP', float('nan')):.3f}  "
                  f"op[P={op['precision']:.3f} R={op['recall']:.3f}]  ({row['seconds']}s)")

        # baselines (operate on the filtered log-intensity matrix)
        for bname in BASELINES:
            s_te = E.baseline_scores(bname, reps.submatrix("intensity_log", test_samples), ia_te, ib_te,
                                     seed=cfg["seed"] + fold_idx)
            s_tr = E.baseline_scores(bname, reps.submatrix("intensity_log", train_samples), ia_tr, ib_tr,
                                     seed=cfg["seed"] + fold_idx)
            m_te = E.pairwise_metrics(lab_te, s_te)
            if lab_te.sum() > 0:
                m_te.update(E.retrieval_metrics(pid_te, ia_te, ib_te, s_te, ks=tuple(cfg["recall_ks"])))
            thr, _, _ = E.threshold_at_fdr(s_tr, lab_tr == 0, cfg["op_fdr"])
            op = E.operating_point_metrics(lab_te, s_te, thr)
            per_fold_rows.append({"fold": fold_idx, "model": f"baseline:{bname}",
                                  "representation": "intensity_log", "seconds": 0.0, **m_te,
                                  "op_threshold": op["threshold"], "op_precision": op["precision"],
                                  "op_recall": op["recall"], "op_F1": op["F1"],
                                  "op_observed_FDR": op["observed_FDR"]})

    detail = pd.DataFrame(per_fold_rows)
    metric_cols = [c for c in detail.columns if c not in ("fold", "model", "representation", "seconds")]
    summary = (detail.groupby("model")[metric_cols].agg(["mean", "std"]))
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    overfit = pd.DataFrame(per_fold_train).groupby("model")[["train_AUROC", "test_AUROC"]].mean()
    overfit["gap"] = overfit["train_AUROC"] - overfit["test_AUROC"]
    overfit["flagged_overfit"] = overfit["gap"] > 0.1
    # median leak-free threshold per model (used for the longitudinal breakdown)
    long_dfs = {}
    for mname, recs in longitudinal.items():
        thr = float(np.median(op_thresholds.get(mname, [np.nan])))
        long_dfs[mname] = E.longitudinal_consistency(recs, threshold=thr if np.isfinite(thr) else None)
    return detail, summary, overfit.reset_index(), long_dfs


# --------------------------------------------------------------------------------------
# dbSNP-derived per-SAAP-feature population frequencies (for Model 2 / cross-dataset)
# --------------------------------------------------------------------------------------

def _saap_feature_frequencies(ds: D.PeptidoformDataset, saap_features, dbsnp_table):
    fm = ds.feature_meta
    out = np.full(len(saap_features), np.nan)
    for i, pep in enumerate(saap_features):
        prob, matched = E._saap_match_probability(fm.at[pep, "annotation_wo_pos"], fm.at[pep, "proteins"],
                                                  dbsnp_table)
        if matched:
            out[i] = prob
    return out


# --------------------------------------------------------------------------------------
# Cross-dataset robustness (weight-loss model vs. external COVID cohort)
# --------------------------------------------------------------------------------------

def cross_dataset_eval(wl: D.PeptidoformDataset, covid: D.PeptidoformDataset, cfg, dbsnp_table=None):
    print("\n[cross] evaluating cross-dataset robustness against the COVID-19 sera cohort")
    out = {}
    # Train "final" models on ALL weight-loss samples.
    reps = FoldRepresentations(wl, wl.samples, cfg)
    if dbsnp_table is not None:
        reps.saap_freq = _saap_feature_frequencies(wl, reps.saap_features, dbsnp_table)

    # within-weight-loss reference scores
    ia_w, ib_w, lab_w = D.all_pairs(wl.samples, wl.sample_patient)
    pos_idx = np.where(lab_w == 1)[0]

    # COVID samples projected into each model's feature space
    covid_saap = D.covid_saap_detection_in_feature_space(covid, reps.saap_features)
    covid_log = reps.pre.transform(covid.intensity)
    covid_log_missing = reps.pre_missing.transform(covid.intensity)
    covid_pca = reps.pca.transform(covid_log)

    wl_idx = {s: i for i, s in enumerate(wl.samples)}
    n_wl, n_cov = len(wl.samples), len(covid.samples)
    # all weight-loss x COVID cross pairs (index a -> weight-loss row, index b -> covid row)
    cross_a = np.repeat(np.arange(n_wl), n_cov)
    cross_b = np.tile(np.arange(n_cov), n_wl)

    models = build_models(cfg, saap_freq=reps.saap_freq, saap_pvalue=reps.saap_pvalue)
    rows = []
    for model in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(reps.submatrix(model.representation, wl.samples), [wl.patient_of(s) for s in wl.samples])
            within = model.score_pairs(reps.submatrix(model.representation, wl.samples), ia_w, ib_w)
            within_pos = within[pos_idx]
            if model.representation == "saap":
                Xw, Xc = reps.saap_detection.values, covid_saap.values
            elif model.representation == "intensity_pca":
                Xw, Xc = reps.pca_matrix.values, covid_pca
            elif model.representation == "intensity_log_missing":
                Xw, Xc = reps.log_missing_matrix.values, covid_log_missing
            else:
                Xw, Xc = reps.log_matrix.values, covid_log
            X_stack = np.vstack([Xw, Xc])
            cross_scores = model.score_pairs(X_stack, cross_a, cross_b + n_wl)
        sep = E.cross_dataset_separation(within_pos, cross_scores)
        rows.append({"model": model.name, **sep})
        print(f"[cross]  {model.name:<18} within-pos median={sep['within_pos_score_median']:.4f} "
              f"cross max={sep['cross_score_max']:.4f}  AUROC(within+ vs cross)={sep.get('AUROC_within_vs_cross', float('nan')):.4f} "
              f"cross-false-match@95%-recall={sep.get('cross_false_match_rate@95pct_within_recall', float('nan')):.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Open-set sample identification with FDR control (the search-style evaluation)
# --------------------------------------------------------------------------------------

class _BaselineScorer:
    """Adapter so reference baselines (raw cosine / Spearman) fit the model interface."""
    representation = "intensity_log"
    def __init__(self, name):
        self.name = f"baseline:{name}"
        self._name = name
    def fit(self, X, patient_ids):
        return self
    def score_pairs(self, X, ia, ib):
        return E.baseline_scores(self._name, X, np.asarray(ia), np.asarray(ib))


def identification_evaluation(wl: D.PeptidoformDataset, covid, cfg, dbsnp_table=None):
    """Treat sample identification as a database search with FDR control.

    Per patient-level CV fold, the test-fold samples are *queries* searched against:
      * the *target* database -- all weight-loss samples;
      * a *decoy* database -- one permuted-feature decoy "sample" per target sample
        (so the target:decoy size ratio is 1, as in standard target-decoy FDR);
      * (if available) the external COVID cohort -- 92 samples that must never match.
    For every query we keep the best similarity to a same-patient sample, a
    different-patient sample, a decoy, and a COVID sample, then aggregate over folds and
    report (per model) the number of *correctly* identified queries at each FDR level,
    using both the target-decoy estimate and the true (label-based) FDR.
    """
    print("\n[ident] open-set sample identification with FDR control")
    folds = list(D.patient_kfold(wl.patients, n_splits=cfg["n_splits"], seed=cfg["seed"]))
    acc = {}  # model -> {"s_correct": [arrays], "s_wrong": [...], "s_decoy": [...], "s_cross": [...]}
    db_samples = list(wl.samples)
    db_patient = np.array([wl.patient_of(s) for s in db_samples])
    for fold_idx, (train_patients, test_patients) in enumerate(folds):
        train_samples = [s for s in wl.samples if wl.patient_of(s) in set(train_patients)]
        test_samples = [s for s in wl.samples if wl.patient_of(s) in set(test_patients)]
        reps = FoldRepresentations(wl, train_samples, cfg)
        if dbsnp_table is not None:
            reps.saap_freq = _saap_feature_frequencies(wl, reps.saap_features, dbsnp_table)
        q_idx_in_db = np.array([db_samples.index(s) for s in test_samples])
        q_patient = np.array([wl.patient_of(s) for s in test_samples])
        # COVID projected into each representation
        covid_block = {}
        if covid is not None:
            covid_block["saap"] = D.covid_saap_detection_in_feature_space(covid, reps.saap_features).values

            covid_log = reps.pre.transform(covid.intensity)
            covid_log_missing = reps.pre_missing.transform(covid.intensity)

            covid_block["intensity_log"] = covid_log
            covid_block["intensity_log_missing"] = covid_log_missing
            covid_block["intensity_pca"] = reps.pca.transform(covid_log)
        models = list(build_models(cfg, saap_freq=reps.saap_freq, saap_pvalue=reps.saap_pvalue))
        models += [_BaselineScorer(b) for b in ("raw_cosine", "raw_spearman")]  # baselines too
        for model in models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(reps.submatrix(model.representation, train_samples),
                          [wl.patient_of(s) for s in train_samples])
            Xdb = reps.matrix_for(model.representation).values
            n_db = Xdb.shape[0]
            Xdec = E.make_permuted_decoys(Xdb, n_db, seed=cfg["seed"] + 17 * fold_idx)
            blocks = [Xdb, Xdec]
            if covid is not None:
                blocks.append(covid_block[model.representation])
            offsets = np.cumsum([0] + [b.shape[0] for b in blocks])
            Xall = np.vstack(blocks)
            n_all = Xall.shape[0]
            nq = len(test_samples)
            ia = np.repeat(q_idx_in_db, n_all)              # query's row inside the (db-first) stack
            ib = np.tile(np.arange(n_all), nq)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S = model.score_pairs(Xall, ia, ib).reshape(nq, n_all)
            s_correct = np.full(nq, np.nan); s_wrong = np.full(nq, np.nan)
            s_decoy = np.full(nq, np.nan); s_cross = np.full(nq, np.nan)
            rank_true = np.full(nq, np.nan)
            db_lo, db_hi = offsets[0], offsets[1]
            dec_lo, dec_hi = offsets[1], offsets[2]
            cov_lo = offsets[2] if covid is not None else None
            for i in range(nq):
                row = S[i]
                row_db = row[db_lo:db_hi]
                qi = q_idx_in_db[i]
                same = (db_patient == q_patient[i]).copy(); same[qi] = False
                diff = (db_patient != q_patient[i])
                if same.any():
                    sc_i = row_db[same].max()
                    s_correct[i] = sc_i
                    # rank of the first same-patient candidate among ALL candidates (real
                    # non-self + decoy + cross): 1 + #candidates strictly more similar
                    not_correct = row.copy()
                    not_correct[qi] = -np.inf                       # exclude self
                    not_correct[db_lo:db_hi][same] = -np.inf        # exclude same-patient targets
                    rank_true[i] = 1 + int(np.sum(not_correct > sc_i))
                if diff.any():
                    s_wrong[i] = row_db[diff].max()
                s_decoy[i] = row[dec_lo:dec_hi].max()
                if cov_lo is not None:
                    s_cross[i] = row[cov_lo:].max()
            a = acc.setdefault(model.name, {"s_correct": [], "s_wrong": [], "s_decoy": [],
                                            "s_cross": [], "rank_true": []})
            a["s_correct"].append(s_correct); a["s_wrong"].append(s_wrong)
            a["s_decoy"].append(s_decoy); a["s_cross"].append(s_cross)
            a["rank_true"].append(rank_true)
    rows = []
    for mname, a in acc.items():
        per_query = {k: np.concatenate(v) for k, v in a.items()}
        res = E.open_set_identification(per_query, fdr_levels=tuple(cfg["fdr_levels"]),
                                        recall_ks=tuple(cfg["recall_ks"]))
        rows.append({"model": mname, **res})
        f1 = int(round(cfg["fdr_levels"][0] * 100))
        print(f"[ident]  {mname:<18} Recall@1(open-set)={res['recall@1_openset']:.3f}  "
              f"correct IDs @true{f1}%FDR={res.get(f'n_correct_trueFDR{f1}','?')}/{res['n_queries_with_true_match_in_db']}  "
              f"@decoy{f1}%FDR={res.get(f'n_correct_decoyFDR{f1}','?')}  "
              f"(est.FDR={res.get(f'decoy_estimated_FDR_decoyFDR{f1}', float('nan')):.3f} vs "
              f"true={res.get(f'observed_FDR_decoyFDR{f1}', float('nan')):.3f}, "
              f"cross-cohort hits={res.get(f'n_cross_cohort_hits_decoyFDR{f1}','?')})")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

def make_config(quick: bool):
    cfg = dict(
        seed=0,
        n_splits=5,
        recall_ks=[1, 5, 10],
        # Representation A (SAAP) -- plan section 2.2 Model 1/2 grids
        saap_pvalue_min=1.0,          # keep SAAP peptidoforms with PValue (>=-log10 p) >= this
        saap_min_detection_rate=0.10,  # SAAP must be detected in >= 10% of training samples
        weighted_eps=1e-4,
        # Representation B (intensity)
        max_missing=0.5,              # drop peptidoforms missing in > 50% of training samples
        min_variance_quantile=0.10,
        rf_top_k_features=1000,       # cap on features fed to the random forest
        impute="min",                 # per-feature minimum of observed log-intensities
        pca_components=50,
        # pair sampling
        neg_per_pos=5,
        # random forest (Model 4)
        rf_trees=300, rf_max_depth=None, rf_min_leaf=2,
        # contrastive embedding (Model 5)
        emb_dim=64, emb_hidden=128, emb_dropout=0.3, emb_lr=1e-3, emb_weight_decay=1e-4,
        emb_margin=1.0, emb_epochs=80,
        # evaluation
        op_fdr=0.01,            # FDR for the leak-free operating point (threshold from train pairs)
        fdr_levels=[0.01, 0.05],  # FDR levels for the open-set identification analysis
        max_features=None,
    )
    if quick:
        cfg.update(n_splits=3, rf_trees=80, emb_epochs=15, pca_components=20,
                   rf_top_k_features=500, max_features=8000)
    return cfg


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--quick", action="store_true", help="fast/smaller configuration for a smoke test")
    p.add_argument("--no-download", action="store_true", help="do not attempt to download missing data")
    p.add_argument("--no-covid", action="store_true", help="skip the cross-dataset (COVID) analysis")
    p.add_argument("--no-worldwide", action="store_true", help="skip the worldwide-identifiability analysis")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = make_config(args.quick)
    cfg["seed"] = args.seed
    os.makedirs(args.results_dir, exist_ok=True)

    # ---- 1. data ------------------------------------------------------------------
    wl_path = os.path.join(args.data_dir, "weightloss_peptidoforms.tsv")
    covid_path = os.path.join(args.data_dir, "covid_peptidoforms.tsv")
    dbsnp_path = os.path.join(args.data_dir, "SAAP_frequencies_dbSNP_2021.tsv")
    need = []
    if not os.path.exists(wl_path):
        need.append("weightloss_peptidoforms.tsv")
    if not args.no_covid and not os.path.exists(covid_path):
        need.append("covid_peptidoforms.tsv")
    if not args.no_worldwide and not os.path.exists(dbsnp_path):
        need.append("SAAP_frequencies_dbSNP_2021.tsv")
    if need and not args.no_download:
        print(f"[run] missing data files {need} -- downloading into {args.data_dir}")
        DL.download_all(args.data_dir)
    elif need:
        print(f"[run] WARNING: missing data files {need} and --no-download set; "
              f"run `python src/download_data.py` first.")
        if "weightloss_peptidoforms.tsv" in need:
            return 1

    wl = D.load_peptidoform_table(wl_path, "weightloss", max_features=cfg["max_features"])
    covid = None
    if not args.no_covid and os.path.exists(covid_path):
        covid = D.load_peptidoform_table(covid_path, "covid", max_features=cfg["max_features"])
    dbsnp_table = None
    if not args.no_worldwide and os.path.exists(dbsnp_path):
        try:
            dbsnp_table = E.load_dbsnp_frequencies(dbsnp_path)
        except Exception as exc:  # pragma: no cover
            print(f"[run] WARNING: failed to load dbSNP table ({exc}); worldwide-identifiability disabled")

    # ---- 2. cross-validated query-grouped (primary) + pair-level (secondary) metrics ---
    detail, summary, overfit, long_dfs = cross_validate(wl, cfg, dbsnp_table=dbsnp_table)
    print("\n=========== Cross-validated PRIMARY metrics: query-grouped retrieval (mean over folds) ===========")
    primary_cols = ["model", "Top1_accuracy_mean", "Recall@1_mean", "Recall@5_mean", "Recall@10_mean",
                    "MRR_mean", "mAP_mean"]
    primary_cols = [c for c in primary_cols if c in summary.columns]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(summary[primary_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=========== Cross-validated SECONDARY metrics: pair-level diagnostics + leak-free op. point ======")
    secondary_cols = ["model", "AUROC_mean", "AUPRC_mean", "TPR@FPR=0.01_mean", "TPR@FPR=0.001_mean",
                      "op_precision_mean", "op_recall_mean", "op_observed_FDR_mean"]
    secondary_cols = [c for c in secondary_cols if c in summary.columns]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(summary[secondary_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  (op_* = precision/recall/observed-FDR at a threshold chosen on TRAIN pairs at "
          f"{100*cfg['op_fdr']:.0f}% true FDR -- a leak-free operating point, unlike best_F1 which is tuned on test.)")
    print("\nTrain-vs-test AUROC gap (flagged if > 0.1; for the non-parametric similarity models "
          "this mostly reflects training samples participating in PCA/feature selection, not model overfitting):")
    with pd.option_context("display.width", 200):
        print(overfit.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    detail.to_csv(os.path.join(args.results_dir, "cv_detail.csv"), index=False)
    summary.to_csv(os.path.join(args.results_dir, "cv_summary.csv"), index=False)
    overfit.to_csv(os.path.join(args.results_dir, "cv_overfit_check.csv"), index=False)
    for mname, ldf in long_dfs.items():
        if not ldf.empty:
            ldf.insert(0, "model", mname)
    long_all = pd.concat([d for d in long_dfs.values() if not d.empty], ignore_index=True) if long_dfs else pd.DataFrame()
    if not long_all.empty:
        long_all.to_csv(os.path.join(args.results_dir, "longitudinal_consistency.csv"), index=False)

    # ---- 3. open-set ("search-style") sample identification with FDR control -----
    ident_df = identification_evaluation(wl, covid, cfg, dbsnp_table=dbsnp_table)
    ident_df.to_csv(os.path.join(args.results_dir, "identification_fdr.csv"), index=False)
    f1 = int(round(cfg["fdr_levels"][0] * 100)); f2 = int(round(cfg["fdr_levels"][-1] * 100))
    print("\n=========== Open-set identification: retrieval over the augmented database ===========")
    ret_cols = ["model", "Top1_accuracy", "Recall@1", "Recall@5", "Recall@10", "MRR",
                "n_queries_with_true_match_in_db"]
    ret_cols = [c for c in ret_cols if c in ident_df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(ident_df[ret_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=========== Open-set identification: number identified at fixed FDR ===========")
    id_cols = ["model", f"n_correct_trueFDR{f1}", f"n_correct_trueFDR{f2}",
               f"n_correct_decoyFDR{f1}", f"observed_FDR_decoyFDR{f1}", f"decoy_estimated_FDR_decoyFDR{f1}",
               f"n_cross_cohort_hits_decoyFDR{f1}", "n_pass_null_p99.9"]
    id_cols = [c for c in id_cols if c in ident_df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(ident_df[id_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  (n_correct_trueFDR* = correct IDs with the threshold set by the *true* FDR (labels known); "
          f"n_correct_decoyFDR{f1}/decoy_estimated_FDR = target-decoy estimate; observed_FDR_decoyFDR{f1} = "
          f"the *actual* FDR at the decoy-chosen threshold; n_pass_null_p99.9 = pass the decoy-free "
          f"entrapment threshold; cross-cohort hits should be 0.)")

    # ---- 4. longitudinal consistency (best model by op_recall) ------------------
    if not long_all.empty:
        best_model = summary.sort_values("op_recall_mean", ascending=False)["model"].iloc[0] \
            if "op_recall_mean" in summary.columns else summary["model"].iloc[0]
        sub = long_all[long_all["model"] == best_model]
        if not sub.empty:
            print(f"\n================ Longitudinal consistency (model: {best_model}) ================")
            print("  (same-patient pairs binned by time-point gap; frac_above_threshold = sensitivity "
                  "at the leak-free operating point)")
            with pd.option_context("display.width", 200):
                print(sub.drop(columns=["model"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 5. cross-dataset robustness --------------------------------------------
    cross_df = None
    if covid is not None:
        cross_df = cross_dataset_eval(wl, covid, cfg, dbsnp_table=dbsnp_table)
        cross_df.to_csv(os.path.join(args.results_dir, "cross_dataset.csv"), index=False)

    # ---- 6. worldwide identifiability -------------------------------------------
    ww_summary = None
    if dbsnp_table is not None:
        print("\n[run] computing worldwide identifiability for the weight-loss samples")
        ww_detail, ww_summary = E.worldwide_identifiability(wl, dbsnp_table)
        ww_detail.to_csv(os.path.join(args.results_dir, "worldwide_identifiability.csv"))
        with open(os.path.join(args.results_dir, "worldwide_identifiability_summary.json"), "w") as fh:
            json.dump(ww_summary, fh, indent=2)
        print(f"[run]   {ww_summary['n_identifiable']}/{ww_summary['n_samples']} samples "
              f"({100*ww_summary['frac_identifiable']:.1f}%) are uniquely identifiable worldwide "
              f"(< 1 in 10 billion).")
        
    feature_outputs = FA.write_feature_analysis(
        results_dir=args.results_dir,
        ds=wl,
        cfg=cfg,
        dbsnp_table=dbsnp_table,
        top_k=25,
        verbose=True,
    )

    assess_outputs = FAssess.write_feature_assessment(
        results_dir=args.results_dir,
        figures_dir=os.path.join(REPO_ROOT, "information", "figures"),
        ds=wl,
        cfg=cfg,
        dbsnp_table=dbsnp_table,
        top_k=25,
        seed=cfg["seed"],
        verbose=True,
    )

    # ---- write a combined summary -------------------------------------------------
    combined = {
        "config": cfg,
        "n_weightloss_samples": len(wl.samples),
        "n_weightloss_patients": len(wl.patients),
        "cv_summary": summary.to_dict(orient="records"),
        "overfit_check": overfit.to_dict(orient="records"),
        "identification_fdr": ident_df.to_dict(orient="records"),
        "longitudinal_consistency": (long_all.to_dict(orient="records") if not long_all.empty else None),
        "cross_dataset": (cross_df.to_dict(orient="records") if cross_df is not None else None),
        "worldwide_identifiability": ww_summary,
        "feature_analysis": feature_outputs["summary"],
        "feature_analysis_files": feature_outputs["paths"],
        "feature_assessment": assess_outputs.get("paths", {}),
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as fh:
        json.dump(combined, fh, indent=2, default=str)
    print(f"\n[run] wrote results to {args.results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
