"""Evaluation utilities for the Sample Identifiability project (Project 2).

Implements (project plan, section 3):

  * pairwise matching metrics -- AUROC, AUPRC, TPR at a fixed low FPR, best-F1 operating
    point;
  * retrieval-style metrics -- Recall@1, Recall@k, mean average precision (mAP);
  * reference baselines -- majority-class, random, and raw cosine / Spearman similarity;
  * cross-dataset robustness -- how well within-cohort same-patient pairs separate from
    weight-loss <-> external-COVID pairs (which must all be scored as "different");
  * worldwide identifiability -- per sample, the cumulative random-match probability of
    its detected single-amino-acid polymorphisms against dbSNP population frequencies,
    and whether that probability is below 1 in 10 billion.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import all_uniprot_accessions, parse_saap_substitution  # noqa: E402

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


# --------------------------------------------------------------------------------------
# Pairwise metrics
# --------------------------------------------------------------------------------------

def tpr_at_fpr(y_true, scores, target_fpr):
    """True-positive rate at the largest score threshold whose false-positive rate does
    not exceed ``target_fpr``.  Returns ``nan`` if there are no negatives."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0 or (y_true == 0).sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = fpr <= target_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def best_f1(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    ys = y_true[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    denom = (2 * tp + fp + fn)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    if len(f1) == 0:
        return 0.0, float("nan")
    k = int(np.argmax(f1))
    thr = float(scores[order][k])
    return float(f1[k]), thr


def pairwise_metrics(y_true, scores, fpr_targets=(0.01, 0.001)):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    out = {"n_pairs": int(len(y_true)), "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum())}
    if y_true.sum() == 0 or (y_true == 0).sum() == 0:
        out.update({"AUROC": float("nan"), "AUPRC": float("nan")})
    else:
        out["AUROC"] = float(roc_auc_score(y_true, scores))
        out["AUPRC"] = float(average_precision_score(y_true, scores))
    for t in fpr_targets:
        out[f"TPR@FPR={t:g}"] = tpr_at_fpr(y_true, scores, t)
    f1v, thr = best_f1(y_true, scores)
    out["best_F1"] = f1v
    out["best_F1_threshold"] = thr
    return out


# --------------------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------------------

def _similarity_matrix(n, ia, ib, scores):
    S = np.full((n, n), -np.inf, dtype=float)
    S[ia, ib] = scores
    S[ib, ia] = scores
    np.fill_diagonal(S, -np.inf)
    return S


def retrieval_metrics(patient_ids, ia, ib, scores, ks=(1, 5, 10)):
    """Query-grouped retrieval metrics.

    For each query sample (= any sample that has at least one other same-patient sample in
    the set) the remaining samples are ranked by similarity and we compute:
      * Top-1 accuracy / Recall@k -- whether a same-patient sample appears at rank 1 / in
        the top k;
      * MRR -- mean reciprocal rank of the *first* same-patient sample;
      * mAP -- mean average precision treating every same-patient sample as relevant.
    Queries with no same-patient sample in the set are excluded (they cannot be retrieved).
    """
    pid = np.asarray(patient_ids)
    n = len(pid)
    S = _similarity_matrix(n, np.asarray(ia), np.asarray(ib), np.asarray(scores, dtype=float))
    recall_hits = {k: [] for k in ks}
    aps, rrs = [], []
    for q in range(n):
        rel = (pid == pid[q])
        rel[q] = False
        if not rel.any():
            continue
        order = np.argsort(S[q], kind="stable")[::-1]  # most similar first (stable for ties)
        ranked_rel = rel[order]
        for k in ks:
            recall_hits[k].append(1.0 if ranked_rel[:k].any() else 0.0)
        hits = np.where(ranked_rel)[0]                  # 0-based ranks of relevant items
        rrs.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)
        precisions = [(i + 1) / (rank + 1) for i, rank in enumerate(hits)]
        aps.append(float(np.mean(precisions)) if precisions else 0.0)
    out = {"n_queries": len(aps)}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recall_hits[k])) if recall_hits[k] else float("nan")
    out["Top1_accuracy"] = out.get("Recall@1", float("nan"))
    out["MRR"] = float(np.mean(rrs)) if rrs else float("nan")
    out["mAP"] = float(np.mean(aps)) if aps else float("nan")
    return out


# --------------------------------------------------------------------------------------
# Baseline scorers (operate on the filtered log-intensity matrix)
# --------------------------------------------------------------------------------------

def baseline_scores(name, X, ia, ib, seed=0):
    ia = np.asarray(ia); ib = np.asarray(ib)
    if name == "random":
        return np.random.RandomState(seed).rand(len(ia))
    if name == "majority_different":
        return np.zeros(len(ia))                       # always "different patient"
    if name == "raw_cosine":
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        return np.sum(Xn[ia] * Xn[ib], axis=1)
    if name == "raw_neg_euclidean":
        return -np.linalg.norm(X[ia] - X[ib], axis=1)
    if name == "raw_spearman":
        from scipy.stats import rankdata
        R = np.apply_along_axis(rankdata, 1, X)
        Rn = (R - R.mean(axis=1, keepdims=True))
        Rn = Rn / np.clip(np.linalg.norm(Rn, axis=1, keepdims=True), 1e-12, None)
        return np.sum(Rn[ia] * Rn[ib], axis=1)
    raise ValueError(f"unknown baseline: {name}")


# --------------------------------------------------------------------------------------
# FDR control on labelled pairs + leak-free operating point
# --------------------------------------------------------------------------------------

def threshold_at_fdr(scores, is_false, fdr_level):
    """Highest-yield score threshold whose accepted set (``scores >= threshold``) has a
    q-value (running-min FDR) <= ``fdr_level``.  Tied scores are handled correctly: only
    the end of each tied-score run is a valid cut point, so an all-equal score vector (e.g.
    the majority baseline) cannot be split into a spuriously clean prefix.  Returns
    ``(threshold, n_accepted, n_accepted_true)``; ``threshold`` is ``+inf`` if nothing
    qualifies."""
    scores = np.asarray(scores, dtype=float)
    is_false = np.asarray(is_false, dtype=bool)
    n = len(scores)
    if n == 0:
        return float("inf"), 0, 0
    order = np.argsort(scores, kind="mergesort")[::-1]   # high score -> low score, stable
    s = scores[order]
    f = is_false[order].astype(float)
    fdr = np.cumsum(f) / np.arange(1, n + 1, dtype=float)
    block_end = np.ones(n, dtype=bool)
    block_end[:-1] = s[:-1] != s[1:]                     # only run ends are valid thresholds
    fdr_be = np.where(block_end, fdr, np.inf)
    q = np.minimum.accumulate(fdr_be[::-1])[::-1]        # monotone q-value over valid cut points
    ok = block_end & (q <= fdr_level + 1e-12)
    if not ok.any():
        return float("inf"), 0, 0
    cut = int(np.where(ok)[0].max())
    thr = float(s[cut])
    accepted = scores >= thr - 1e-12
    return thr, int(accepted.sum()), int((accepted & ~is_false).sum())


def operating_point_metrics(y_true, scores, threshold):
    """Precision / recall / F1 / TP / FP / FN at a fixed decision threshold."""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores, dtype=float) >= threshold
    tp = int((pred & y_true).sum()); fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum()); tn = int((~pred & ~y_true).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else 0.0)
    return {"threshold": float(threshold), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "F1": f1,
            "observed_FDR": (fp / (tp + fp) if (tp + fp) else float("nan"))}


# --------------------------------------------------------------------------------------
# Open-set sample identification with FDR control (the search-style evaluation)
# --------------------------------------------------------------------------------------

def make_permuted_decoys(X, n_decoys, seed=0):
    """Decoy "samples": each is built by independently permuting, within every feature
    column, the values drawn from real samples -- this keeps each feature's marginal
    distribution but destroys the joint structure that makes a real sample identifiable.
    Returns an ``(n_decoys, n_features)`` array."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.RandomState(seed)
    out = np.empty((n_decoys, d), dtype=float)
    for j in range(d):
        col = X[:, j]
        out[:, j] = col[rng.randint(0, n, size=n_decoys)]
    return out


def open_set_identification(per_query, *, fdr_levels=(0.01, 0.05), recall_ks=(1, 5, 10),
                            null_percentiles=(95.0, 99.0, 99.9)):
    """Open-set ("search-style") identification evaluation.

    ``per_query`` is a dict of equal-length 1-D arrays, one entry per query sample:
      ``s_correct`` -- best similarity to a *same-patient* database sample (NaN if the
                       query's patient has no other sample in the database);
      ``s_wrong``   -- best similarity to a *different-patient* database sample;
      ``s_decoy``   -- best similarity to a permuted-feature decoy sample (decoy DB sized
                       equal to the target DB, so the target:decoy ratio is 1);
      ``s_cross``   -- best similarity to an external-cohort (COVID) sample, or NaN if not
                       evaluated;
      ``rank_true`` -- 1-based rank of the *first* same-patient sample among ALL candidates
                       (real + decoy + cross), used for Recall@k / MRR (NaN if none).

    Reports the query-grouped retrieval metrics (Top-1, Recall@k, MRR) over the full
    augmented database, and -- for each FDR level -- how many queries are *correctly*
    identified, both with the threshold chosen from the target-decoy estimate and with the
    threshold chosen from the (label-based) true FDR, so the optimism/pessimism of the
    decoy model is visible.  Also reports the entrapment-style empirical "wrong-match" null
    percentiles (decoy-free) and the number of queries that get top-matched to an external
    COVID sample (which should be zero).
    """
    sc = np.asarray(per_query["s_correct"], dtype=float)
    sw = np.asarray(per_query["s_wrong"], dtype=float)
    sd = np.asarray(per_query["s_decoy"], dtype=float)
    sx = np.asarray(per_query.get("s_cross", np.full_like(sc, np.nan)), dtype=float)
    rt = np.asarray(per_query.get("rank_true", np.full_like(sc, np.nan)), dtype=float)
    has_true = ~np.isnan(sc)                       # queries that *can* be identified
    n_q = len(sc)
    n_eligible = int(has_true.sum())

    # ---- open-set Recall@1: does the true match out-rank every wrong / decoy / cross hit?
    competitors = np.fmax(np.nan_to_num(sw, nan=-np.inf),
                          np.fmax(np.nan_to_num(sd, nan=-np.inf), np.nan_to_num(sx, nan=-np.inf)))
    correct_argmax = has_true & (sc > competitors)
    recall1_openset = float(correct_argmax.sum() / n_eligible) if n_eligible else float("nan")
    # ---- query-grouped retrieval metrics over the augmented database -----------------
    rt_e = rt[has_true]
    rt_e = rt_e[~np.isnan(rt_e)]
    recall_at_k = {k: (float(np.mean(rt_e <= k)) if len(rt_e) else float("nan")) for k in recall_ks}
    mrr = float(np.mean(1.0 / rt_e)) if len(rt_e) else float("nan")

    # ---- empirical wrong-match null (entrapment-style, decoy-free) -------------------
    null = sw[~np.isnan(sw)]
    null_thresholds = {p: float(np.percentile(null, p)) for p in null_percentiles} if len(null) else {}
    null_yield = {}
    for p, thr in null_thresholds.items():
        # queries whose true match both wins and clears the empirical-null threshold
        null_yield[p] = int((correct_argmax & (sc >= thr)).sum())

    # ---- target / decoy / observed FDR over the per-query "winning hit" --------------
    # winning target hit per query = max(s_correct, s_wrong); a "false" target hit is one
    # where s_wrong >= s_correct (i.e. the best target match is the wrong patient).
    win_target = np.fmax(np.nan_to_num(sc, nan=-np.inf), np.nan_to_num(sw, nan=-np.inf))
    is_false_target = ~(has_true & (sc >= np.nan_to_num(sw, nan=-np.inf)))
    win_decoy = np.nan_to_num(sd, nan=-np.inf)

    # decoy-competition: for FDR estimation we compare each query's best target hit to its
    # best decoy hit; the larger one is the query's "PSM".  is_decoy_psm marks decoy wins.
    psm_score = np.fmax(win_target, win_decoy)
    is_decoy_psm = win_decoy > win_target

    def _summary_at_threshold(thr, label):
        accept = psm_score >= thr - 1e-12
        n_target_acc = int((accept & ~is_decoy_psm).sum())
        n_decoy_acc = int((accept & is_decoy_psm).sum())
        n_correct = int((accept & ~is_decoy_psm & has_true & (sc >= np.nan_to_num(sw, nan=-np.inf))).sum())
        n_cross_acc = int((accept & ~np.isnan(sx) & (sx > win_target) & (sx > win_decoy)).sum())
        denom = max(n_target_acc, 1)
        return {f"threshold_{label}": float(thr),
                f"n_identified_{label}": n_target_acc,
                f"n_correct_{label}": n_correct,
                f"observed_FDR_{label}": 1.0 - n_correct / denom,
                f"decoy_estimated_FDR_{label}": n_decoy_acc / denom,
                f"n_cross_cohort_hits_{label}": n_cross_acc}

    out = {"n_queries": n_q, "n_queries_with_true_match_in_db": n_eligible,
           "Top1_accuracy": recall_at_k.get(1, recall1_openset),
           "recall@1_openset": recall1_openset, "MRR": mrr,
           "argmax_correct": int(correct_argmax.sum())}
    out.update({f"Recall@{k}": v for k, v in recall_at_k.items()})
    out.update({f"null_threshold_p{p:g}": t for p, t in null_thresholds.items()})
    out.update({f"n_pass_null_p{p:g}": v for p, v in null_yield.items()})
    for f in fdr_levels:
        # threshold from the target-decoy estimate
        thr_d, _, _ = threshold_at_fdr(psm_score, is_decoy_psm, f)
        out.update(_summary_at_threshold(thr_d, f"decoyFDR{int(round(f*100))}"))
        # threshold from the *true* FDR (only the target PSMs participate)
        tmask = ~is_decoy_psm
        thr_o, _, _ = threshold_at_fdr(psm_score[tmask], is_false_target[tmask], f)
        out.update(_summary_at_threshold(thr_o, f"trueFDR{int(round(f*100))}"))
    return out


# --------------------------------------------------------------------------------------
# Longitudinal consistency: does the score decay with the time gap between samples?
# --------------------------------------------------------------------------------------

def longitudinal_consistency(same_patient_pairs, threshold=None):
    """``same_patient_pairs`` is a list of ``(time_gap, score)`` tuples for within-patient
    pairs.  Returns, per integer time gap, the count, mean / min score, and (if a decision
    ``threshold`` is given) the fraction of pairs at that gap that score above it -- i.e.
    sensitivity broken down by how far apart the two time points are."""
    import collections
    by_gap = collections.defaultdict(list)
    for gap, score in same_patient_pairs:
        if gap is None:
            continue
        by_gap[int(gap)].append(float(score))
    rows = []
    for gap in sorted(by_gap):
        vals = np.array(by_gap[gap])
        row = {"time_gap": gap, "n_pairs": len(vals), "mean_score": float(vals.mean()),
               "min_score": float(vals.min())}
        if threshold is not None:
            row["frac_above_threshold"] = float((vals >= threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Cross-dataset robustness
# --------------------------------------------------------------------------------------

def cross_dataset_separation(within_pos_scores, cross_scores):
    """How separable are within-cohort same-patient pairs from cross-cohort pairs.

    Returns AUROC treating within-cohort positives as the positive class and all
    cross-cohort (weight-loss vs. COVID) pairs as negatives, plus summary statistics of
    the cross-cohort score distribution (which should sit below the within distribution)
    and -- at an operating threshold that retains 95% of true within-cohort matches --
    the fraction of cross-cohort pairs that leak through as false matches.
    """
    wp = np.asarray(within_pos_scores, dtype=float)
    cs = np.asarray(cross_scores, dtype=float)
    out = {
        "n_within_pos": int(len(wp)), "n_cross": int(len(cs)),
        "within_pos_score_p05": float(np.percentile(wp, 5)) if len(wp) else float("nan"),
        "within_pos_score_median": float(np.median(wp)) if len(wp) else float("nan"),
        "cross_score_max": float(np.max(cs)) if len(cs) else float("nan"),
        "cross_score_median": float(np.median(cs)) if len(cs) else float("nan"),
    }
    if len(wp) and len(cs):
        y = np.concatenate([np.ones(len(wp)), np.zeros(len(cs))])
        s = np.concatenate([wp, cs])
        out["AUROC_within_vs_cross"] = float(roc_auc_score(y, s))
        thr = float(np.percentile(wp, 5))  # keep 95% of true within-cohort matches
        out["cross_false_match_rate@95pct_within_recall"] = float(np.mean(cs >= thr))
    return out


# --------------------------------------------------------------------------------------
# Worldwide identifiability from dbSNP SAAP frequencies
# --------------------------------------------------------------------------------------

_IDENTIFIABILITY_THRESHOLD = 1e-10  # "< 1 in 10 billion"


def load_dbsnp_frequencies(path, verbose=True):
    """Return a dict ``{accession: {("X->Z"): (max_freq, min_nonzero_freq, n_rows, median_AN)}}``.

    The dbSNP TSV columns are: Accession, Variation, Location, refSNP cluster ID number,
    AC (allele count), AN (total observations), Frequency(ratio) = AC / AN.
    """
    if verbose:
        print(f"[eval] loading dbSNP SAAP frequency table: {path}")
    df = pd.read_csv(
        path, sep="\t",
        usecols=["Accession", "Variation", "AC", "AN", "Frequency(ratio)"],
        dtype={"Accession": "string", "Variation": "string"},
    )
    df["AC"] = pd.to_numeric(df["AC"], errors="coerce").fillna(0).astype("int64")
    df["AN"] = pd.to_numeric(df["AN"], errors="coerce").fillna(0).astype("int64")
    df["Frequency(ratio)"] = pd.to_numeric(df["Frequency(ratio)"], errors="coerce").fillna(0.0)
    table = {}
    grouped = df.groupby(["Accession", "Variation"], observed=True)
    for (acc, var), g in grouped:
        freqs = g["Frequency(ratio)"].values
        nonzero = freqs[freqs > 0]
        an = g["AN"].values
        an_pos = an[an > 0]
        # Laplace-style floor for variants observed with allele count 0:
        floor = (1.0 / float(np.median(an_pos))) if len(an_pos) else 1e-6
        max_freq = float(np.max(freqs)) if len(freqs) else 0.0
        if max_freq <= 0:
            max_freq = floor
        min_nonzero = float(np.min(nonzero)) if len(nonzero) else floor
        table.setdefault(str(acc), {})[str(var)] = (max_freq, min_nonzero, int(len(g)),
                                                    float(np.median(an_pos)) if len(an_pos) else float("nan"))
    if verbose:
        print(f"[eval]   indexed {len(table)} proteins / {len(grouped)} (protein, variation) entries")
    return table


def _saap_match_probability(annotation_wo_pos, proteins_field, dbsnp_table):
    """Return ``(prob, matched)`` for one SAAP peptidoform.

    ``prob`` is the (conservative) probability that a random individual carries this
    polymorphism -- the maximum dbSNP allele frequency across all protein positions that
    match this (accession, X->Z) pair.  ``matched`` is False if no dbSNP entry was found,
    in which case the SAAP is *excluded* from the identifiability product (the project
    plan's conservative recommendation) rather than guessed.
    """
    parsed = parse_saap_substitution(annotation_wo_pos)
    if parsed is None:
        return None, False
    orig, targets = parsed
    variations = [f"{orig}->{t}" for t in targets]
    best = None
    for acc in all_uniprot_accessions(proteins_field):
        per_acc = dbsnp_table.get(acc)
        if not per_acc:
            continue
        for v in variations:
            hit = per_acc.get(v)
            if hit is None:
                continue
            f = hit[0]
            best = f if best is None else max(best, f)
    if best is None:
        return None, False
    return float(min(max(best, 1e-12), 1.0)), True


def worldwide_identifiability(ds, dbsnp_table, *, verbose=True):
    """Per-sample worldwide identifiability assessment for a peptidoform dataset.

    Returns a DataFrame indexed by sample with columns:
      n_detected_saaps        -- detected peptidoforms carrying a genuine SAAP annotation
      n_informative_saaps     -- of those, how many have a dbSNP population frequency
      log10_random_match_prob -- log10 product of match probabilities over informative SAAPs
      identifiable             -- True iff random-match probability < 1e-10
    plus a summary dict.
    """
    fm = ds.feature_meta
    saap_rows = fm.index[fm["annotation_wo_pos"].map(lambda a: parse_saap_substitution(a) is not None)]
    # Precompute per-SAAP-feature match probabilities once.
    feat_prob = {}
    for pep in saap_rows:
        p, matched = _saap_match_probability(fm.at[pep, "annotation_wo_pos"], fm.at[pep, "proteins"], dbsnp_table)
        feat_prob[pep] = (p, matched)

    det = ds.intensity[saap_rows].notna()
    records = []
    for s in ds.samples:
        detected = [pep for pep in saap_rows if det.at[s, pep]]
        informative = [pep for pep in detected if feat_prob[pep][1]]
        if informative:
            log10p = float(np.sum(np.log10([feat_prob[pep][0] for pep in informative])))
        else:
            log10p = float("nan")
        records.append({
            "sample": s,
            "patient": ds.patient_of(s),
            "n_detected_saaps": len(detected),
            "n_informative_saaps": len(informative),
            "log10_random_match_prob": log10p,
            "random_match_prob": (10.0 ** log10p) if informative else float("nan"),
            "identifiable": bool(informative) and (log10p < np.log10(_IDENTIFIABILITY_THRESHOLD)),
        })
    out = pd.DataFrame.from_records(records).set_index("sample")
    summary = {
        "n_samples": int(len(out)),
        "n_identifiable": int(out["identifiable"].sum()),
        "frac_identifiable": float(out["identifiable"].mean()),
        "median_detected_saaps": float(out["n_detected_saaps"].median()),
        "median_informative_saaps": float(out["n_informative_saaps"].median()),
        "n_saap_features": int(len(saap_rows)),
        "n_saap_features_with_freq": int(sum(1 for v in feat_prob.values() if v[1])),
        "identifiability_threshold": _IDENTIFIABILITY_THRESHOLD,
    }
    if verbose:
        print(f"[eval]   worldwide identifiability: {summary['n_identifiable']}/{summary['n_samples']} samples "
              f"({100*summary['frac_identifiable']:.1f}%) reach < 1 in 10 billion; "
              f"median {summary['median_informative_saaps']:.0f} frequency-annotated SAAPs/sample")
    return out, summary
