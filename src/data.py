"""Loading and preprocessing for the Sample Identifiability project (Project 2).

This module turns the raw MassIVE/ProteoSAFe peptidoform tables into the two sample
representations used by the models (see the project plan, section 1.3):

  * Representation A -- SAAP detection: a binary matrix ``samples x SAAP-peptidoforms``
    where entry (s, f) is 1 iff peptidoform f (which carries a single-amino-acid
    polymorphism annotation) was detected in sample s.

  * Representation B -- continuous intensity: a real matrix ``samples x peptidoforms``
    of log2 ion-count intensities after missingness filtering, low-variance filtering
    and simple imputation of the remaining missing entries.

It also provides patient-level cross-validation folds and same-/different-patient pair
generation, which are needed because the 336 samples come from only 58 patients with
repeated time points (so naive sample-level splitting would leak identity).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------------------
# Column bookkeeping
# --------------------------------------------------------------------------------------

# Metadata columns that may appear in the peptidoform / variant expression tables.
# Everything that is *not* one of these and does not look like an "unmodified-sequence"
# companion column is treated as a per-sample intensity column.
_META_COLUMNS = {
    "rowid", "ccms_row_id", "id",
    "peptidoform", "peptidoform id", "unmod peptidoform", "unmod_peptidoform",
    "variant", "variant id", "unmod variant",
    "total",
    "peptidoforms- unmodified sequence", "variants- unmodified sequence",
    "proteins", "mass", "charge", "num mods", "all mods", "is decoy",
    "lorikeet input", "orig cluster fdr", "pep prefix",
    "annotation", "annotation without position", "known", "num mod frags",
    "pvalue", "% explained", "variant fdr",
    "rep cluster task", "rep cluster user", "rep cluster index", "num tasks",
    "rep spectrum filename", "rep spectrum scan",
    "outlier groups", "outlier group ratio",
    "peptidoform", "canonical proteins", "top protein", "top canonical protein",
    "top protein fdr", "top canonical protein fdr", "protein", "start_aa", "end_aa",
    "top_canonical_protein", "canonical_proteins",
    "psp_diseases", "psp_regulatory_function", "psp_regulatory_interactions",
    "psp_modifications", "psp_ptm_variants", "psp_site_match", "drugbank_drugs",
    "num_psp_drugbank_events",
}

# "X->Z" substitutions in the annotation columns that are *not* genuine single-amino-acid
# polymorphisms (per the dataset documentation).
_NON_SAAP_SUBSTITUTIONS = {"Cys->Dha", "Gln->pyro-Glu", "Glu->pyro-Glu", "Trp->Kynurenin"}

# Ambiguous substitution targets are written as a set, e.g. "V->IL" means V->I or V->L.
_AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")


def _normalise(name: str) -> str:
    return str(name).strip().lstrip("#").lower()


def _strip_dyn_prefix(name: str) -> str:
    # ProteoSAFe exports per-sample columns as "_dyn_#<sample name>".
    return re.sub(r"^_dyn_#", "", str(name))


def _is_unmod_companion(name: str) -> bool:
    n = _normalise(_strip_dyn_prefix(name))
    return (
        n.endswith("_unmod")
        or n.endswith("- unmodified sequence")
        or n.endswith("-unmodified sequence")
        or n.endswith("- unmod")
        or n.endswith("-unmod")
    )


def _looks_like_count_column(name: str) -> bool:
    n = _normalise(_strip_dyn_prefix(name))
    return n.startswith("num_g") and n.endswith("spectra_for_unmodified_sequence")


# --------------------------------------------------------------------------------------
# SAAP annotation parsing
# --------------------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_SUBST_RE = re.compile(r"^([A-Z])->([A-Z]+)(?:/\d+)?$")


def parse_saap_substitution(annotation_without_position: str):
    """Return ``(orig_aa, [possible_target_aas])`` for a SAAP annotation, else ``None``.

    Examples
    --------
    ``"S+14[S->T/1]"``  -> ``("S", ["T"])``
    ``"V+14[V->IL/1]"`` -> ``("V", ["I", "L"])``
    ``"M+16[Oxidation]"`` -> ``None`` (not a substitution)
    ``"C-34[Cys->Dha]"``  -> ``None`` (excluded pseudo-SAAP)
    """
    if annotation_without_position is None:
        return None
    text = str(annotation_without_position)
    m = _BRACKET_RE.search(text)
    if not m:
        return None
    inside = m.group(1).strip()
    if inside in _NON_SAAP_SUBSTITUTIONS or "pyro-Glu" in inside or "Kynurenin" in inside:
        return None
    sm = _SUBST_RE.match(inside)
    if not sm:
        return None
    orig, targets = sm.group(1), sm.group(2)
    if orig not in _AA_LETTERS:
        return None
    targets = [t for t in targets if t in _AA_LETTERS and t != orig]
    if not targets:
        return None
    return orig, targets


def annotation_within_peptide_position(annotation_with_position: str):
    """Extract the 1-based position of a SAAP within its peptide from the ``Annotation``
    column, e.g. ``"A+16,11[A->S/1]"`` -> ``11``.  Returns ``None`` if unavailable."""
    if annotation_with_position is None:
        return None
    text = str(annotation_with_position)
    # The first "&"-separated event is the representative one.
    first = text.split("&")[0]
    m = re.search(r",\s*(\d+)\s*\[", first)
    if not m:
        return None
    return int(m.group(1))


def first_uniprot_accession(proteins_field: str):
    """Return the first UniProt accession from a "sp|P02768|ALBU_HUMAN;..." style field.

    Identifiers starting with ``tr|`` or that contain a ``-`` (isoforms) are deprioritised
    for the *primary* accession but still returned by :func:`all_uniprot_accessions`."""
    accs = all_uniprot_accessions(proteins_field)
    return accs[0] if accs else None


def all_uniprot_accessions(proteins_field: str):
    if proteins_field is None or (isinstance(proteins_field, float) and np.isnan(proteins_field)):
        return []
    out, seen = [], set()
    primary, secondary = [], []
    for token in re.split(r"[;|]", str(proteins_field)):
        token = token.strip()
        if not token or token in ("sp", "tr"):
            continue
        # tokens look like "sp|P02768|ALBU_HUMAN" or just "P02768"
        parts = token.split("|")
        cand = parts[1] if len(parts) >= 2 else parts[0]
        cand = cand.strip()
        if not re.match(r"^[A-Z0-9\-]{4,}$", cand):
            continue
        if cand in seen:
            continue
        seen.add(cand)
        base = cand.split("-")[0]
        if "-" in cand or token.startswith("tr"):
            secondary.append(base)
        else:
            primary.append(base)
    for a in primary + secondary:
        if a not in out:
            out.append(a)
    return out


# --------------------------------------------------------------------------------------
# Raw table loading
# --------------------------------------------------------------------------------------

@dataclass
class PeptidoformDataset:
    """A loaded peptidoform expression table.

    Attributes
    ----------
    name : str
        Short label (e.g. ``"weightloss"`` or ``"covid"``).
    feature_meta : pandas.DataFrame
        One row per peptidoform with columns ``peptidoform``, ``unmod_peptidoform``,
        ``proteins``, ``annotation``, ``annotation_wo_pos``, ``pvalue``, ``charge``.
        The DataFrame index is the peptidoform string (used as the cross-dataset key).
    intensity : pandas.DataFrame
        ``samples x peptidoforms`` matrix of raw ion-count intensities (0 -> NaN).
        Rows are sample names, columns are peptidoform strings (aligned to ``feature_meta``).
    sample_patient : dict[str, str]
        Maps each sample name to its patient id.
    sample_timepoint : dict[str, int | None]
        Maps each sample name to an integer time point index (or ``None`` if the cohort
        has a single sample per patient).
    """

    name: str
    feature_meta: pd.DataFrame
    intensity: pd.DataFrame
    sample_patient: dict
    sample_timepoint: dict = field(default_factory=dict)

    # ----- convenience views -------------------------------------------------------

    @property
    def samples(self):
        return list(self.intensity.index)

    @property
    def patients(self):
        # preserve first-seen order
        out = []
        for s in self.samples:
            p = self.sample_patient[s]
            if p not in out:
                out.append(p)
        return out

    def patient_of(self, sample):
        return self.sample_patient[sample]

    def timepoint_of(self, sample):
        return self.sample_timepoint.get(sample)

    def saap_feature_mask(self):
        """Boolean Series over peptidoforms: True if the peptidoform carries a genuine
        single-amino-acid polymorphism annotation."""
        return self.feature_meta["annotation_wo_pos"].map(
            lambda a: parse_saap_substitution(a) is not None
        )


def _patient_id_from_sample(sample_name: str) -> str:
    """Heuristically extract a patient id from a sample-column name.

    Weight-loss peptidoforms : ``"Patient_01.Timepoint_1"`` -> ``"Patient_01"``
    COVID-19 peptidoforms    : ``"Healthy.HC10.1"``         -> ``"HC10"``
                               ``"Severe-COVID-19.XG31.1"`` -> ``"XG31"``
    """
    s = _strip_dyn_prefix(sample_name)
    parts = s.split(".")
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if first.lower().startswith("patient_"):  # weight-loss style: <patient>.<timepoint>
        return first
    # condition.<patient>.<replicate> style (COVID): take the middle token if present
    if len(parts) >= 3:
        return parts[1]
    return parts[1] if len(parts) == 2 else parts[0]


def _timepoint_from_sample(sample_name: str):
    """Extract an integer time-point index from a sample name, e.g.
    ``"Patient_01.Timepoint_3"`` -> ``3``.  Returns ``None`` if there is no time point."""
    s = _strip_dyn_prefix(sample_name)
    m = re.search(r"[Tt]ime[ ._-]?point[ ._-]?(\d+)", s)
    return int(m.group(1)) if m else None


def _to_numeric_intensity(series: pd.Series) -> pd.Series:
    """Parse a per-sample column: strip thousands separators, non-numeric -> NaN, 0 -> NaN."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    num = pd.to_numeric(cleaned, errors="coerce")  # "", "N/A", etc. become NaN
    return num.where(num != 0.0, np.nan)


_KEEP_META_NORMALISED = {
    "peptidoform", "variant", "unmod peptidoform", "unmod_peptidoform", "unmod variant",
    "proteins", "annotation", "annotation without position", "pvalue", "charge",
}


def load_peptidoform_table(path: str, name: str, *, max_features=None, verbose=True) -> PeptidoformDataset:
    """Load a peptidoform expression TSV into a :class:`PeptidoformDataset`.

    Only the metadata columns we use plus the per-sample intensity columns are read
    (the parallel ``*_unmod`` companion columns and other bookkeeping columns are skipped)
    to keep the memory footprint of the ~177 MB weight-loss table manageable.
    """
    if verbose:
        print(f"[data] loading {name} peptidoform table: {path}")
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    hnorm = {c: _normalise(_strip_dyn_prefix(c)) for c in header}

    def _keep_col(c):
        n = hnorm[c]
        if n in _KEEP_META_NORMALISED:
            return True
        if n in _META_COLUMNS:
            return False
        if _is_unmod_companion(c) or _looks_like_count_column(c):
            return False
        return True  # treat as a per-sample intensity column

    usecols = [c for c in header if _keep_col(c)]
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False,
                     usecols=usecols, na_values=["", "N/A", "NA", "nan"])
    df = df[usecols]  # preserve original column order
    if max_features is not None:
        df = df.head(int(max_features)).copy()

    cols = list(df.columns)
    norm = {c: _normalise(_strip_dyn_prefix(c)) for c in cols}

    # Locate the metadata columns we care about (names differ slightly across tables).
    def find(*candidates):
        for cand in candidates:
            for c in cols:
                if norm[c] == cand:
                    return c
        return None

    col_peptidoform = find("peptidoform", "variant")
    col_unmod = find("unmod peptidoform", "unmod_peptidoform", "unmod variant")
    col_proteins = find("proteins")
    col_ann = find("annotation")
    col_ann_wo = find("annotation without position")
    col_pvalue = find("pvalue")
    col_charge = find("charge")
    if col_peptidoform is None:
        raise RuntimeError(f"could not find a 'Peptidoform'/'Variant' column in {path}")

    # Per-sample intensity columns.
    sample_cols = []
    for c in cols:
        if norm[c] in _META_COLUMNS:
            continue
        if _is_unmod_companion(c) or _looks_like_count_column(c):
            continue
        # Anything left that is not obviously metadata is a sample intensity column.
        sample_cols.append(c)
    if not sample_cols:
        raise RuntimeError(f"could not identify any per-sample intensity columns in {path}")

    sample_names = [_strip_dyn_prefix(c) for c in sample_cols]
    sample_patient = {sn: _patient_id_from_sample(sn) for sn in sample_names}
    sample_timepoint = {sn: _timepoint_from_sample(sn) for sn in sample_names}

    # Build the feature metadata frame, keyed by the (modified) peptidoform string.
    pep = df[col_peptidoform].astype(str).str.strip()
    feature_meta = pd.DataFrame({
        "peptidoform": pep.values,
        "unmod_peptidoform": (df[col_unmod].astype(str).str.strip().values if col_unmod else pep.values),
        "proteins": (df[col_proteins].values if col_proteins else None),
        "annotation": (df[col_ann].values if col_ann else None),
        "annotation_wo_pos": (df[col_ann_wo].values if col_ann_wo else None),
        "pvalue": (pd.to_numeric(df[col_pvalue], errors="coerce").values if col_pvalue else np.nan),
        "charge": (pd.to_numeric(df[col_charge], errors="coerce").values if col_charge else np.nan),
    })
    # Disambiguate duplicate peptidoform strings (rare) so the index stays unique.
    pep_index = feature_meta["peptidoform"].copy()
    dup = pep_index.duplicated(keep=False)
    if dup.any():
        counts = {}
        new_idx = []
        for v in pep_index:
            counts[v] = counts.get(v, 0) + 1
            new_idx.append(v if counts[v] == 1 else f"{v}__{counts[v]}")
        pep_index = pd.Index(new_idx)
    else:
        pep_index = pd.Index(pep_index.values)
    feature_meta.index = pep_index

    # Build the intensity matrix: samples (rows) x peptidoforms (cols).
    intensity = pd.DataFrame(index=sample_names, columns=pep_index, dtype=float)
    for raw_col, sn in zip(sample_cols, sample_names):
        intensity.loc[sn, :] = _to_numeric_intensity(df[raw_col]).values
    intensity = intensity.astype(float)

    # Drop samples with no detected peptidoforms at all (e.g. "Empty"/"Norm" channels).
    nonempty = intensity.notna().sum(axis=1) > 0
    dropped = [s for s, keep in zip(intensity.index, nonempty) if not keep]
    if dropped and verbose:
        print(f"[data]   dropping {len(dropped)} empty sample column(s): {dropped}")
    intensity = intensity.loc[nonempty]
    sample_patient = {s: sample_patient[s] for s in intensity.index}
    sample_timepoint = {s: sample_timepoint[s] for s in intensity.index}

    if verbose:
        n_pat = len(set(sample_patient.values()))
        n_saap = int(feature_meta["annotation_wo_pos"].map(lambda a: parse_saap_substitution(a) is not None).sum())
        print(f"[data]   {name}: {intensity.shape[0]} samples, {n_pat} patients, "
              f"{intensity.shape[1]} peptidoforms ({n_saap} carry a SAAP annotation)")
    return PeptidoformDataset(name=name, feature_meta=feature_meta, intensity=intensity,
                              sample_patient=sample_patient, sample_timepoint=sample_timepoint)


# --------------------------------------------------------------------------------------
# Representation A: SAAP detection vectors
# --------------------------------------------------------------------------------------

def build_saap_detection(ds: PeptidoformDataset, *, min_detection_rate=0.0, max_pvalue_neglog10=None):
    """Binary ``samples x SAAP-peptidoforms`` detection matrix.

    Parameters
    ----------
    min_detection_rate : float
        Keep only SAAP peptidoforms detected in at least this fraction of samples.
    max_pvalue_neglog10 : float or None
        ``PValue`` is stored as ``-log10(probability)`` so larger == higher confidence.
        If given, keep only SAAP peptidoforms whose ``PValue`` is at least this value
        (i.e. a confidence floor).

    Returns ``(detection_df, feature_index)`` where ``detection_df`` is float {0,1}.
    """
    saap_mask = ds.saap_feature_mask().values
    feats = ds.feature_meta.index[saap_mask]
    if max_pvalue_neglog10 is not None:
        pv = ds.feature_meta.loc[feats, "pvalue"]
        feats = feats[(pv.fillna(-np.inf) >= max_pvalue_neglog10).values]
    detection = ds.intensity[feats].notna().astype(float)
    if min_detection_rate > 0:
        rate = detection.mean(axis=0)
        feats = rate.index[(rate >= min_detection_rate).values]
        detection = detection[feats]
    return detection, list(detection.columns)


# --------------------------------------------------------------------------------------
# Representation B: filtered / imputed log-intensity matrix
# --------------------------------------------------------------------------------------

@dataclass
class IntensityPreprocessor:
    """Fit-on-train / apply-on-anything preprocessor for Representation B.

    Steps: select features by max-missingness (computed on the training samples),
    drop near-zero-variance features, log2-transform, impute missing entries
    (per-feature minimum of observed log values by default), optionally L2-normalise rows.

    If ``include_missing_indicators`` is True, ``transform()`` returns the imputed log-intensity
    features concatenated with a binary missingness mask for the same retained features.
    """

    max_missing: float = 0.5             # drop peptidoforms missing in > this fraction of train samples
    min_variance_quantile: float = 0.10  # drop the lowest-variance features (by quantile)
    top_k_by_variance: int | None = None  # if set, keep only the top-k highest-variance features
    impute: str = "min"                  # "min" | "halfmin" | "median" | "zero"
    log_transform: bool = True
    l2_normalise: bool = False
    include_missing_indicators: bool = False

    features_: list = field(default_factory=list)
    impute_values_: np.ndarray | None = None

    def _log(self, X: np.ndarray) -> np.ndarray:
        if not self.log_transform:
            return X
        return np.log2(X)

    def fit(self, intensity_df: pd.DataFrame):
        X = intensity_df.values.astype(float)
        n = X.shape[0]
        missing_frac = np.isnan(X).sum(axis=0) / max(n, 1)
        keep = missing_frac <= self.max_missing
        feats = intensity_df.columns[keep]
        Xk = X[:, keep]
        logXk = self._log(np.where(np.isnan(Xk), np.nan, Xk))

        # variance (over observed values) based filtering
        with np.errstate(invalid="ignore"):
            var = np.nanvar(logXk, axis=0)
        var = np.where(np.isnan(var), 0.0, var)
        if self.top_k_by_variance is not None and self.top_k_by_variance < len(feats):
            order = np.argsort(var)[::-1][: self.top_k_by_variance]
            sel = np.zeros(len(feats), dtype=bool)
            sel[order] = True
        else:
            thresh = np.quantile(var, self.min_variance_quantile) if len(var) else 0.0
            sel = var > thresh
            if sel.sum() == 0:
                sel = np.ones(len(feats), dtype=bool)
        feats = feats[sel]
        logXk = logXk[:, sel]

        # imputation values, computed from training data
        if self.impute == "median":
            with np.errstate(invalid="ignore"):
                impute_vals = np.nanmedian(logXk, axis=0)
        elif self.impute == "zero":
            impute_vals = np.zeros(logXk.shape[1])
        else:  # "min" or "halfmin"
            with np.errstate(invalid="ignore"):
                impute_vals = np.nanmin(logXk, axis=0)
            if self.impute == "halfmin":
                impute_vals = impute_vals - 1.0  # one log2 unit below the minimum
        impute_vals = np.where(np.isnan(impute_vals), 0.0, impute_vals)

        self.features_ = list(feats)
        self.impute_values_ = impute_vals
        return self

    def output_feature_names(self):
        if not self.include_missing_indicators:
            return list(self.features_)
        miss_names = [f"{f}__missing" for f in self.features_]
        return list(self.features_) + miss_names

    def transform(self, intensity_df: pd.DataFrame) -> np.ndarray:
        if not self.features_:
            raise RuntimeError("IntensityPreprocessor.transform called before fit")
        # Align to the fitted feature set (missing features -> all-NaN columns).
        X = intensity_df.reindex(columns=self.features_).values.astype(float)
        logX = self._log(np.where(np.isnan(X), np.nan, X))
        mask = np.isnan(logX)
        if mask.any():
            logX = np.where(mask, np.broadcast_to(self.impute_values_, logX.shape), logX)
        if self.l2_normalise:
            norms = np.linalg.norm(logX, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            logX = logX / norms
        if self.include_missing_indicators:
            miss = mask.astype(float)
            return np.concatenate([logX, miss], axis=1)
        return logX

    def fit_transform(self, intensity_df: pd.DataFrame) -> np.ndarray:
        return self.fit(intensity_df).transform(intensity_df)

# --------------------------------------------------------------------------------------
# Patient-level cross-validation folds
# --------------------------------------------------------------------------------------

def patient_kfold(patients, n_splits=5, seed=0):
    """Yield ``(train_patients, test_patients)`` splits, partitioning *patients* (not
    samples) into ``n_splits`` folds so all time points of a patient stay together."""
    patients = list(patients)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(patients))
    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(order):
        folds[i % n_splits].append(patients[idx])
    for k in range(n_splits):
        test = sorted(folds[k])
        train = sorted(p for f in (folds[:k] + folds[k + 1:]) for p in f)
        yield train, test


def patient_holdout_split(patients, train_frac=0.6, val_frac=0.2, seed=0):
    """Single patient-level train/val/test partition (60/20/20 by default)."""
    patients = list(patients)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(patients))
    n = len(patients)
    n_tr = int(round(train_frac * n))
    n_va = int(round(val_frac * n))
    tr = sorted(patients[i] for i in order[:n_tr])
    va = sorted(patients[i] for i in order[n_tr:n_tr + n_va])
    te = sorted(patients[i] for i in order[n_tr + n_va:])
    return tr, va, te


# --------------------------------------------------------------------------------------
# Same-/different-patient pair generation
# --------------------------------------------------------------------------------------

def all_pairs(samples, sample_patient):
    """All unordered sample pairs within *samples*, with a same-patient label.

    Returns ``(idx_a, idx_b, label)`` numpy arrays where the indices are positions into
    the *samples* list and ``label`` is 1 for same-patient pairs, 0 otherwise.
    """
    samples = list(samples)
    n = len(samples)
    ia, ib, lab = [], [], []
    for i in range(n):
        pi = sample_patient[samples[i]]
        for j in range(i + 1, n):
            ia.append(i)
            ib.append(j)
            lab.append(1 if pi == sample_patient[samples[j]] else 0)
    return np.array(ia), np.array(ib), np.array(lab)


def sampled_pairs(samples, sample_patient, neg_per_pos=5, seed=0, max_pos=None):
    """All within-patient (positive) pairs plus a random subsample of between-patient
    (negative) pairs at roughly ``neg_per_pos`` negatives per positive.  Used for
    *training* the supervised models (Models 4 & 5) under class imbalance."""
    ia, ib, lab = all_pairs(samples, sample_patient)
    pos = np.where(lab == 1)[0]
    neg = np.where(lab == 0)[0]
    rng = np.random.RandomState(seed)
    if max_pos is not None and len(pos) > max_pos:
        pos = rng.choice(pos, size=max_pos, replace=False)
    n_neg = min(len(neg), int(round(neg_per_pos * len(pos))))
    neg = rng.choice(neg, size=n_neg, replace=False) if n_neg < len(neg) else neg
    sel = np.concatenate([pos, neg])
    rng.shuffle(sel)
    return ia[sel], ib[sel], lab[sel]


# --------------------------------------------------------------------------------------
# Cross-dataset alignment (weight-loss vs. external COVID-19 cohort)
# --------------------------------------------------------------------------------------

def align_intensity_to_features(ds: PeptidoformDataset, feature_list):
    """Return a ``samples x len(feature_list)`` intensity DataFrame for *ds*, with
    columns that *ds* does not contain filled with NaN.  Used to put the external COVID
    samples into the same peptidoform feature space as the weight-loss model."""
    return ds.intensity.reindex(columns=list(feature_list))


def covid_saap_detection_in_feature_space(covid: PeptidoformDataset, saap_features):
    """Binary detection matrix for the COVID samples restricted to *saap_features*
    (peptidoform strings defined from the weight-loss SAAP feature set)."""
    aligned = covid.intensity.reindex(columns=list(saap_features))
    return aligned.notna().astype(float)
