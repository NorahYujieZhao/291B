# Sample Identifiability — Methods, Evaluation Design, and Results

CSE 291 class project (Project 2: *Sample identifiability*). Code in [`src/`](src/),
reproducible with `python src/run.py`. All numbers below are from the full run on the
downloaded data (`results/`).

---

## 1. Task

Given blood-plasma proteomics samples, decide whether two samples come from the **same
individual**. Concretely we learn a **scoring function** `s(q, c)` where the *query* `q` is
a sample and the *candidate* `c` is another sample; `s(q, c)` should be high iff `q` and `c`
come from the same patient. The model must additionally **reject cross-dataset matches** —
no sample from the weight-loss cohort may be matched to a sample from the external COVID-19
sera cohort — and we separately assess the **worldwide identifiability** of each sample (is
the combination of single-amino-acid polymorphisms it contains rare enough to identify the
source individual with probability < 1 in 10 billion?).

Because the scoring function is used to *rank candidates for a query* and to *control the
false-discovery rate of identifications*, the evaluation is **grouped by query sample**
(not a flat per-pair classification) — mirroring how the spectrum-matching projects evaluate
PSM/SSM scoring functions: *(i) select the best identification per query*, *(ii) maximise the
number of identifications at a fixed FDR*. Pairs are still the natural *training* instance and
flat per-pair ROC/PR are reported as *secondary* diagnostics.

## 2. Data

| File (`data/`) | Source | Role |
|---|---|---|
| `weightloss_peptidoforms.tsv` | MSV000080596 *Plasma weight loss*, "Peptidoforms intensities" (ProteoSAFe task `b11323cd…`, view `mq_peptidoforms_intensity`) | **primary** — 336 MS runs from 58 patients across up to 7 time points; 40,921 peptidoforms (7,431 carry a SAAP annotation) |
| `covid_peptidoforms.tsv` | MSV000085507 *COVID-19 sera*, "Peptidoforms TMT" (task `81d64e27…`, view `peptidoform_expression_table`) | external cohort — 92 samples, 1 per patient; used as cross-dataset negatives / decoys |
| `SAAP_frequencies_dbSNP_2021.tsv` | dbSNP single-amino-acid-polymorphism population allele frequencies (≈2.6 M rows, 19,688 proteins) | worldwide-identifiability calculation; per-SAAP weights for Model 2 |
| `weightloss_variant_coords.tsv` | MSV000080596 "Variants amino acid coordinates per protein" | optional (position-aware analysis) |

`src/download_data.py` knows the exact URLs and pulls everything into `data/`. Following the
dataset documentation: zeros are treated as missing (N/A), only the non-`_unmod` intensity
columns are used, and the data are extremely sparse (≈94% of peptidoforms missing in >50% of
samples). Patient counts: 42 patients have 7 samples, 1 has 6, 6 have 3, 9 have 2.

## 3. Methods

### 3.1 Two sample representations

* **Representation A — SAAP detection (binary).** For each sample, a binary vector over the
  peptidoforms that carry a genuine single-amino-acid polymorphism annotation (`[X→Z]`, with
  the pseudo-SAAPs `Cys→Dha`, `Gln/Glu→pyro-Glu`, `Trp→Kynurenin` excluded). Entry = 1 iff
  the peptidoform was detected (non-missing intensity). A SAAP feature is kept if it passes a
  `PValue` confidence floor and is detected in ≥10% of *training* samples.
* **Representation B — log-intensity (continuous).** Peptidoforms detected in >50% of training
  samples are removed; among the rest, the top-1000 by log-intensity variance are kept;
  observed intensities are `log2`-transformed; missing entries are imputed with the per-feature
  minimum of the observed log-values (computed on training samples). PCA (50 components, fit on
  training samples) is applied for the models that use it.

All feature selection, imputation, PCA and model fitting are done **on training samples only**.

### 3.2 The five models (from the project plan)

| # | Model | Representation | Description |
|---|---|---|---|
| 1 | **SAAP Jaccard** | A | `s(A,B) = |S_A ∩ S_B| / |S_A ∪ S_B|` over detected SAAPs. Parameter-free. |
| 2 | **Population-frequency-weighted SAAP** | A | Weighted Jaccard with `w_i = −log10(freq_i + ε)` from dbSNP (corpus detection rate as a fallback when no dbSNP frequency exists), each weight further scaled by a `PValue`-derived confidence factor. |
| 3 | **Cosine / Euclidean on PCA features** | B → PCA | Cosine similarity between PCA-reduced log-intensity vectors. |
| 4 | **Random forest on pairwise differences** | B (log) | RF classifier on `d = |log2(A) − log2(B)|` (over the filtered log features), `predict_proba` of "same patient"; reports OOB accuracy and feature importances. |
| 5 | **Contrastive metric learning** | B → PCA | Shallow MLP (BatchNorm + ReLU + dropout) trained with a contrastive loss; early stopping; `s(A,B) = −‖emb(A) − emb(B)‖`. |

**Reference baselines:** majority-class (always "different"), random scoring, raw cosine and
raw Spearman similarity on the preprocessed log-intensity matrix.

### 3.3 Pair sampling and cross-validation

Patient-level 5-fold CV: all time points of a patient stay in the same fold, so naive
sample-level splitting cannot leak identity. *Training* the supervised models (4 & 5) uses
all within-patient (positive) pairs plus a ≈5:1 subsample of between-patient (negative) pairs.
*Evaluation* uses all pairs / all candidates in the held-out fold.

## 4. Evaluation design

### 4.1 Primary metrics — query-grouped retrieval (in `cross_validate`, closed-set within a fold)

For each query sample, rank the other samples in the held-out fold by `s(q, ·)`:

* **Top-1 accuracy / Recall@k** (k = 1, 5, 10) — a same-patient sample at rank 1 / in the top k.
* **MRR** — mean reciprocal rank of the *first* same-patient sample.
* **mAP** — mean average precision treating every same-patient sample as relevant.

Queries with no same-patient sample in the fold are excluded.

### 4.2 Primary metrics — open-set ("search-style") identification with FDR control (in `identification_evaluation`)

Each held-out query is searched against the **full database** (all 336 weight-loss samples) plus
an equally sized set of **permuted-feature decoy "samples"** (each decoy column is a bootstrap
resample of a real column — keeps marginals, destroys joint structure) plus the **92 external
COVID samples** (which must never match). The top hit is the identification.

* **Top-1 / Recall@k / MRR** over the augmented database (harder than the closed-set version).
* **Number of correctly identified queries at 1% / 5% FDR**, reported two ways:
  * `n_correct_trueFDR{1,5}` — threshold chosen from the *true* FDR (we know the patient labels);
  * `n_correct_decoyFDR1`, `decoy_estimated_FDR_decoyFDR1`, and `observed_FDR_decoyFDR1` —
    the target-decoy estimate and the *actual* FDR at that threshold (so the optimism of the
    decoy model is visible — a Stage-2-style cross-check).
* **`n_pass_null_p99.9`** — an entrapment-style, decoy-free check: the 99.9th percentile of every
  query's best *wrong*-patient score is used as a single global threshold; how many queries'
  correct match clears it.
* **`n_cross_cohort_hits`** — queries whose overall top hit is a COVID sample (should be 0).

### 4.3 Secondary / diagnostic metrics

* **Pair-level ROC-AUC / PR-AUC / TPR at fixed low FPR** over all pairs in a held-out fold —
  threshold-free, rank-based diagnostics. Negatives vastly outnumber positives, so PR-AUC and
  TPR@low-FPR are more informative than ROC-AUC.
* **Leak-free operating point** (`op_*`): a decision threshold chosen on the *training* pairs at
  1% true FDR is applied to the *test* pairs — the honest counterpart of "best F1" (which would
  be tuned on the test set). It exposes models whose scores are not comparable across samples.
* **Train-vs-test AUROC gap** (flagged if > 0.1).

### 4.4 Two additional analyses

* **Cross-dataset robustness:** every weight-loss×COVID pair is scored; we report the AUROC of
  within-cohort same-patient pairs vs. cross-cohort pairs and the fraction of cross-cohort pairs
  that pass a threshold retaining 95% of true within-cohort matches.
* **Worldwide identifiability:** for each detected genuine-SAAP peptidoform we look up the dbSNP
  allele frequency by (UniProt accession, `X→Z` substitution), taking the maximum frequency
  over matching protein positions (a deliberately conservative per-SAAP estimate; SAAPs with no
  dbSNP frequency are *excluded* rather than guessed). A sample's cumulative random-match
  probability is the product of these per-SAAP frequencies (independence assumption); the sample
  is "worldwide identifiable" if that product is below 1e-10.

## 5. Results

### 5.1 Primary — query-grouped retrieval (5-fold CV, mean over folds; closed-set within fold)

| Model | Top-1 / Recall@1 | Recall@5 | Recall@10 | MRR | mAP |
|---|---|---|---|---|---|
| **M1 SAAP Jaccard** | **1.000** | 1.000 | 1.000 | 1.000 | 0.997 |
| **M2 SAAP weighted** | **1.000** | 1.000 | 1.000 | 1.000 | 0.997 |
| M3 PCA cosine | 0.940 | 0.973 | 0.976 | 0.953 | 0.597 |
| **M4 Random forest** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| M5 Contrastive | 0.967 | 0.994 | 1.000 | 0.981 | 0.933 |
| baseline raw cosine | 1.000 | 1.000 | 1.000 | 1.000 | 0.987 |
| baseline raw Spearman | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| baseline random / majority | 0.066 / 0.045 | 0.31 / 0.11 | 0.53 / 0.17 | 0.20 / 0.10 | 0.13 / 0.18 |

Within a small held-out fold, almost every method puts the right answer at rank 1 — the dataset
is "easy" in the *ranking* sense because plasma proteomic profiles are individual-specific.

### 5.2 Primary — open-set identification with FDR control (queries = 336; DB = 336 real + 336 decoy + 92 COVID)

| Model | Top-1 | Recall@5 | MRR | # correct @ **true** 1% FDR | @ true 5% FDR | # correct @ **decoy** 1% FDR | actual FDR there | pass entrapment p99.9 | cross-cohort hits |
|---|---|---|---|---|---|---|---|---|---|
| **M1 SAAP Jaccard** | 1.000 | 1.000 | 1.000 | **336 / 336** | 336 | 336 | 0.0% | 330 | 0 |
| **M2 SAAP weighted** | 1.000 | 1.000 | 1.000 | **336 / 336** | 336 | 336 | 0.0% | 331 | 0 |
| M3 PCA cosine | 0.923 | 0.964 | 0.940 | **9 / 336** | 301 | 304 | **5.3%** | 0 | 0 |
| M4 Random forest | 0.988 | 1.000 | 0.994 | **125 / 336** | 332 | 332 | 1.2% | 4 | 0 |
| M5 Contrastive | 0.952 | 0.994 | 0.973 | **190 / 336** | 320 | 317 | 3.7% | 0 | 0 |
| baseline raw cosine | 0.991 | 1.000 | 0.995 | **333 / 336** | 333 | 333 | 0.9% | 0 | 0 |
| baseline raw Spearman | 0.991 | 1.000 | 0.995 | **333 / 336** | 333 | 333 | 0.9% | 0 | 0 |

### 5.3 Secondary — pair-level diagnostics and the leak-free operating point (5-fold CV)

| Model | ROC-AUC | PR-AUC | TPR @ 1% FPR | TPR @ 0.1% FPR | op precision | **op recall** | op observed FDR | train−test AUROC gap |
|---|---|---|---|---|---|---|---|---|
| **M1 SAAP Jaccard** | 0.992 | 0.992 | 0.991 | 0.989 | 0.986 | **0.989** | 0.014 | ≈0 |
| **M2 SAAP weighted** | 0.992 | 0.993 | 0.992 | 0.992 | 0.961 | **0.992** | 0.039 | ≈0 |
| M3 PCA cosine | 0.797 | 0.492 | 0.374 | 0.196 | — | **0.000** | — | 0.199 (flagged) |
| M4 Random forest | 1.000 | 0.998 | 1.000 | 0.939 | 1.000 | **0.086** | 0.000 | ≈0 |
| M5 Contrastive | 0.987 | 0.890 | 0.726 | 0.424 | — | **0.000** | — | 0.012 |
| baseline raw cosine | 0.997 | 0.961 | 0.945 | 0.672 | — | **0.000** | — | — |
| baseline raw Spearman | 0.992 | 0.924 | 0.844 | 0.580 | — | **0.000** | — | — |
| baseline majority / random | 0.50 / 0.49 | 0.08 / 0.08 | 0 / 0 | 0 / 0 | — | 0.000 | — | — |

(`op recall = 0` and `op precision = —` means: at a threshold chosen honestly on the training
pairs, the model accepts essentially nothing on the held-out pairs.)

### 5.4 Cross-dataset robustness

| Model | within-cohort positive median score | cross-cohort max score | AUROC (within+ vs cross) | cross false-match rate at 95% within-recall |
|---|---|---|---|---|
| M1 SAAP Jaccard | 0.843 | **0.000** | 0.996 | **0.0%** |
| M2 SAAP weighted | 0.852 | **0.000** | 0.996 | **0.0%** |
| M3 PCA cosine | 0.761 | 0.998 | 0.957 | 6.6% |
| M4 Random forest | 0.962 | 0.954 | 0.994 | 2.7% |
| M5 Contrastive | (−0.13) | (−0.09) | 0.996 | 2.2% |

No weight-loss query is ever top-matched to a COVID sample, for any model (the COVID samples lose
the per-query competition). But for M3 a non-trivial fraction of the *full set* of cross-cohort
pairs would pass a permissive threshold.

### 5.5 Worldwide identifiability

**327 / 336 samples (97.3%)** have a cumulative random-match probability below 1 in 10 billion;
the median sample carries 297 SAAPs with a dbSNP frequency. Of 7,431 SAAP peptidoforms, 2,887 have
a dbSNP allele-frequency annotation. The 9 non-identifiable samples are exactly the 6 "healthy
male / female" extra subjects (`Patient_F2`, `Patient_M2`, `Patient_M3`), which have only 4–7
detected SAAPs each (apparently shallower runs); all 52 numbered patients are identifiable. Because
we take the *maximum* dbSNP frequency over matching positions, 97.3% is a conservative lower bound.

### 5.6 Longitudinal consistency (best model: M2)

| time-point gap | n same-patient pairs | mean Jaccard | min Jaccard | sensitivity at the leak-free threshold |
|---|---|---|---|---|
| 1 | 278 | 0.850 | 0.000 | 0.986 |
| 2 | 220 | 0.840 | 0.000 | 0.986 |
| 3 | 171 | 0.848 | 0.780 | 1.000 |
| 4 | 128 | 0.842 | 0.767 | 1.000 |
| 5 | 85 | 0.845 | 0.795 | 1.000 |
| 6 | 42 | 0.846 | 0.794 | 1.000 |

SAAP-detection similarity does **not** decay with the time gap between samples (mean ≈ 0.84–0.85
at every gap), confirming SAAPs behave as stable identity markers across the year-long study.

## 6. Observations / what the results say

1. **AUROC alone is misleading on this dataset.** With threshold-free rank metrics nearly every
   method looks excellent (ROC-AUC ≥ 0.99, Recall@1 = 1.0) — even a zero-parameter raw-cosine
   baseline. That only says same-patient pairs *tend to outrank* different-patient pairs, which
   follows from plasma proteomic profiles being individual-specific (plus batch effects). It does
   **not** imply a usable decision rule.

2. **Once you require a transferable decision rule, a clear hierarchy appears.** Under a leak-free
   fixed pairwise threshold (`op_*`) or under FDR-controlled open-set identification:
   * **SAAP-detection models (M1, M2)** are the clear winners — `op recall ≈ 0.99`; **336 / 336**
     queries correctly identified at 1% true FDR; they pass even the strict decoy-free entrapment
     criterion (330–331 / 336); and they reject the external cohort perfectly (0% cross false
     matches). Their Jaccard scores are bimodal (≈0.5–0.85 for same person, ≈0 for different
     person and for cross-cohort), so a natural threshold exists and transfers across folds.
   * **Raw cosine / Spearman** rank well per query (Top-1 ≈ 0.99 open-set, 333 / 336 confident
     IDs) but have **no usable global threshold** (`op recall = 0`) and fail the entrapment check
     — because the absolute similarity of a sample to *everything* depends on its sequencing
     depth / batch, so a single threshold is not meaningful.
   * **Random forest (M4)** and **contrastive embedding (M5)** are intermediate: great at ranking,
     but their `predict_proba` / embedding distances are not calibrated for a global threshold
     (`op recall ≈ 0.09 / 0.0`), so only 125 / 336 (M4) and 190 / 336 (M5) queries are confident
     at 1% true FDR.
   * **PCA-cosine (M3)** is the weakest — only 9 / 336 confident at 1% true FDR — and (a Stage-2
     point) the **target-decoy FDR estimate is unreliable for it**: it claims 304 IDs at "1% FDR"
     but the *actual* FDR there is 5.3%. For M1/M2/M4 the decoy estimate is within ≈1% of the
     truth.

3. **Worldwide identifiability is essentially universal here** — 97.3% of samples, conservatively.
   The few exceptions are low-depth runs with too few detected SAAPs, not biologically harder
   individuals. This is the strongest privacy-relevant finding: a single plasma proteomics run
   typically contains enough SAAP evidence to single out the source individual.

4. **Recommended headline metrics for the report:** (i) Top-1 accuracy and Recall@k/MRR
   (per query), (ii) **number of samples identified at 1% / 5% FDR** (target-decoy *and* — since
   we have ground-truth labels — empirical FDR), (iii) cross-cohort false-match rate. Pair-level
   ROC-AUC / PR-AUC / F1 belong in an appendix as diagnostics, not as the main result.

## 7. Limitations / Stage-2 notes

* Effective sample size is small (336 samples from 58 patients, repeated measures not
  independent); per-fold metrics have non-trivial variance, and a proper confidence interval
  would require bootstrapping over *patients* rather than over pairs.
* Pair-level metrics treat each sample pair as one instance, but pairs sharing a sample are
  correlated — fine for diagnostics, not for inference. The query-grouped metrics use the sample
  (n = 336) as the unit, which is the cleaner basis for the headline counts.
* The worldwide-identifiability calculation matches SAAPs by (accession, `X→Z`) and takes the
  maximum frequency over positions, and assumes independence between SAAPs. Position-aware
  matching using `weightloss_variant_coords.tsv` and a dependency-aware combination would tighten
  it; the current numbers are a conservative lower bound on identifiability.
* Missingness-based feature filtering, which is necessary given the sparsity, can drop rare SAAPs
  that would be the most identifying — the trade-off is exposed by the SAAP detection-rate
  hyperparameter.

## 8. Reproducing

```bash
pip install -r requirements.txt
python src/download_data.py        # datasets into data/
python src/run.py                  # full pipeline; results into results/
python src/run.py --quick          # fast smoke-test configuration
```

Outputs in `results/`: `cv_summary.csv` / `cv_detail.csv` (primary + secondary metrics, per fold
and mean±std), `cv_overfit_check.csv`, `identification_fdr.csv` (open-set identification + FDR),
`longitudinal_consistency.csv`, `cross_dataset.csv`, `worldwide_identifiability.csv` /
`…_summary.json`, and `summary.json` (everything combined, plus the configuration used).
