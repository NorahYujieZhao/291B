# Sample Identifiability — CSE 291 Project 2

Determine whether two blood-plasma proteomics samples come from the **same patient**, using
the MSV000080596 *Plasma weight-loss* dataset (336 mass-spectrometry runs from 58 patients
across multiple time points), reject cross-dataset matches against the external
MSV000085507 *COVID-19 sera* cohort, and assess the *worldwide identifiability* of each
sample from its detected single-amino-acid polymorphisms (SAAPs).

See [`report.md`](report.md) for the write-up (methods, evaluation design, and results).

This repository implements the approach in `291B_ProjectPlan.pdf`:

| # | Model | Representation | Notes |
|---|-------|---------------|-------|
| 1 | SAAP Jaccard similarity | binary SAAP detection vectors | parameter-free baseline |
| 2 | Population-frequency-weighted SAAP similarity | binary SAAP detection vectors | weights rare SAAPs (dbSNP allele freq), down-weights low-confidence IDs by `PValue` |
| 3 | Cosine / Euclidean similarity on PCA features | filtered log-intensity → PCA | continuous-abundance baseline |
| 4 | Random forest on pairwise differences | **log-intensity + missingness mask** (2000-D); pairwise absolute differences | same- vs different-patient; OOB + importances |
| 5 | Contrastive metric learning | filtered log-intensity → PCA | shallow MLP; contrastive loss; embedding distance as similarity |

Reference baselines: majority-class (always "different"), random scoring, raw cosine and
raw Spearman similarity on the preprocessed intensity matrix.

## Optional: evaluation figures for LaTeX / Overleaf

After a full `run.py` (so `results/identification_fdr.csv` and `results/cv_summary.csv` exist), install **matplotlib** (not listed in `requirements.txt` because the main pipeline does not need it), then:

```bash
python scripts/plot_evaluation_figures.py --results-dir results --out-dir information/figures
```

If the repo includes `pyproject.toml` / `uv.lock`, you can use `uv sync` and
`uv run python scripts/plot_evaluation_figures.py --results-dir results --out-dir information/figures` instead.

This writes three PDFs under `information/figures/`:

- `eval_fig_openset_true_fdr1.pdf`
- `eval_fig_leakfree_op_recall.pdf`
- `eval_fig_decoy_vs_observed_fdr.pdf`

## Optional: post-hoc feature analysis outputs for the report

After a full `run.py`, the pipeline can also export feature-importance summaries and top-peptide
tables for the final report. These outputs are generated automatically if `run.py` calls
`src/feature_analysis.py`, or can be generated separately with:

```bash
python src/feature_analysis.py --top-k 25
```

This writes additional files under `results/`:

- `rf_feature_importance_raw.csv`
- `rf_top_peptides.csv`
- `saap_feature_importance.csv`
- `top_peptide_examples.csv`
- `feature_analysis_summary.json`

These files are intended for the final-stage feature-importance, peptide-level, and
protein-mapping analyses.

### Feature-class correlation and supporting figures (TA Stage 2)

After ``top_peptide_examples.csv`` exists, run (or use full ``run.py``, which calls this automatically):

```bash
python src/feature_assessment.py --top-k 25
# or: uv run python scripts/plot_feature_assessment.py
```

Writes:

- `feature_class_correlation.csv` — pair-level AUROC / point-biserial *r* vs same-patient vs different-patient
- `information/figures/feature_heatmap_intensity.pdf` — log-intensity heatmap (samples ordered by patient)
- `information/figures/feature_heatmap_saap.pdf` — SAAP detection heatmap
- `information/figures/feature_hist_logdiff_pairs.pdf` — |Δlog₂| distributions for same vs different pairs
- `information/figures/feature_hist_saap_codetect.pdf` — SAAP co-detection rate by pair class

## Evaluation (three complementary views, not just AUROC)

Threshold-free rank metrics (AUROC, AUPRC, TPR@low-FPR, Recall@1/@k, mAP) tell you whether
same-patient pairs *tend to outrank* different-patient pairs — but a high AUROC does not
imply a usable fixed decision rule. So the evaluation also reports:

1. **Leak-free operating point** (`op_*` columns): a decision threshold is chosen on the
   *training* pairs at 1% true FDR, then precision / recall / observed-FDR are measured on
   the held-out *test* pairs. This is the honest counterpart of "best F1" (which is tuned on
   the test set). It exposes models whose scores are not comparable across samples.
2. **Open-set identification with FDR control** (search-style, mirrors the project guidance
   "(i) best identification per query, (ii) maximise identifications at fixed FDR"): each
   query sample is searched against all samples + an equally sized set of permuted-feature
   *decoy* samples + the external COVID cohort; the top hit is the identification. We report
   how many queries are *correctly* identified at 1% / 5% FDR — using both the target-decoy
   estimate and the true (label-based) FDR, so the optimism of the decoy model is visible —
   plus an entrapment-style empirical-null check (99.9th percentile of best-wrong-match score)
   and the count of weight-loss queries that get top-matched to a COVID sample (should be 0).
3. **Longitudinal consistency**: same-patient pair scores binned by the time-point gap, to
   check whether identification degrades for distant time points.

## Layout

```text
data/                       # datasets (downloaded; git-ignored)
src/
  download_data.py          # fetch all datasets into data/
  data.py                   # load tables, build SAAP / intensity representations,
                            #   feature selection, patient-level CV folds, pair generation
  evaluate.py               # AUROC/AUPRC/TPR@FPR, Recall@k/mAP, baselines,
                            #   cross-dataset robustness, worldwide identifiability
  feature_analysis.py       # post-hoc feature ranking and top-peptide summary export
  models.py                 # Models 1–5
  run.py                    # end-to-end pipeline (download → preprocess → CV → cross-dataset → worldwide)
scripts/
  plot_evaluation_figures.py  # PDF figures from results/*.csv → information/figures/
  plot_feature_assessment.py  # class correlation + feature heatmaps / histograms
results/                    # CSV / JSON outputs (created by run.py)
requirements.txt
pyproject.toml              # optional (some checkouts): uv; see uv.lock for pinned deps
```

## Quick start

```bash
pip install -r requirements.txt

# 1. download the datasets (~250 MB the first time; ~530 MB extra if you also want the
#    optional variant-coordinate table)
python src/download_data.py

# 2. run everything (patient-level 5-fold CV + cross-dataset test + worldwide identifiability)
python src/run.py --seed 0    # fixed seed matches numbers in report.md

# faster smoke test on a subset of features / fewer folds / fewer epochs:
python src/run.py --quick

# skip the optional analyses:
python src/run.py --no-covid --no-worldwide

# optional: run post-hoc feature analysis directly
python src/feature_analysis.py --top-k 25
```

`run.py` downloads any missing data automatically (pass `--no-download` to disable). Outputs
are printed and written to `results/`:

* `cv_summary.csv` / `cv_detail.csv` — per-model pairwise + retrieval metrics and the leak-free
  operating point (`op_*`), mean ± std over folds (and per fold);
* `cv_overfit_check.csv` — train-vs-test AUROC gap (flagged if > 0.1);
* `identification_fdr.csv` — open-set identification: correct IDs at each FDR level (target-decoy
  and true FDR), open-set Recall@1, empirical-null yield, cross-cohort hit count;
* `longitudinal_consistency.csv` — per-model same-patient score and sensitivity by time-point gap;
* `cross_dataset.csv` — within-cohort same-patient vs. weight-loss↔COVID separation per model;
* `worldwide_identifiability.csv` / `worldwide_identifiability_summary.json` — per-sample random-match
  probability and whether it is below 1 in 10 billion;
* `rf_feature_importance_raw.csv` — raw Random Forest feature importances over abundance and missingness features;
* `rf_top_peptides.csv` — peptide-level Random Forest importance summary;
* `saap_feature_importance.csv` — ranked SAAP features based on within-patient vs between-patient sharing;
* `top_peptide_examples.csv` — merged shortlist of top peptide examples for downstream analysis;
* `feature_analysis_summary.json` — summary of the exported feature-analysis outputs;
* `feature_class_correlation.csv` — pair-level association with same- vs different-patient (see feature assessment below);
* `feature_assessment_summary.json` — paths to supporting feature figures;
* `summary.json` — everything combined, including the configuration used.

## Data sources

All tables come from the class-project links (MassIVE/ProteoSAFe result views + a Google-Drive
file); `src/download_data.py` knows the exact URLs:

| File in `data/` | Source |
|---|---|
| `weightloss_peptidoforms.tsv` | MSV000080596 — "Peptidoforms intensities" (task `b11323cd…`, view `mq_peptidoforms_intensity`) — **primary data** |
| `covid_peptidoforms.tsv` | MSV000085507 — "Peptidoforms TMT" (task `81d64e27…`, view `peptidoform_expression_table`) — cross-dataset negatives |
| `SAAP_frequencies_dbSNP_2021.tsv` | dbSNP single-amino-acid-polymorphism allele frequencies (Google Drive) |
| `weightloss_variant_coords.tsv` | MSV000080596 — "Variants amino acid coordinates per protein" (large; only used for optional position-aware analysis) |

Per the dataset documentation: zeros are treated as missing (N/A), only the non-`_unmod`
intensity columns are used as features, and the data are extremely sparse (≈94% of the
40,921 peptidoforms are missing in >50% of samples) — hence the missingness/variance feature
filtering before any model is trained.

## Methodology notes

* **No identity leakage.** All splits are at the *patient* level (`data.patient_kfold`), so all
  time points of a patient stay in the same fold; preprocessing (feature selection, scaling,
  PCA, model fitting) is done on training samples only.
* **Class imbalance.** Negative (different-patient) pairs vastly outnumber positives. Metrics
  use *all* pairs within each test fold; the supervised models (4 & 5) subsample negatives at
  ~5:1 for training. AUROC/AUPRC/TPR@low-FPR and the retrieval metrics (Recall@1/@k, mAP) are
  reported alongside the baselines so trivial decision rules are visible.
* **Cross-dataset rejection.** The COVID samples are projected into the weight-loss model's
  feature space (matching peptidoform strings; non-overlapping features become missing and are
  imputed) and every weight-loss↔COVID pair is scored — none should look like a same-patient
  pair.
* **Worldwide identifiability (approximation).** For each detected genuine-SAAP peptidoform we
  look up the dbSNP allele frequency by (UniProt accession, `X→Z` substitution), taking the
  maximum frequency over protein positions when several match — a deliberately conservative
  per-SAAP estimate. SAAPs with no dbSNP frequency are excluded (the plan's conservative
  recommendation) rather than guessed. A sample's cumulative random-match probability is the
  product of these per-SAAP frequencies (independence assumption); the sample is "worldwide
  identifiable" if that product is below 1e-10. (Position-aware matching using
  `weightloss_variant_coords.tsv` would tighten this further.)
