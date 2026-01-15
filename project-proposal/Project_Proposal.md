# M.S. Applied Data Science: Capstone Project Proposal

**Due Date:** January 25, 2025  
**Student Name:** Angel Morenu  
**Course:** EGN 6933 – Project in Applied Data Science  
**Project Type:** Individual

---

## 1. Project Title & Team Members

**Project Name:** Machine Learning Classification of Pathogenic vs. Benign Non-Coding Genetic Variants Using DNA Sequence Embeddings  

**Team Lead:** Angel Morenu (Individual Project)

**Collaboration:** Dylan Tan (data/features collaborator; provides an external pathogenicity dataset with precomputed ESM2 embedding features for benchmarking/augmentation)

---

## 2. Problem Statement & Impact

The majority of genetic variants associated with human disease lie in non-coding regions of the genome, where their functional effects are difficult to interpret experimentally. While coding variants often have clearer consequences on protein structure, non-coding variants exert subtler regulatory effects and remain a major bottleneck in rare disease interpretation and precision medicine.

This capstone focuses on a prediction-first, semester-feasible supervised task: **binary classification of non-coding variants as pathogenic vs. benign** using DNA sequence representations. The objective is not to fully resolve biological mechanisms, but to build a reproducible ML pipeline that can prioritize non-coding variants for downstream analysis.

**Stakeholders:** rare disease research teams, clinical genomics analysts (decision support), and computational genomics researchers.

**Societal/Ethical context:** ClinVar is public and de-identified, but the project will treat variant data as potentially sensitive; model outputs are **not for clinical diagnosis**. Class imbalance and label uncertainty will be handled explicitly (exclude VUS/conflicting records; report imbalance-aware metrics).

**Multidisciplinary reach:** This project connects applied data science (ML, evaluation, deployment) with computational genomics and healthcare/rare-disease workflows, where variant prioritization can reduce manual review burden and focus follow-up research.

---

## 3. Data Acquisition & Viability

**Data Source (URL/provider):** ClinVar public releases (variant summaries/VCF): https://www.ncbi.nlm.nih.gov/clinvar/  

**Dataset in hand (verified):** ClinVar `variant_summary.txt.gz` has been downloaded to `Project/data/clinvar/variant_summary.txt.gz` and the file header/rows were successfully parsed (verified January 13, 2026).

**Additional dataset (collaboration):** Dylan Tan’s pathogenicity dataset with precomputed ESM2 embedding features (accessed via UF HiPerGator).

- **Main path:** `/blue/xiaofan/dtan1/Output_Folder/ESM2_PrimateAI_Scripts/`

- **Baseline directory:** `Baseline/` (includes `esm2_selected_features.pkl` and gene-specific datasets such as `esm2_BRCA1_embed.pkl`)

- **CoordsData directory:** `CoordsData/` (includes `esm2_selected_coord_features.pkl`, which appends coordinate columns)

- **Reference code for loading/cleaning:** `/blue/xiaofan/dtan1/Scripts/ESM2_PrimateAI_Scripts/FinTest_MLP.py`

- **Loading note:** these `.pkl` files are saved record-by-record (multiple pickled objects in one file), so loading requires iteratively unpickling in a loop until EOF (see `load_data_clean_data(...)` in `FinTest_MLP.py`).

**HiPerGator working path (student):** `/blue/xiaofan/angel.morenu` (used for staging project outputs, logs, and derived artifacts when running on UF HiPerGator).

**Local preprocessing (standardized naming):** For reproducibility, the project uses neutral, dataset-descriptive script and output names for this collaborator dataset (while retaining Dylan Tan attribution in documentation). The scripts `scripts/inspect_esm2_primateai_pkl.py` and `scripts/ingest_esm2_primateai.py` validate the `(2560,)` embedding shape and write versioned Parquet outputs under `data/processed/esm2_primateai/` (`data/processed/esm2_primateai/esm2_primateai_v3_baseline_strict.parquet`, `label_missing=0`).

**Label definition / filtering:**

- Keep Pathogenic/Likely pathogenic vs. Benign/Likely benign.

- Exclude VUS and conflicting interpretations to reduce label noise.

**Non-coding subset definition:**

- Use consequence annotation (Ensembl VEP) to filter to non-coding consequences (e.g., intronic, UTR, upstream/downstream, intergenic).

**Feature construction:**

- Extract a fixed-length DNA sequence window around each variant (final window size selected after EDA; planned sweep: 101 bp, 201 bp, 501 bp).

- Compute pretrained DNA embedding features from each window (DNABERT-style). Optional ablation (time/compute permitting): compare one newer DNA foundation model embedding to test robustness.

- Incorporate ESM2 embedding features from the collaborator dataset as an additional feature set for benchmarking (and, if compatible with labels/records, optional augmentation).

**Data pipeline plan (cleaning, feature engineering, storage):**

- Standardize to a single genome assembly per experiment (GRCh37 or GRCh38) and remove duplicates. Default target is GRCh38 unless EDA indicates a need for GRCh37 compatibility.

- Store the curated labeled table as versioned Parquet/CSV (including split assignment and filters used).

- Cache embeddings to disk so model training is reproducible and does not require re-embedding every run.

**Data viability:**

- ClinVar is public, de-identified, and accessible without special permissions.

- The collaborator dataset and ESM2 features will be accessed through UF HiPerGator (OnDemand Jupyter) and handled under course research/data-handling best practices (no PHI; research-only; no clinical use claims).

- The pipeline will be scripted end-to-end so the dataset can be recreated deterministically from a ClinVar release.

---

## 4. Technical Execution & Complexity

**End-to-end pipeline:**

- Download ClinVar release, normalize variant representation, and filter labels.

- Annotate consequences with Ensembl VEP to isolate non-coding variants.

- Extract reference/alternate sequence windows and compute embeddings.

- In parallel, ingest Dylan Tan’s dataset and associated ESM2 embedding features for benchmarking (same evaluation protocol and leakage-aware splitting where applicable).

- Train and compare embedding-based classifiers; save models, splits, and metrics.

**Models (prediction-focused):**

- Logistic Regression baseline

- Random Forest baseline

- Shallow MLP (optional improvement)

**What makes this technically complex (Master’s-level):**

- Applies pretrained DNA foundation-model embeddings to non-coding variant classification (transfer learning).

- Uses leakage-aware splitting (chromosome holdout) and paired statistical tests to compare models.

- Delivers an end-to-end, reproducible pipeline plus a user-facing inference app.

**Reproducibility plan:**
- Fixed random seeds, configuration-driven runs, and saved train/val/test splits.

- Environment capture via conda (and optional Docker).

- Compute environment: UF HiPerGator via OnDemand Jupyter for embedding generation (as needed), training, and evaluation; datasets/features stored under a documented project directory (paths recorded in configs).

**Coding standards (screening rubric):**
- Typed, modular Python package structure (data/feature/model/eval/app).

- Formatting/linting (e.g., `black` + `ruff`) and a small set of unit tests (e.g., `pytest`) for key pipeline steps.


---

## 5. Deployment Plan: “The App”

**User action:** score a user-specified variant and return a calibrated pathogenic probability.

**Interfaces:**
- Streamlit app: accepts a single variant (`chrom`, `pos`, `ref`, `alt`, `assembly`) or a small CSV; returns probability + predicted class at a documented threshold.

- CLI: batch scoring for a CSV/VCF-derived table.

**Deliverables (end-to-end execution):**
- Reproducible pipeline (scripts/configs) that rebuilds the dataset from ClinVar.

- Trained model artifact(s) + evaluation report with plots and statistical comparisons.

- Streamlit app + CLI for scoring.

---

## 6. Statistical Analysis & Evaluation

**Evaluation Metrics (success criteria):**

- Primary (imbalance-aware): AUROC and AUPRC

- Secondary: precision, recall, F1, balanced accuracy

**Experimental design (avoid leakage):**
- Primary split: chromosome-based holdout (train/val/test separated by chromosomes) to reduce local-sequence leakage.

- Sensitivity analysis: alternate chromosome partitions to test stability.

**Baselines and comparisons:**
- Logistic Regression vs. Random Forest vs. (optional) shallow MLP, all trained on the same embedding features.

**Statistical tests / validation (not due to chance):**
- Bootstrapped confidence intervals for AUROC and AUPRC on the held-out test set.

- Paired model comparison:

  - DeLong test for AUROC (paired predictions)

  - Paired permutation test or paired bootstrap test for AUPRC differences

**Handling class imbalance:**

- Use class weights and threshold tuning driven by the validation set.

- Report PR curves and emphasize AUPRC as the key metric.


**Calibration (probabilities you can interpret):**

- Calibrate final model probabilities using the validation set (e.g., Platt scaling or isotonic regression) and report calibration metrics (e.g., Brier score).

**Interpretability (supporting evidence):**

- Model-side: feature importance (RF) and calibrated probability outputs.

- Sequence-side (lightweight): attribution over the sequence window for a small set of representative variants; optional motif overlap check for plausibility.

- Optional (time permitting): SHAP-style analysis on embedding features to provide global/local explanations for representative predictions.

**Contingency plan (iterative refinement):**

- If positive class size is too small after filtering, expand to include "Likely pathogenic" and/or relax to a broader non-coding consequence set while keeping labels high-confidence (still excluding VUS/conflicting).

---

## 7. Project Timeline & Milestones

- **Weeks 1–4:** Download ClinVar; define filters; consequence annotation (VEP); EDA; finalize window size and chromosome split; confirm ingestion of Dylan Tan dataset + ESM2 features on HiPerGator
- **Weeks 5–8:** Embedding generation; train LR/RF baselines; initial AUROC/AUPRC; iterate on imbalance handling
- **Weeks 9–12:** Add MLP (optional); finalize evaluation; bootstrap CIs + paired tests; error analysis
- **Weeks 13–15:** Streamlit app + CLI; documentation; final report writing; presentation/demo prep

**Collaboration (Individual Project):** This is an individual capstone, with planned feedback cycles via instructor/advisor check-ins and peer review of the demo/report.

---

## 8. New Knowledge Acquisition

- Practical use of pretrained DNA language models for embedding extraction in a reproducible pipeline (DNABERT-style)

- Integrating externally provided embedding feature sets (ESM2) into a unified evaluation pipeline

- Variant consequence annotation workflow (Ensembl VEP) to operationalize “non-coding” filtering

---

## References

[1] Y. Ji, Z. Zhou, H. Liu, and R. V. Davuluri, "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome," Bioinformatics, vol. 37, no. 15, pp. 2112–2120, 2021.

[2] M. J. Landrum et al., "ClinVar: improving access to variant interpretations and supporting evidence," Nucleic Acids Res., vol. 46, no. D1, pp. D1062–D1067, 2018.

[3] W. McLaren et al., "The Ensembl Variant Effect Predictor," Genome Biol., vol. 17, no. 1, p. 122, 2016.

[4] Z. Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," Science, 2023. (ESM2)

### Data & Tool Resources

- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- Ensembl VEP: https://www.ensembl.org/info/docs/tools/vep/
- scikit-learn: https://scikit-learn.org/
- Streamlit: https://streamlit.io/
