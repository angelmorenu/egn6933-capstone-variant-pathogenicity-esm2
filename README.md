# Machine Learning Classification of Pathogenic vs. Benign Missense Variants

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**EGN 6933 – Capstone Project in Applied Data Science**

**Student:** Angel Morenu  
**Semester:** Spring 2026  
**Faculty Advisor:** Dr. Xiao Fan  
**Instructor:** Dr. Edwin Marte Zorrilla  

## Overview

This capstone project develops a machine learning pipeline to classify missense variants as pathogenic versus benign using pretrained protein language model embeddings (ESM2). The working dataset is a post-quality-control, missense-only ClinVar-derived table (ClinVar 20240805) provided by Dr. Fan’s lab (Dylan Tan), plus aligned ESM2 embeddings.

## Project Information

**Objective:** Build a reproducible, prediction-focused ML pipeline for binary classification of missense variants using embedding-style features from protein language models.

**Key Components:**
- **Data:** ClinVar-derived missense-only dataset (post-QC) provided by Dr. Fan/Dylan; ClinVar downloads are used for reference/mapping
- **Features:** Precomputed ESM2 protein language model embeddings (with optional feature generation if needed)
- **Models:** Logistic Regression, Random Forest, and optional shallow MLP
- **Evaluation:** AUROC/AUPRC with gene/protein-aware holdout splitting to prevent data leakage; the test split is treated as **generalization to unseen genes/proteins** (a biological extrapolation set rather than IID data)
- **Deployment:** Streamlit web application and command-line interface for variant scoring

## Features

- ✅ **Transfer Learning:** Leverages pretrained protein language models (ESM2) for feature extraction
- ✅ **Leakage-Aware Evaluation:** Gene/protein-aware train/test splits prevent inflated performance
- ✅ **Rigorous Statistics:** Bootstrapped confidence intervals, DeLong tests, paired comparisons
- ✅ **Production-Ready:** Reproducible pipeline with cached embeddings, versioned datasets, and deployment interfaces
- ✅ **Class Imbalance Handling:** Class weighting, threshold tuning, AUPRC-focused evaluation
- ✅ **Interpretability:** Feature importance, calibrated probabilities, attribution analysis

## Repository Structure

```
egn6933-capstone-variant-pathogenicity-esm2/
├── README.md                           # Project overview and documentation
├── project-proposal/                   # Formal capstone proposal documents
│   ├── Project_Proposal.md            # Full formal proposal (prose, inline citations)
├── research/                           # Literature review and references
│   └── papers/                        # Research papers (excluded from git)
├── scripts/                           # Data processing and utility scripts
│   ├── inspect_esm2_primateai_pkl.py  # ESM2 dataset schema inspection
│   ├── ingest_esm2_primateai.py       # ESM2 to Parquet ingestion with label policies
│   ├── make_pickle_id_to_chrposrefalt.py  # Map pickle numeric ID -> chr_pos_ref_alt via ClinVar
│   ├── build_week2_training_table.py  # Week 2: build trainable TSV+NPY (defaults to cleaned missense_strict labels when present)
│   ├── sanity_check_week2_table.py    # Sanity checks for Week 2 artifacts
│   ├── make_week3_splits.py           # Week 3: leakage-aware (gene-grouped) train/val/test splits
│   ├── make_week4_curated_dataset.py  # Week 4: curated Parquet (label + split + embedding)
│   ├── week4_eda.py                   # Week 4: EDA + go/no-go checks
│   ├── baseline_train_eval.py          # Baseline train/eval (LR/RF) + calibration, bootstrap CIs, plots
│   └── build_clinvar_labels_from_vcf.py  # Optional: reproduce label counts from a ClinVar VCF
├── src/                               # Core project source code
│   ├── variant_classifier/            # Main package
│   └── variant_embeddings/            # Embedding utilities
├── notebooks/                         # Jupyter notebooks for EDA and prototyping
├── config/                            # Configuration files (hyperparameters, splits, decisions)
├── tests/                            # Unit and integration tests
├── data/                             # Data directory (excluded from git)
│   ├── raw/                          # Original downloads (if applicable)
│   ├── processed/                    # Cleaned, labeled datasets
│   └── embeddings/                   # Cached embedding features
├── models/                           # Trained model artifacts (excluded from git)
├── results/                          # Evaluation outputs, plots, reports
├── app/                              # Streamlit web application
├── docs/                             # Additional documentation
├── .gitignore                        # Git exclusions (data, models, PDFs)
├── pyproject.toml                    # Python project configuration
├── environment.yml                   # Conda environment specification
└── LICENSE                           # MIT License
```

**Note:** Data files (e.g., Dylan’s large `.pkl`/`.txt`, `data/processed/*.npy`), trained models, and generated outputs (including `results/`) are excluded from version control per `.gitignore`.

## Getting Started

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/angelmorenu/egn6933-capstone-variant-pathogenicity-esm2.git
   cd egn6933-capstone-variant-pathogenicity-esm2
   ```

2. **Create the environment:**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate egn6933-variant-embeddings
   
   # Or using pip
   pip install -e .
   ```

3. **Download data:**
   ```bash
   # Data is not included in the repository.
   # Required for Week 2 build (local files):
   # - data/Dylan Tan/esm2_selected_features.pkl
   # - data/Dylan Tan/clinvar_20240805.missense_strict_updated.txt (or ...missense_strict.txt)
   # - data/clinvar/variant_summary.txt.gz
   ```

### Usage

#### Data Ingestion (Prototype Embedding Dataset)
```bash
# Inspect ESM2 PKL schema
python scripts/inspect_esm2_primateai_pkl.py \
    --pkl-path /path/to/esm2_selected_features.pkl

# Ingest ESM2 PKL to Parquet with label policy
python scripts/ingest_esm2_primateai.py \
    --pkl-path /path/to/esm2_selected_features.pkl \
    --output-parquet data/processed/esm2_primateai.parquet \
    --label-policy strict

# Build mapping from pickle numeric ID (ClinVar VariationID) -> chr_pos_ref_alt (default: GRCh38 SNVs)
python scripts/make_pickle_id_to_chrposrefalt.py --max-ids 100000000
# Outputs:
# - data/processed/pickle_id_to_chrposrefalt.tsv
# - data/processed/pickle_id_to_chrposrefalt_ambiguous.tsv

# Week 2: build a trainable table (writes TSV + NumPy embeddings + meta)
# Default behavior: if Dylan's cleaned missense_strict table exists under data/Dylan Tan/, it is used as the label source.
conda run -n egn6933-variant-embeddings python scripts/build_week2_training_table.py --max-rows 5000
# Outputs:
# - data/processed/week2_training_table_strict.tsv.gz
# - data/processed/week2_training_table_strict_embeddings.npy
# - data/processed/week2_training_table_strict_meta.json

# Sanity-check the artifacts (alignment, duplicates, label balance)
conda run -n egn6933-variant-embeddings python scripts/sanity_check_week2_table.py \
   --prefix data/processed/week2_training_table_strict

# Week 3: leakage-aware gene/protein-aware splits (all variants from the same gene stay in one split)
# Writes a Parquet with a `split` column + split index files.
conda run -n egn6933-variant-embeddings python scripts/make_week3_splits.py \
   --input-tsv data/processed/week2_training_table_strict.tsv.gz \
   --clinvar-variant-summary data/clinvar/variant_summary.txt.gz \
   --out-prefix data/processed/week2_training_table_strict

```

**Splitting rationale (gene-disjoint + prevalence-aware):** All variants from the same mapped gene are kept in a single split to reduce information leakage. For small pilots with few genes and highly skewed per-gene label rates, a purely size-based greedy assignment can produce large train/val/test prevalence differences (which can distort AUPRC and threshold selection). For the 5k pilot, the recommended setting is the prevalence-aware local-search method:

```bash
conda run -n egn6933-variant-embeddings python scripts/make_week3_splits.py \
   --method search \
   --min-groups-per-split 5 \
   --min-split-rows 400 \
   --search-iters 20000 \
   --out-prefix data/processed/week2_training_table_strict
```

To sanity-check robustness to the split seed (still gene-disjoint), rerun with a different `--seed` and rebuild Week 4:

```bash
conda run -n egn6933-variant-embeddings python scripts/make_week3_splits.py \
   --method search \
   --min-groups-per-split 5 \
   --min-split-rows 400 \
   --search-iters 20000 \
   --seed 37 \
   --out-prefix data/processed/week2_training_table_strict

conda run -n egn6933-variant-embeddings python scripts/make_week4_curated_dataset.py
conda run -n egn6933-variant-embeddings python scripts/week4_eda.py
```

```bash
# Week 4: build a single curated dataset Parquet (label + split + embedding vectors)
conda run -n egn6933-variant-embeddings python scripts/make_week4_curated_dataset.py
# Outputs:
# - data/processed/week4_curated_dataset.parquet
# - data/processed/week4_curated_dataset_meta.json

# Week 4: EDA + go/no-go checks (writes tables/plot + a JSON report)
conda run -n egn6933-variant-embeddings python scripts/week4_eda.py
# Outputs:
# - data/processed/week4_eda/counts_by_split.tsv
# - data/processed/week4_eda/unique_genes_by_split.tsv
# - data/processed/week4_eda/positive_rate_by_split.png
# - data/processed/week4_eda/go_no_go.json
```

#### Model Training (Baselines)
```bash
# Minimal baseline (Logistic Regression) using the curated dataset Parquet.
python scripts/baseline_train_eval.py \
   --data data/processed/week4_curated_dataset.parquet \
   --out-json results/baseline_logreg_metrics.json

# Controlled nonlinear baseline: shallow Random Forest (gene-disjoint evaluation).
python scripts/baseline_train_eval.py \
   --model rf \
   --rf-max-depth 4 \
   --rf-n-estimators 200 \
   --data data/processed/week4_curated_dataset.parquet \
   --out-json results/baseline_rf_report.json

# Recommended: C sweep (L2 strength), optional calibration on val, bootstrap CIs on test, and plots.
python scripts/baseline_train_eval.py \
   --data data/processed/week4_curated_dataset.parquet \
   --c-grid 0.01,0.1,1,10,100 \
   --select-metric auprc \
   --calibration platt \
   --bootstrap-iters 1000 \
   --plot-pr results/pr_curves_val_vs_test.png \
   --plot-reliability results/reliability_test.png \
   --plot-scores-test results/test_score_distributions.png \
   --out-json results/baseline_logreg_report.json
```

#### Variant Scoring (Coming Soon)
```bash
# Scoring interfaces (Streamlit + CLI) will be added later in the semester.
# (TBD)
```

## Project Timeline

### Phase 1: Data & Infrastructure (Weeks 1-4) ✅
- ✅ Define ClinVar-based data plan (missense-only scope)
- ✅ Define label filtering (strict pathogenic/benign)
- ✅ Prototype ingestion and label-policy tooling
- ✅ Validate Parquet outputs (strict variants, 0 NaN embeddings)
- ✅ Set up git repository with clean commit history
- ✅ Confirm canonical ID mapping (pickle numeric ID ⇄ ClinVar VariationID ⇄ chr_pos_ref_alt)
- ✅ Build missense-only training table and lock label mapping/exclusions (Week 2)
- ✅ Design leakage-aware gene/protein-aware split strategy (target: Week 3)
- ✅ Finalize the curated dataset artifact (target: Week 4)
  - Parquet with `label` (0/1), `split` (train/val/test), and embedding vectors
- ✅ Produce core EDA plots/tables
  - Class balance overall + by split
  - Distribution by gene/protein (if available)
  - Embedding dimensionality checks and summary statistics
  - Missense consequence QC summary (e.g., retained fraction after filtering)
- ✅ Write down “go/no-go” checks before model training
  - Minimum positive class size
  - No leakage across gene/protein splits
  - No duplicate variants across splits
  - Embeddings present and consistent for all retained samples

### Phase 2: Feature Engineering & Baselines (Weeks 5-8)
- [ ] Finalize feature set (ESM2 embedding dimensions and QC)
- [ ] Implement optional missing-feature generation and caching
- [ ] Train Logistic Regression baseline
- [ ] Train Random Forest baseline
- [ ] Initial AUROC/AUPRC evaluation

### Phase 3: Refinement & Evaluation (Weeks 9-12)
- [ ] Implement optional MLP
- [ ] Bootstrapped confidence intervals
- [ ] Paired statistical tests (DeLong, permutation)
- [ ] Error analysis and interpretability

### Phase 4: Deployment & Documentation (Weeks 13-15)
- [ ] Streamlit web application
- [ ] Command-line interface
- [ ] Final report and documentation
- [ ] Project presentation

## Key Milestones

- [x] **Jan 14, 2026:** ESM2 ingestion pipeline implemented with label policies
- [x] **Jan 15, 2026:** Proposal updated to coding-variant scope
- [x] **Jan 15, 2026:** GitHub repository initialized and pushed
- [x] **Jan 16, 2026:** Confirmed pickle numeric ID matches ClinVar VariationID; generated chr_pos_ref_alt mapping
- [x] **Jan 20, 2026:** Proposal updated to missense-only scope and ClinVar-primary framing
- [x] **Jan 20, 2026:** Week 2 5k training artifacts generated (TSV + embeddings + meta) using cleaned missense_strict labels by default
- [x] **Jan 20, 2026:** Added sanity checks for Week 2 artifacts (alignment/duplicates/balance)
- [x] **Jan 26, 2026:** Implemented leakage-aware (gene-grouped) train/val/test splits and wrote split artifacts (Parquet + index files + meta)
- [x] **Week 4:** Finalized curated dataset artifact + Week 4 EDA/go-no-go checks
- [ ] **Week 8:** Baseline models trained and evaluated
- [ ] **Week 12:** Final model selection and statistical validation
- [ ] **Week 15:** Deployment and final presentation

## Technical Stack

**Languages & Frameworks:**
- Python 3.11.14 (core language)
- PyTorch or TensorFlow (optional, for MLP)
- scikit-learn (baseline models, metrics)
- pandas 2.3.3, NumPy 1.26.4 (data manipulation)
- pyarrow 22.0.0 (Parquet I/O)

**Bioinformatics Tools:**
- ESM / ESM2 (protein language model embeddings)
- ClinVar API/FTP
- Ensembl VEP (optional consequence standardization)

**Deployment:**
- Streamlit (web application)
- Click or argparse (CLI)
- Docker (containerization, optional)

**Development:**
- Git/GitHub (version control)
- Conda (environment management)
- Black, Ruff (code formatting/linting)
- pytest (testing)

## Data Sources

- **ESM / ESM2:** https://github.com/facebookresearch/esm
- **ClinVar:** https://www.ncbi.nlm.nih.gov/clinvar/
- **ClinVar-derived cleaned missense dataset (post-QC):** Provided by Dr. Fan’s lab (Dylan Tan), derived from ClinVar 20240805
- **Ensembl VEP (optional):** https://www.ensembl.org/info/docs/tools/vep/

## References

- **Landrum et al. (2018).** "ClinVar: improving access to variant interpretations and supporting evidence." *Nucleic Acids Research*, 46(D1), D1062–D1067.
- **McLaren et al. (2016).** "The Ensembl Variant Effect Predictor." *Genome Biology*, 17(1), 122.

## Contributing

This is an individual capstone project. External contributions are not accepted, but feedback and suggestions are welcome via issues or email.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Angel Morenu**  
M.S. Applied Data Science  
University of Florida  
Email: angel.morenu@ufl.edu

**Faculty Advisor:** Dr. Xiao Fan  
**Course Instructor:** Dr. Edwin Marte Zorrilla

## Acknowledgments

- **Dylan Tan** for providing the cleaned missense dataset and aligned precomputed embedding features
- **Dr. Xiao Fan** for project guidance and access to HiPerGator computational resources
- **ClinVar** and **Ensembl** for providing public genomic variant databases and annotation tools

---

**Repository:** https://github.com/angelmorenu/egn6933-capstone-variant-pathogenicity-esm2  
**Course:** EGN 6933 – Project in Applied Data Science, Spring 2026  
**University of Florida**
