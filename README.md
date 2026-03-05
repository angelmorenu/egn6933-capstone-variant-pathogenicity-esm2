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

This capstone project develops a machine learning pipeline to classify missense variants as pathogenic versus benign using pretrained protein language model embeddings (ESM2). The working dataset is a curated, missense-only subset of ClinVar 20240805 (with strict pathogenic/benign labels), combined with precomputed ESM2 embeddings provided by Dr. Fan's lab (Dylan Tan).

**Canonical Variant Identifier:** Each variant is uniquely identified by a composite key `chr_pos_ref_alt` (chromosome, genomic position, reference allele, alternate allele), which maps 1:1 to Dylan's precomputed ESM2 embedding vectors. This ensures full traceability and reproducibility across all modeling steps.

## Project Information

**Objective:** Build a reproducible, prediction-focused ML pipeline for binary classification of missense variants using embedding-style features from protein language models.

**Key Components:**
- **Data:** ClinVar 20240805 missense-only curated dataset with strict pathogenic/benign labels (sourced directly from ClinVar). Variants are canonically identified via `chr_pos_ref_alt` keys. **All variants in the modeling dataset are mapped to and uniquely identified by Dylan's precomputed ESM2 embeddings** (provided by Dr. Fan's lab).
- **Features:** Precomputed ESM2 protein language model embeddings (with optional feature generation if needed)
- **Models:** Logistic Regression, Random Forest, and optional shallow MLP
- **Evaluation:** Gene/protein-aware holdout splitting to prevent data leakage; the test split is treated as **generalization to unseen genes/proteins** (a biological extrapolation set rather than IID data). **Primary metric: AUROC (per Dr. Fan);** AUPRC is reported for completeness.
- **Deployment:** Streamlit web application and command-line interface for variant scoring

## Features

- ✅ **Transfer Learning:** Leverages pretrained protein language models (ESM2) for feature extraction
   - All variants used in modeling are mapped to and uniquely identified by **Dylan's precomputed ESM2 embeddings**, ensuring full traceability and compliance with project requirements.
- ✅ **Leakage-Aware Evaluation:** Gene/protein-aware train/test splits prevent inflated performance
  - Planned: Homology-aware audit to screen for sequence similarity across splits
- ✅ **Rigorous Statistics:** Bootstrapped confidence intervals, DeLong tests, McNemar exact tests, paired comparisons
- ✅ **Production-Ready:** Reproducible pipeline with cached embeddings, versioned datasets, and deployment interfaces
- ✅ **Class Imbalance Handling:** Class weighting, threshold tuning, AUROC-first evaluation (AUPRC reported as secondary)
- ✅ **Interpretability:** Feature importance, calibrated probabilities, embedding-space visualization (UMAP/t-SNE), attribution analysis

## Repository Structure & Navigation Guide

This section explains the folder and file organization of the repository. Each major folder serves a specific purpose in the project workflow.

### How to Navigate This Repository

1. **Start with this README** for project overview and setup instructions
2. **Review `project-proposal/`** for the formal capstone proposal and project scope
3. **Check `docs/`** for detailed technical documentation, weekly checklists, and progress summaries
4. **Explore `notebooks/`** for exploratory data analysis (EDA) and prototyping work
5. **Use `scripts/`** for executable data processing and model training pipelines
6. **Examine `src/`** for core reusable Python modules and package code
7. **View `results/`** (not in git) for evaluation outputs, plots, and performance reports
8. **Consult `milestones/`** for weekly progress tracking and timeline updates

### Folder Structure & Purpose

```
egn6933-capstone-variant-pathogenicity-esm2/
├── README.md                           # Project overview and documentation
├── LICENSE                             # MIT License
├── .gitignore                          # Git exclusions (data, models, PDFs)
├── config/                             # Configuration files and project decisions
│   ├── decisions.md                   # Design decisions and rationale
│   └── label_maps/                    # ClinVar label mapping configurations
├── data/                               # Local data directory (gitignored; not in remote repo)
│   ├── clinvar/                       # ClinVar downloads (variant_summary.txt.gz, VCF)
│   ├── Dylan Tan/                     # Lab-provided embeddings and curated tables
│   └── processed/                     # Processed datasets (Parquet, NPY, TSV)
├── docs/                               # Detailed technical documentation
│   ├── README.md                      # Documentation index
│   ├── week1_4_checklist.md           # Weeks 1-4 execution checklist
│   ├── week5_8_checklist.md           # Weeks 5-8 execution checklist (✅ COMPLETE)
│   ├── week5_baseline_conclusion.md   # Baseline model selection summary
│   ├── week6_seed_sensitivity.md      # Robustness testing across split seeds
│   ├── week7_hyperparameter_selection.md  # Hyperparameter tuning decisions
│   ├── week8_summary.md               # ✅ Week 8 comprehensive summary 
│   ├── chromosome_split_design.md     # Split strategy design document
│   └── system-architecture.md         # System architecture overview
├── environment.yml                     # Conda environment specification
├── milestones/                         # Project timeline and progress tracking
│   └── progress-tracker.md            # Weekly milestone tracker (UPDATED: Week 8 ✅)
├── notebooks/                          # Jupyter notebooks for exploration
│   └── 01_eda_clinvar.ipynb           # ClinVar exploratory data analysis
├── project-proposal/                   # Formal capstone proposal (APPROVED)
│   ├── Project_Proposal.md            # Full formal proposal (markdown)
│   ├── Morenu_Project_Proposal.docx   # Word version for submission
│   └── *.backup*.docx                 # Backup versions before edits
├── research/                           # Local literature workspace (gitignored; not in remote repo)
│   ├── README.md                      # Research notes index
│   ├── Papers/                        # Research papers (PDFs excluded)
│   └── citations.md                   # Citation tracking
├── requirements.txt                    # Pip requirements (alternative to conda env)
├── results/                            # Model outputs and evaluation (EXCLUDED from git)
│   ├── Week 5/                        # Week 5 baseline results (24 JSON/PNG files)
│   ├── baseline_logreg_metrics.json   # Logistic regression results
│   ├── baseline_rf_report.json        # Random forest results
│   └── *.png                          # Evaluation plots (PR curves, reliability diagrams)
├── scripts/                           # Executable data processing pipelines
│   ├── README.md                      # Scripts usage documentation
│   ├── inspect_esm2_primateai_pkl.py  # ESM2 dataset schema inspection
│   ├── ingest_esm2_primateai.py       # ESM2 to Parquet ingestion with label policies
│   ├── make_pickle_id_to_chrposrefalt.py  # Map pickle numeric ID -> chr_pos_ref_alt via ClinVar
│   ├── build_week2_training_table.py  # Week 2: build trainable TSV+NPY (defaults to cleaned missense_strict labels when present)
│   ├── sanity_check_week2_table.py    # Sanity checks for Week 2 artifacts
│   ├── make_week3_splits.py           # Week 3: leakage-aware (gene-grouped) train/val/test splits
│   ├── make_week4_curated_dataset.py  # Week 4: curated Parquet (label + split + embedding)
│   ├── week4_eda.py                   # Week 4: EDA + go/no-go checks
│   ├── baseline_train_eval.py         # Baseline train/eval (LR/RF) + calibration, bootstrap CIs, plots
│   ├── compare_baselines_stats.py     # Statistical comparison (DeLong, McNemar, bootstrap)
│   ├── plot_compare_baselines_stats.py  # Visualization for statistical comparisons
│   └── build_clinvar_labels_from_vcf.py  # Optional: reproduce label counts from VCF
├── src/                                # Core reusable Python modules
│   ├── variant_classifier/             # Classification pipeline modules
│   └── variant_embeddings/             # Embedding utilities and feature engineering
├── tests/                              # Unit and integration tests
│   └── test_placeholder.py            # Placeholder test file
├── pyproject.toml                      # Python project configuration (PEP 518)
└── .git/                               # Git repository metadata
```

### Important Notes on Excluded Files

The following files and directories are **excluded from version control** (see `.gitignore`):
- **`data/`** - Local-only raw and processed data files (large binary files, embeddings, Parquet datasets; not in remote repo)
- **`results/`** - All model evaluation outputs, plots, and performance reports
- **`research/`** - Local literature PDFs and personal research notes (not in remote repo)
- **Binary artifacts:** `.pkl`, `.parquet`, `.npy`, `.pt`, `.pth`, `.h5` (machine learning artifacts)
- **Documents:** `.pdf`, `.docx` (generated documents; keep only markdown sources)
- **System files:** `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`

This keeps the repository clean and focused on source code, scripts, and documentation while avoiding large binary files.

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
   # - data/Dylan Tan/esm2_selected_features.pkl (Dylan's precomputed ESM2 embeddings from Dr. Fan's lab)
   # - data/clinvar/variant_summary.txt.gz (ClinVar download for variant mapping)
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

Week 6 robustness in practice (two split seeds + bootstrapped CIs):

```bash
# Seed 13 splits (gene-disjoint)
conda run -n egn6933-variant-embeddings python scripts/make_week3_splits.py \
   --seed 13 \
   --out-prefix data/processed/week2_training_table_strict

# Build curated dataset and archive to seed-specific filename
conda run -n egn6933-variant-embeddings python scripts/make_week4_curated_dataset.py
cp -f data/processed/week4_curated_dataset.parquet data/processed/week4_curated_dataset_seed13.parquet
cp -f data/processed/week4_curated_dataset_meta.json data/processed/week4_curated_dataset_seed13_meta.json

# RF baseline + bootstrap CIs on test
conda run -n egn6933-variant-embeddings python scripts/baseline_train_eval.py \
   --model rf \
   --rf-max-depth 4 \
   --rf-n-estimators 200 \
   --data data/processed/week4_curated_dataset_seed13.parquet \
   --bootstrap-iters 1000 \
   --out-json results/baseline_rf_seed13_bootstrap.json

# Seed 37 (repeat)
conda run -n egn6933-variant-embeddings python scripts/make_week3_splits.py \
   --seed 37 \
   --out-prefix data/processed/week2_training_table_strict

conda run -n egn6933-variant-embeddings python scripts/make_week4_curated_dataset.py
cp -f data/processed/week4_curated_dataset.parquet data/processed/week4_curated_dataset_seed37.parquet
cp -f data/processed/week4_curated_dataset_meta.json data/processed/week4_curated_dataset_seed37_meta.json

conda run -n egn6933-variant-embeddings python scripts/baseline_train_eval.py \
   --model rf \
   --rf-max-depth 4 \
   --rf-n-estimators 200 \
   --data data/processed/week4_curated_dataset_seed37.parquet \
   --bootstrap-iters 1000 \
   --out-json results/baseline_rf_seed37_bootstrap.json

# Summary note
# - docs/week6_seed_sensitivity.md
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
   --select-metric auroc \
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

### Phase 2: Feature Engineering & Baselines (Weeks 5-8) ✅
- ✅ Finalize feature set (ESM2 embedding dimensions and QC)
- ✅ Train Logistic Regression baseline
- ✅ Train Random Forest baseline
- ✅ Initial AUROC/AUPRC evaluation
- ✅ Seed robustness testing and bootstrap confidence intervals

### Phase 3: Refinement & Evaluation (Weeks 9-12)
- [ ] Implement optional MLP
- [ ] Bootstrapped confidence intervals and paired statistical tests (DeLong, McNemar)
- [ ] **Homology-aware leakage audit:** Screen for high sequence similarity across train/val/test splits; adjust grouping if strong homology detected
- [ ] **Embedding-space visualization:** Generate UMAP/t-SNE plot (colored by pathogenic/benign label and split assignment) as a stakeholder-facing diagnostic
- [ ] Error analysis and interpretability

### Phase 4: Deployment & Documentation (Weeks 13-15)
- [ ] Streamlit web application
- [ ] Command-line interface
- [ ] Final report and documentation
- [ ] Project presentation

## Key Milestones

- [x] **Week 1**
   - ESM2 ingestion pipeline implemented with label policies
   - Proposal updated to coding-variant scope
   - GitHub repository initialized and pushed
   - Confirmed pickle numeric ID matches ClinVar VariationID; generated chr_pos_ref_alt mapping
- [x] **Week 2**
   - Proposal updated to missense-only scope and ClinVar-primary framing
   - Week 2 5k training artifacts generated (TSV + embeddings + meta) using cleaned missense_strict labels by default
   - Added sanity checks for Week 2 artifacts (alignment/duplicates/balance)
- [x] **Week 3:** Implemented leakage-aware (gene-grouped) train/val/test splits and wrote split artifacts (Parquet + index files + meta)
- [x] **Week 4:** Finalized curated dataset artifact + Week 4 EDA/go-no-go checks
- [x] **Week 5:** Trained/evaluated baseline models (LogReg, shallow RF) + optional Platt calibration; saved reports/plots under `results/`
- [x] **Week 6:** Seed robustness (seed 13 vs 37) + bootstrap confidence intervals on test set
- [x] **Week 7:** Hyperparameter selection (LogReg C-sweep, RF grid) + class imbalance handling review
- [x] **Week 8:** Threshold selection + statistical testing prep
- [ ] **Week 12:** Final model selection and statistical validation
- [ ] **Week 15:** Deployment and final presentation

## Technical Stack

**Languages & Frameworks:**
- Python 3.11+ (core language; tested locally with Python 3.12)
- PyTorch or TensorFlow (optional, for MLP)
- scikit-learn 1.4.2 (baseline models, metrics)
- scipy 1.13.1 (scientific computing)
- pandas 2.3.3, NumPy 1.26.4 (data manipulation)
- pyarrow 15.0.2 (Parquet I/O)

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
- **ClinVar:** https://www.ncbi.nlm.nih.gov/clinvar/ (primary source for missense variant labels and metadata)
- **Precomputed ESM2 Embeddings:** Provided by Dr. Fan's lab (Dylan Tan)
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

## Acknowledgments

- **Dylan Tan** for providing the cleaned missense dataset and aligned precomputed embedding features
- **Dr. Xiao Fan** for project guidance and access to Lab via  HiPerGator computational resources
- **Dr. Edwin Marte Zorrilla** for instruction and guidance throughout the capstone project
- **ClinVar** and **Ensembl** for providing public genomic variant databases and annotation tools

---

**Repository:** https://github.com/angelmorenu/egn6933-capstone-variant-pathogenicity-esm2  

**Course:** EGN 6933 – Project in Applied Data Science, Spring 2026  
**University of Florida**
