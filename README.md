# Machine Learning Classification of Pathogenic vs. Benign Coding Genetic Variants

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**EGN 6933 – Capstone Project in Applied Data Science**

**Student:** Angel Morenu  
**Semester:** Spring 2026  
**Faculty Advisor:** Dr. Xiao Fan  
**Instructor:** Dr. Edwin Marte Zorrilla  

## Overview

This capstone project develops a machine learning pipeline to classify coding genetic variants as pathogenic versus benign using pretrained protein language model embeddings (ESM2). The work supports precision medicine workflows by enabling computational prioritization of variants for downstream experimental follow-up.

## Project Information

**Objective:** Build a reproducible, prediction-focused ML pipeline for binary classification of coding variants using embedding-style features from protein language models.

**Key Components:**
- **Data:** Public curated coding-variant pathogenicity dataset (Dr. Fan’s group; shared via Dylan Tan)
- **Features:** Precomputed ESM2 protein language model embeddings (with optional feature generation if needed)
- **Models:** Logistic Regression, Random Forest, and optional shallow MLP
- **Evaluation:** AUROC/AUPRC with gene/protein-aware holdout splitting to prevent data leakage
- **Deployment:** Streamlit web application and command-line interface for variant scoring

**Collaboration:** Dylan Tan provides the public curated dataset and guidance/reference code for working with embedding-style feature tables.

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
│   └── build_week2_training_table.py  # Week 2: strict labels + canonical keys -> trainable table
├── src/                               # Core project source code
│   ├── variant_classifier/            # Main package
│   └── variant_embeddings/            # Embedding utilities
├── notebooks/                         # Jupyter notebooks for EDA and prototyping
├── config/                            # Configuration files (hyperparameters, splits)
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
├── environment.yml                    # Conda environment specification
└── LICENSE                           # MIT License
```

**Note:** Data files, trained models, research PDFs, and generated outputs are excluded from version control per `.gitignore`.

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

3. **Obtain the curated dataset:**
   ```bash
   # Data is not included in the repository.
   # Use the provided ingestion scripts once you have the dataset path.
   ```

### Usage

#### Data Ingestion (Curated ESM2 Dataset)
```bash
# Inspect ESM2 PKL schema
python scripts/inspect_esm2_primateai_pkl.py \
    --pkl-path /path/to/esm2_selected_features.pkl

# Ingest ESM2 PKL to Parquet with label policy
python scripts/ingest_esm2_primateai.py \
    --pkl-path /path/to/esm2_selected_features.pkl \
    --output-parquet data/processed/esm2_primateai.parquet \
    --label-policy strict

# Build mapping from Dylan pickle numeric ID (ClinVar VariationID) -> chr_pos_ref_alt (default: GRCh38 SNVs)
python scripts/make_pickle_id_to_chrposrefalt.py --max-ids 100000000
# Outputs:
# - data/processed/pickle_id_to_chrposrefalt.tsv
# - data/processed/pickle_id_to_chrposrefalt_ambiguous.tsv

# Week 2: build a trainable table with canonical keys + strict labels
# (writes TSV + NumPy embeddings)
python scripts/build_week2_training_table.py --max-rows 5000
# Outputs:
# - data/processed/week2_training_table_strict.tsv.gz
# - data/processed/week2_training_table_strict_embeddings.npy
# - data/processed/week2_training_table_strict_meta.json
```

#### Model Training (Coming Soon)
```bash
# Train baseline models
python src/train.py --config configs/baseline.yaml

# Evaluate on test set
python src/evaluate.py --model-path models/best_model.pkl --test-data data/processed/test.parquet
```

#### Variant Scoring (Coming Soon)
```bash
# Streamlit web app
streamlit run app/main.py

# CLI scoring
python -m variant_classifier.score --variant chr17:41234567:A:G --assembly GRCh38
```

## Project Timeline

### Phase 1: Data & Infrastructure (Weeks 1-4) ✅
- ✅ Obtain and validate curated ESM2 feature dataset
- ✅ Define label filtering (strict pathogenic/benign)
- ✅ Ingest curated ESM2 dataset with flexible label policies
- ✅ Validate Parquet outputs (strict variants, 0 NaN embeddings)
- ✅ Set up git repository with clean commit history
- ✅ Confirm canonical ID mapping (pickle numeric ID ⇄ ClinVar VariationID ⇄ chr_pos_ref_alt)
- 🔄 Join curated embeddings to canonical variant keys and lock label mapping/exclusions (target: Week 2)
- 🔄 Design leakage-aware gene/protein-aware split strategy (target: Week 3)


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
- [ ] **Week 4:** Complete gene/protein-aware split design
- [ ] **Week 8:** Baseline models trained and evaluated
- [ ] **Week 12:** Final model selection and statistical validation
- [ ] **Week 15:** Deployment and final presentation

## Technical Stack

**Languages & Frameworks:**
- Python 3.11+ (core language)
- PyTorch or TensorFlow (optional, for MLP)
- scikit-learn (baseline models, metrics)
- pandas, numpy (data manipulation)
- pyarrow (Parquet I/O)

**Bioinformatics Tools:**
- ESM / ESM2 (protein language model embeddings)
- ClinVar API/FTP (optional upstream/validation)
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

- **Public curated coding-variant dataset:** Dr. Fan’s group (shared via Dylan Tan)
- **ESM / ESM2:** https://github.com/facebookresearch/esm
- **ClinVar (optional):** https://www.ncbi.nlm.nih.gov/clinvar/
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

- **Dylan Tan** for providing precomputed ESM2 embedding features and reference code for coding variants
- **Dr. Xiao Fan** for project guidance and access to HiPerGator computational resources
- **ClinVar** and **Ensembl** for providing public genomic variant databases and annotation tools

---

**Repository:** https://github.com/angelmorenu/egn6933-capstone-variant-pathogenicity-esm2  
**Course:** EGN 6933 – Project in Applied Data Science, Spring 2026  
**University of Florida**
