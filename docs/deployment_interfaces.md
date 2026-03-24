# Deployment Interfaces: Streamlit Web Application & CLI

## Overview

This document describes two production-ready deployment interfaces for the variant pathogenicity prediction model:

1. **Streamlit Web Application** (`app.py`) – Interactive web interface for single-variant and batch scoring
2. **Command-Line Interface** (`scripts/score_variants.py`) – Automation-friendly batch processing tool

Both interfaces use the same trained Random Forest model and precomputed ESM2 embeddings, ensuring consistency with the Final Report metrics.

---

## Part 1: Streamlit Web Application

### Features

#### 1.1 Single-Variant Scoring
- **Interactive form** accepting canonical variant identifiers (`chr_pos_ref_alt`)
- **Alternative input modes** for manual chromosome/position/allele entry
- **Adjustable confidence threshold** for prediction classification
- **Real-time prediction output** with:
  - Binary classification (PATHOGENIC / BENIGN)
  - Calibrated pathogenicity probability
  - Confidence score (max ensemble probability)
  - Visual color-coded results (red for pathogenic, green for benign)

#### 1.2 Batch Scoring Interface
- **CSV file upload** with support for multiple variant formats
- **Progress indicator** for processing status
- **Ranked results table** sorted by predicted pathogenicity score (descending)
- **Downloadable CSV export** with stable column names:
  - `rank`: Position in ranked list (1-indexed)
  - `variant`: Canonical variant identifier
  - `pathogenicity_score`: Probability [0, 1]
  - `prediction`: PATHOGENIC or BENIGN
  - `confidence`: Model confidence

#### 1.3 Model Performance Dashboard
- **Key metrics display**:
  - Test AUROC: 0.9299 [0.9145, 0.9453]
  - Test AUPRC: 0.9473 [0.9328, 0.9618]
  - F1 Score: 0.8761
  - Error Rate: 14.6%

- **Performance visualizations**:
  - **ROC Curve**: Interactive Plotly graph with AUROC annotation
  - **Precision-Recall Curve**: Shows model trade-off between sensitivity/specificity
  - **Confusion Matrix**: Heatmap of TP/FP/FN/TN
  - **Per-Gene Error Rates**: Bar chart showing error distribution across genes

- **Statistical test results**:
  - DeLong AUROC comparison (RF vs. XGBoost)
  - Bootstrap confidence intervals
  - Non-significance statement (p = 0.5523)

#### 1.4 Explainability View
- **Prediction breakdown**:
  - Model consensus probability
  - Prediction margin (distance from threshold)
  - Ensemble agreement confidence
  
- **Feature importance hints**:
  - Top 10 influential ESM2 embedding dimensions
  - Relative importance scores (bar chart)
  - Interpretation: dimensions with highest variance/discriminative power

- **Training example analogs** (future extension):
  - Similar variants from training set (nearest neighbors in embedding space)
  - Their pathogenic classifications for comparison

#### 1.5 Additional Features
- **Variant history**: Recent predictions stored in session state for quick reference
- **Sidebar navigation**: Quick access to all sections
- **Model information card**: Summary of training data, architecture, and performance
- **Responsive design**: Works on desktop, tablet, and mobile browsers

### Architecture

```
app.py (streamlit application)
├── SESSION STATE & CONFIGURATION
│   ├── Streamlit page config (layout, title, icons)
│   ├── Custom CSS styling (metric cards, color-coded predictions)
│   └── Session state initialization (variant_history, model_loaded, rf_model, embeddings_data, metadata)
│
├── DATA LOADING & CACHING
│   ├── load_model_and_data() [cached]
│   │   ├── Load embeddings from week2_training_table_strict_embeddings.npy
│   │   ├── Load metadata from week2_training_table_strict_meta.json
│   │   └── Return (rf_model, embeddings, metadata) tuple
│   │
│   ├── load_performance_metrics() [cached]
│   │   └── Load error_analysis_report.json from results directory
│   │
│   └── create_dummy_model()
│       └── Demo Random Forest for testing
│
├── MAIN INTERFACE
│   ├── main() – Entry point with navigation sidebar
│   │   ├── Route to single_variant_section()
│   │   ├── Route to batch_upload_section()
│   │   ├── Route to performance_dashboard_section()
│   │   └── Route to about_section()
│   │
│   ├── single_variant_section()
│   │   ├── Left column: Interactive form (canonical ID or manual entry, threshold)
│   │   ├── Right column: Prediction results (PATHOGENIC/BENIGN with probability)
│   │   ├── Prediction breakdown (consensus, confidence metrics)
│   │   ├── Feature importance visualization (Plotly bar chart)
│   │   └── Variant history table
│   │
│   ├── batch_upload_section()
│   │   ├── CSV file uploader widget
│   │   ├── Data preview (first 10 rows)
│   │   ├── Batch scoring button with spinner
│   │   ├── Ranked results table (sorted by pathogenicity)
│   │   └── CSV download button
│   │
│   ├── performance_dashboard_section()
│   │   ├── Key metrics cards (AUROC, AUPRC, F1, Error Rate)
│   │   ├── ROC/PR curve visualizations (Plotly)
│   │   ├── Confusion matrix heatmap
│   │   ├── Per-gene error rate bar chart
│   │   └── Statistical test results (DeLong p-value, CIs)
│   │
│   └── about_section()
│       ├── Project overview and features
│       ├── Dataset information summary
│       └── Contact & references
│
└── __main__
    └── Run Streamlit application
```

### Usage

```bash
# Install dependencies
pip install streamlit scikit-learn plotly pandas numpy

# Run the application (default: http://localhost:8501)
streamlit run app.py

# Run on custom host/port
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

### Data Requirements

The application expects the following data files in the project directory:

```
data/processed/
├── week2_training_table_strict_embeddings.npy      (5000 × 1280 float32 array)
├── week2_training_table_strict_meta.json           (metadata with variant index)
└── ...

results/
├── error_analysis_report.json                      (performance metrics)
└── ...

models/
└── rf_model.pkl  (optional: trained Random Forest; uses placeholder if absent)
```

### Production Deployment

#### Deploy to Streamlit Cloud
```bash
# Push to GitHub, then:
# 1. Visit https://share.streamlit.io
# 2. Connect your GitHub repo
# 3. Select branch and app.py
# 4. Deploy
```

#### Deploy to Docker Container
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

#### Deploy to Heroku / AWS / Google Cloud
```bash
# See documentation in docs/deployment.md
```

---

## Part 2: Command-Line Interface

### Features

#### 2.1 Single-Variant Scoring
```bash
python scripts/score_variants.py --variant chr1_100000_A_G
```

**Output:**
```
[INFO] Loading model and embeddings...
[INFO] Loaded embeddings: shape=(5000, 1280)
[INFO] Loaded metadata: 5000 variants indexed
[INFO] Scoring single variant: chr1_100000_A_G

[RESULT]
  Variant: chr1_100000_A_G
  Prediction: PATHOGENIC
  Pathogenicity Score: 0.8756
  Confidence: 0.9234
```

#### 2.2 Batch Scoring
```bash
python scripts/score_variants.py --input variants.csv --output results.csv --format csv
```

**Input CSV Format:**
```csv
variant_id
chr1_100000_A_G
chr2_150000_T_C
chr3_200000_G_A
...
```

**Output CSV:**
```csv
variant_id,pathogenicity_score,prediction,confidence,model,timestamp
chr1_100000_A_G,0.8756,PATHOGENIC,0.9234,RandomForest,2026-03-20T15:30:45.123456
chr2_150000_T_C,0.3421,BENIGN,0.8765,RandomForest,2026-03-20T15:30:46.234567
chr3_200000_G_A,0.7654,PATHOGENIC,0.9012,RandomForest,2026-03-20T15:30:47.345678
```

#### 2.3 JSON Output with Metadata
```bash
python scripts/score_variants.py --input variants.csv --output results.json --format json
```

**Output JSON:**
```json
{
  "metadata": {
    "timestamp": "2026-03-20T15:30:45.123456",
    "model": "RandomForest",
    "n_variants_scored": 3,
    "n_pathogenic": 2,
    "n_benign": 1
  },
  "results": [
    {
      "variant_id": "chr1_100000_A_G",
      "pathogenicity_score": 0.8756,
      "prediction": "PATHOGENIC",
      "confidence": 0.9234,
      "model": "RandomForest",
      "timestamp": "2026-03-20T15:30:45.123456"
    },
    ...
  ]
}
```

#### 2.4 Adjustable Threshold
```bash
# Score with 60% confidence threshold (default 50%)
python scripts/score_variants.py --input variants.csv --output results.csv --threshold 0.6
```

#### 2.5 Error Handling
- Invalid variant ID format → Reports error with expected format
- Missing embedding → Lists unavailable variants with reasons
- Missing file → FileNotFoundError with path details
- Invalid threshold → Constraint checking (0.0 ≤ threshold ≤ 1.0)

### Architecture

```
scripts/score_variants.py (command-line interface)
├── CONFIGURATION & PATHS
│   ├── PROJECT_ROOT – project directory
│   ├── DATA_DIR – embeddings and metadata location
│   ├── MODELS_DIR – trained model storage
│   └── RESULTS_DIR – evaluation metrics
│
├── DATA STRUCTURES
│   └── VariantScore (dataclass)
│       ├── variant_id: str
│       ├── pathogenicity_score: float
│       ├── prediction: str
│       ├── confidence: float
│       ├── model: str
│       └── timestamp: str
│
├── MODEL & DATA LOADING
│   ├── load_embeddings_and_metadata()
│   │   ├── Load week2_training_table_strict_embeddings.npy
│   │   ├── Load week2_training_table_strict_meta.json
│   │   └── Return (embeddings, metadata) tuple
│   │
│   └── load_trained_model()
│       └── Load pickled RandomForestClassifier or create placeholder
│
├── VARIANT SCORING
│   ├── parse_variant_identifier()
│   │   ├── Validate canonical format (chr_pos_ref_alt)
│   │   ├── Check chromosome valid (chr1-22, chrX, chrY, chrMT)
│   │   └── Validate alleles (ACGT only)
│   │
│   ├── lookup_embedding()
│   │   └── Map variant_id to embedding index
│   │
│   ├── score_single_variant()
│   │   ├── Validate variant ID
│   │   ├── Lookup embedding
│   │   ├── Predict with threshold
│   │   └── Return VariantScore
│   │
│   └── score_batch()
│       ├── Filter variants with available embeddings
│       ├── Batch predict (vectorized)
│       ├── Apply threshold classification
│       ├── Report failures
│       └── Return sorted scores (descending pathogenicity)
│
├── INPUT/OUTPUT HANDLING
│   ├── read_input_file()
│   │   ├── Auto-detect CSV/TSV format
│   │   ├── Find variant column dynamically
│   │   └── Return list of variant IDs
│   │
│   ├── write_output_csv()
│   │   └── DataFrame → CSV with stable columns
│   │
│   └── write_output_json()
│       └── Results + metadata → pretty-printed JSON
│
├── COMMAND-LINE INTERFACE
│   ├── parse_arguments()
│   │   ├── --variant (single variant mode)
│   │   ├── --input (batch mode)
│   │   ├── --output (file path)
│   │   ├── --format (csv/json)
│   │   ├── --threshold (classification threshold)
│   │   └── --verbose (debug output)
│   │
│   └── main()
│       ├── Parse CLI arguments
│       ├── Load model and embeddings
│       ├── Route to single or batch mode
│       ├── Write output
│       └── Return exit code
│
└── __main__
    └── sys.exit(main())
```

### Usage Examples

```bash
# Single variant scoring
python scripts/score_variants.py --variant chr1_100000_A_G

# Batch scoring (auto-detect output format from extension)
python scripts/score_variants.py -i variants.csv -o results.csv

# Batch scoring with JSON output
python scripts/score_variants.py --input variants.csv --output results.json

# Custom threshold (e.g., for precision-focused predictions)
python scripts/score_variants.py --input variants.csv --output results.csv --threshold 0.7

# Verbose output for debugging
python scripts/score_variants.py --variant chr1_100000_A_G --verbose

# Integration into shell pipeline
cat variants.txt | python scripts/score_variants.py --input - --output scored.csv
```

### Help
```bash
python scripts/score_variants.py --help
```

---

## Part 3: Integration & Reproducibility

### Shared Components

Both interfaces share:

1. **Same trained model** – Identical Random Forest instance
2. **Same embeddings** – Precomputed ESM2 vectors from Dylan Tan
3. **Same metadata** – Variant-to-embedding index mappings
4. **Identical predictions** – Same input → same output across both interfaces
5. **Calibrated thresholds** – Confidence scores from model.predict_proba()

### Reproducibility Verification

To verify both interfaces produce identical results:

```bash
# 1. Score a variant via CLI
python scripts/score_variants.py --variant chr1_100000_A_G

# 2. Score same variant via Streamlit
#    - Navigate to "Single Variant" tab
#    - Enter "chr1_100000_A_G"
#    - Click "Score Variant"
#    - Results should match exactly

# 3. Score batch via CLI
python scripts/score_variants.py --input variants.csv --output cli_results.csv

# 4. Score batch via Streamlit
#    - Navigate to "Batch Upload" tab
#    - Upload variants.csv
#    - Click "Score All Variants"
#    - Download results
#    - Should match cli_results.csv exactly
```

### Model Consistency with Final Report

Both interfaces use:
- **Random Forest** as the reference model (XGBoost available as future extension)
- **Test AUROC**: 0.9299 [0.9145, 0.9453]
- **Test AUPRC**: 0.9473 [0.9328, 0.9618]
- **Test F1 Score**: 0.8761
- **Test Error Rate**: 14.6%
- **Test Brier Score**: 0.136 (RF uncalibrated)

These metrics can be verified in:
- Final Report: `Final Report/Morenu_EGN6933_FinalReport.tex` (Table I)
- Results files: `results/error_analysis_report.json`

---

## Part 4: Future Extensions

### Short-term Enhancements
1. Add SHAP/LIME explainability plots to Streamlit
2. Implement XGBoost as alternate model in CLI (`--model xgboost`)
3. Cache predictions to avoid redundant scoring
4. Add variant filtering/searching to batch results

### Medium-term Roadmap
1. Integrate with biological databases (Uniprot, PDBe) for structural context
2. Add confidence calibration curves for different variant populations
3. Implement meta-predictor ensemble (combine with other tools)
4. Export predictions to standard variant annotation formats (VCF, MAF)

### Long-term Vision
1. Multi-model ensemble voting (RF + XGBoost + MLP)
2. Attention visualization for embedding-based explanations
3. Real-time model retraining with user feedback
4. Integration with variant interpretation workflows (ClinVar, OMIM)

---

## Summary

| Feature | Streamlit App | CLI |
|---------|---------------|-----|
| **Single Variant** | ✅ Interactive form | ✅ Command-line |
| **Batch Scoring** | ✅ CSV upload | ✅ CSV/TSV input |
| **Output Formats** | N/A | ✅ CSV, JSON |
| **Visualization** | ✅ Rich plots | ❌ Text output |
| **Explainability** | ✅ Feature importance | ❌ Scores only |
| **Threshold Control** | ✅ Interactive slider | ✅ --threshold flag |
| **Automation** | ❌ Manual | ✅ Pipeline-friendly |
| **Web Deployment** | ✅ Streamlit Cloud | ✅ Docker container |

Both interfaces are production-ready, thoroughly documented, and ensure reproducibility with the Final Report metrics.
