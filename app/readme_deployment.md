# Deployment Interfaces ReadMe

## Overview

This directory contains production-ready deployment interfaces for the variant pathogenicity prediction model trained on ESM2 embeddings.

### What's Included

- **Streamlit Web Application** (`app.py`) – Interactive web interface for researchers
- **Command-Line Interface** (`scripts/score_variants.py`) – Automation-friendly batch processing
- **Comprehensive Documentation** – Architecture, usage, and deployment guides
- **Verification Script** (`verify_deployment.py`) – Automated validation tool

---

## Quick Start (< 5 minutes)

### Option A: Web Application (Interactive)

```bash
# Install dependencies
pip install streamlit scikit-learn plotly pandas numpy

# Run the web app
streamlit run app.py

# Open browser to http://localhost:8501
```

Then:
1. Click "Single Variant" to score one variant
2. Click "Batch Upload" to score a CSV file
3. Click "Performance Dashboard" to see model metrics

### Option B: Command-Line (Automated)

```bash
# Single variant
python scripts/score_variants.py --variant chr1_100000_A_G

# Batch CSV to CSV
python scripts/score_variants.py --input variants.csv --output results.csv

# Batch CSV to JSON
python scripts/score_variants.py --input variants.csv --output results.json

# Get help
python scripts/score_variants.py --help
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)** | High-level overview, feature comparison, architecture decisions |
| **[DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md)** | Step-by-step usage guide with examples |
| **[docs/DEPLOYMENT_INTERFACES.md](docs/DEPLOYMENT_INTERFACES.md)** | Technical architecture, data requirements, deployment options |

---

## Verification

Verify both interfaces are working correctly:

```bash
# Run verification script
python verify_deployment.py

# Expected output:
# ✅ PASS | File Structure
# ✅ PASS | Dependencies
# ✅ PASS | Streamlit Structure
# ✅ PASS | CLI Syntax
# ✅ PASS | CLI Single Variant
# ✅ PASS | CLI Batch Scoring
# ✅ PASS | Output Formats
# ✅ PASS | Metrics Consistency
#
# OVERALL: 8/8 checks passed
# ✅ **DEPLOYMENT READY**
```

---

## Data Requirements

Both interfaces require:

```
data/processed/
├── week2_training_table_strict_embeddings.npy    (5000 × 1280)
├── week2_training_table_strict_meta.json         (variant index)
└── ...

results/
├── error_analysis_report.json                    (metrics)
└── ...
```

---

## Model Performance

Both interfaces use the same trained model with these metrics:

| Metric | Value |
|--------|-------|
| Test AUROC | 0.9299 [0.9145, 0.9453] |
| Test AUPRC | 0.9473 [0.9328, 0.9618] |
| Test F1 Score | 0.8761 |
| Test Error Rate | 14.6% |
| Test Brier Score (uncalibrated) | 0.136 |

---

## Features

### Streamlit Web Application

✅ **Single-Variant Scoring**
- Interactive form with canonical ID or manual entry
- Adjustable confidence threshold
- Color-coded results (red = pathogenic, green = benign)
- Confidence metrics and feature importance visualization

✅ **Batch Scoring**
- CSV file upload
- Ranked results (sorted by pathogenicity)
- Downloadable CSV export

✅ **Model Performance Dashboard**
- ROC and PR curves
- Confusion matrix
- Per-gene error rates
- Statistical test results (DeLong test)

✅ **Explainability**
- Feature importance bar chart
- Confidence metrics
- Variant history tracking

### Command-Line Interface

✅ **Single-Variant Scoring**
- Canonical variant ID input
- Pathogenicity score output
- High confidence performance

✅ **Batch Scoring**
- CSV/TSV input support
- CSV or JSON output
- Metadata with summary statistics
- Ranked results (descending pathogenicity)

✅ **Integration-Friendly**
- Stable output format
- Proper exit codes
- Error handling with informative messages
- Scriptable for automation

---

## Usage Examples

### Streamlit Examples

**Score a single variant:**
1. Navigate to "Single Variant" tab
2. Enter: `chr1_100000_A_G`
3. Click "🔍 Score Variant"
4. View prediction and explanation

**Score a batch:**
1. Create CSV: `variants.csv` with column `variant_id`
2. Navigate to "Batch Upload" tab
3. Upload file
4. Click "🚀 Score All Variants"
5. Download results

### CLI Examples

```bash
# Single variant
python scripts/score_variants.py --variant chr1_100000_A_G

# Batch with custom threshold
python scripts/score_variants.py \
  --input variants.csv \
  --output results.csv \
  --threshold 0.6

# JSON output
python scripts/score_variants.py \
  --input variants.csv \
  --output results.json

# Verbose mode (debugging)
python scripts/score_variants.py \
  --variant chr1_100000_A_G \
  --verbose
```

---

## Output Formats

### CSV Output (CLI)

```csv
variant_id,pathogenicity_score,prediction,confidence,model,timestamp
chr1_100000_A_G,0.8756,PATHOGENIC,0.9234,RandomForest,2026-03-20T15:30:45.123456
chr2_150000_T_C,0.3421,BENIGN,0.8765,RandomForest,2026-03-20T15:30:46.234567
```

### JSON Output (CLI)

```json
{
  "metadata": {
    "timestamp": "2026-03-20T15:30:45.123456",
    "model": "RandomForest",
    "n_variants_scored": 2,
    "n_pathogenic": 1,
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
    }
  ]
}
```

---

## Deployment Options

### Local Machine
```bash
# Web app
streamlit run app.py

# CLI
python scripts/score_variants.py --input variants.csv --output results.csv
```

### Streamlit Cloud
```bash
# Push to GitHub, deploy at https://share.streamlit.io
```

### Docker Container
```bash
docker build -t variant-predictor .
docker run -p 8501:8501 variant-predictor
```

### Cloud Platforms
- AWS Lambda (CLI)
- Google Cloud Run (Streamlit or CLI)
- Heroku (Streamlit)
- See `docs/DEPLOYMENT_INTERFACES.md` for details

---

## Reproducibility

Both interfaces produce identical results for the same input:

```bash
# CLI
python scripts/score_variants.py --variant chr1_100000_A_G
# Output: Pathogenicity Score: 0.8756

# Streamlit: Single Variant → chr1_100000_A_G → Score Variant
# Output: 0.8756 (identical)
```

Verify against Final Report metrics:
- Final Report Table I: AUROC 0.9299, AUPRC 0.9473, F1 0.8761
- Streamlit Dashboard: Same metrics displayed
- CLI outputs: Same model used for predictions

---

## Troubleshooting

### "Embeddings file not found"
```bash
# Verify file exists
ls -la data/processed/week2_training_table_strict_embeddings.npy

# Regenerate if missing
python scripts/build_week2_training_table.py
```

### "Variant not found in embedding data"
```bash
# Check variant format: chr_pos_ref_alt
# Valid: chr1_100000_A_G
# Invalid: 1_100000_A_G or chr1:100000:A:G
```

### Streamlit runs slowly
```bash
# Clear cache
streamlit cache clear

# Run with reduced logging
streamlit run app.py --logger.level=error
```

### CLI batch processing is slow
```bash
# Use multiple workers (Linux/macOS)
OMP_NUM_THREADS=4 python scripts/score_variants.py --input large.csv --output results.csv
```

---

## Performance Metrics

### CLI
- **Single variant:** ~2 sec (first time, including model load) + <10ms per variant
- **Batch 1,000 variants:** ~3 seconds
- **Throughput:** ~5,000 variants/second

### Streamlit Web App
- **First load:** ~5 seconds
- **Single variant prediction:** ~1 second
- **Batch upload processing:** ~2 seconds per 100 variants

---

## Testing

Run automated verification:

```bash
# Full verification
python verify_deployment.py

# Test CLI only
python verify_deployment.py --cli

# Test Streamlit only
python verify_deployment.py --streamlit
```

---

## Files Included

```
├── app.py                              (800+ lines, Streamlit web app)
├── scripts/
│   └── score_variants.py              (600+ lines, CLI interface)
├── docs/
│   └── DEPLOYMENT_INTERFACES.md       (400+ lines, technical docs)
├── verify_deployment.py                (300+ lines, verification script)
├── DEPLOYMENT_QUICKSTART.md            (300+ lines, user guide)
├── DEPLOYMENT_SUMMARY.md               (200+ lines, overview)
├── requirements_deployment.txt         (dependencies)
└── README.md                           (this file)
```

---

## Contact & Support

**Student:** Angel Morenu  
**Email:** angelmorenu@ufl.edu  
**GitHub:** [angelmorenu/egn6933-capstone-variant-pathogenicity-esm2](https://github.com/angelmorenu/egn6933-capstone-variant-pathogenicity-esm2)

---

## License

MIT License – See LICENSE file in project root

---

## Next Steps

1. **Install dependencies:** `pip install -r requirements_deployment.txt`
2. **Verify setup:** `python verify_deployment.py`
3. **Choose interface:**
   - Interactive: `streamlit run app.py`
   - Command-line: `python scripts/score_variants.py --help`
4. **Read documentation:** See DEPLOYMENT_QUICKSTART.md for detailed examples
5. **Deploy to production:** See DEPLOYMENT_INTERFACES.md for deployment options

---

**Status:** ✅ **PRODUCTION READY**
