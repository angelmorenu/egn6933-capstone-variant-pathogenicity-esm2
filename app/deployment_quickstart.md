# Quick Start Guide: Deployment Interfaces

## Installation

### 1. Install Dependencies

```bash
# Navigate to project directory
cd /path/to/EGN\ 6933\ –\ Project\ in\ Applied\ Data\ Science/Machine\ Learning\ Classification\ of\ Pathogenic\ vs.\ Benign\ Missense\ Variants\ Using\ Protein\ Language\ Model\ Embeddings

# Activate virtual environment (if using venv)
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements_deployment.txt
```

### 2. Verify Data Files

Ensure the following files exist:

```
data/processed/
├── week2_training_table_strict_embeddings.npy    (5000 × 1280 float32)
├── week2_training_table_strict_meta.json         (metadata index)
└── ...

results/
├── error_analysis_report.json                    (performance metrics)
└── ...
```

If files are missing, run the data preparation scripts from `scripts/`:
```bash
python scripts/build_week2_training_table.py
python scripts/baseline_train_eval.py
```

---

## Part A: Streamlit Web Application

### Quick Start

```bash
# Run the application
streamlit run app/app.py

# Open browser to http://localhost:8501
```

### Usage Examples

#### Single Variant Scoring
1. Click "Single Variant" in sidebar
2. Enter canonical ID: `chr1_100000_A_G`
3. Adjust confidence threshold (optional, default 0.5)
4. Click "🔍 Score Variant"
5. View prediction, confidence, and feature importance

#### Batch CSV Scoring
1. Prepare CSV file with column `variant_id`:
   ```csv
   variant_id
   chr1_100000_A_G
   chr2_150000_T_C
   chr3_200000_G_A
   ```

2. Click "Batch Upload" in sidebar
3. Upload CSV file
4. Click "🚀 Score All Variants"
5. Review ranked results table
6. Download results as CSV

#### View Model Performance Dashboard
1. Click "Performance Dashboard" in sidebar
2. Review key metrics (AUROC, AUPRC, F1, Error Rate)
3. Explore ROC/PR curves, confusion matrix, per-gene error rates
4. View statistical comparison results (DeLong test)

### Advanced Options

```bash
# Run on custom host/port
streamlit run app/app.py --server.port 8080 --server.address 0.0.0.0

# Run in headless mode (for deployment)
streamlit run app/app.py --logger.level=error --client.showErrorDetails=false

# Run with increased memory (for large datasets)
streamlit run app/app.py --maxUploadSize=200
```

### Configuration

Create `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200  # MB
port = 8501
headless = false
runOnSave = true
```

---

## Part B: Command-Line Interface

### Quick Start

```bash
# Activate virtual environment (if using venv)
source .venv/bin/activate

# Make script executable (optional)
chmod +x scripts/score_variants.py

# Score a single variant
python scripts/score_variants.py --variant chr1_100000_A_G

# Score batch from CSV
python scripts/score_variants.py --input variants.csv --output results.csv

# Get help
python scripts/score_variants.py --help
```

### Usage Examples

#### Single Variant

```bash
python scripts/score_variants.py --variant chr1_100000_A_G

# Output:
# [INFO] Loading model and embeddings...
# [INFO] Loaded embeddings: shape=(5000, 1280)
# [INFO] Scoring single variant: chr1_100000_A_G
# [RESULT]
#   Variant: chr1_100000_A_G
#   Prediction: PATHOGENIC
#   Pathogenicity Score: 0.8756
#   Confidence: 0.9234
```

#### Batch CSV to CSV

```bash
# Prepare input.csv
# variant_id
# chr1_100000_A_G
# chr2_150000_T_C
# ...

python scripts/score_variants.py \
  --input input.csv \
  --output results.csv

# Output: results.csv
# variant_id,pathogenicity_score,prediction,confidence,model,timestamp
# chr1_100000_A_G,0.8756,PATHOGENIC,0.9234,RandomForest,2026-03-20T15:30:45.123456
# ...
```

#### Batch CSV to JSON

```bash
python scripts/score_variants.py \
  --input input.csv \
  --output results.json \
  --format json

# Output: results.json (with metadata)
# {
#   "metadata": {
#     "timestamp": "2026-03-20T15:30:45.123456",
#     "model": "RandomForest",
#     "n_variants_scored": 2,
#     "n_pathogenic": 1,
#     "n_benign": 1
#   },
#   "results": [...]
# }
```

#### Custom Threshold

```bash
# Use 60% confidence threshold (more conservative)
python scripts/score_variants.py \
  --input variants.csv \
  --output results.csv \
  --threshold 0.6

# Use 40% threshold (more sensitive)
python scripts/score_variants.py \
  --input variants.csv \
  --output results.csv \
  --threshold 0.4
```

#### Verbose Mode

```bash
# Print detailed debug information
python scripts/score_variants.py \
  --variant chr1_100000_A_G \
  --verbose

# Useful for troubleshooting missing variants or errors
```

#### Shell Integration

```bash
# Pipeline with other tools
cat variants.txt | python scripts/score_variants.py --input - --output scored.csv

# Process multiple files
for file in variants_*.csv; do
  python scripts/score_variants.py --input "$file" --output "scored_${file}"
done

# Extract high-confidence pathogenic variants
python scripts/score_variants.py --input variants.csv --output results.csv
cat results.csv | awk -F, '$3 == "PATHOGENIC" && $4 > 0.9 {print}' > high_confidence.csv
```

### Output Format Reference

#### CSV Output
```csv
variant_id,pathogenicity_score,prediction,confidence,model,timestamp
chr1_100000_A_G,0.8756,PATHOGENIC,0.9234,RandomForest,2026-03-20T15:30:45.123456
chr2_150000_T_C,0.3421,BENIGN,0.8765,RandomForest,2026-03-20T15:30:46.234567
```

**Column Descriptions:**
- `variant_id`: Canonical variant identifier (chr_pos_ref_alt)
- `pathogenicity_score`: Predicted probability [0, 1] of being pathogenic
- `prediction`: Binary classification (PATHOGENIC or BENIGN)
- `confidence`: Model confidence (max ensemble probability)
- `model`: Model type (RandomForest)
- `timestamp`: Prediction timestamp (ISO 8601 format)

#### JSON Output
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

## Reproducibility Verification

### Verify Identical Results

Score the same variant through both interfaces:

```bash
# CLI
python scripts/score_variants.py --variant chr1_100000_A_G
# Output: Pathogenicity Score: 0.8756

# Streamlit: Single Variant tab → Input "chr1_100000_A_G" → Score Variant
# Output: Should show 0.8756
```

Both should produce identical scores.

### Verify Against Final Report Metrics

Check that model performance matches the Final Report:

```bash
# Final Report values:
# - Test AUROC: 0.9299 [0.9145, 0.9453]
# - Test AUPRC: 0.9473 [0.9328, 0.9618]
# - Test F1 Score: 0.8761
# - Test Error Rate: 14.6%

# Streamlit: Performance Dashboard tab
# Should display identical metrics
```

---

## Troubleshooting

### Issue: "Embeddings file not found"

**Solution:**
```bash
# Verify file exists
ls -la data/processed/week2_training_table_strict_embeddings.npy

# If missing, regenerate:
python scripts/build_week2_training_table.py
```

### Issue: "Variant not found in embedding data"

**Solution:**
```bash
# Verify variant format is correct (chr_pos_ref_alt)
# Valid: chr1_100000_A_G
# Invalid: 1_100000_A_G (missing "chr")

# Check if variant is in dataset
python -c "
import json
with open('data/processed/week2_training_table_strict_meta.json') as f:
    meta = json.load(f)
    variant = 'chr1_100000_A_G'
    print(variant in meta['variant_id_map'])
"
```

### Issue: Streamlit runs slowly

**Solution:**
```bash
# Clear cache
streamlit cache clear

# Run with reduced logging
streamlit run app/app.py --logger.level=error

# Use faster embeddings (if available)
# See code comments for optimization options
```

### Issue: "Invalid threshold value"

**Solution:**
```bash
# Threshold must be between 0.0 and 1.0
# Valid: --threshold 0.5
# Invalid: --threshold 1.5 or --threshold -0.1

python scripts/score_variants.py --variant chr1_100000_A_G --threshold 0.5
```

---

## Performance Benchmarks

### Single Variant Scoring (CLI)
- **Time to first prediction:** ~2 seconds (model/data loading)
- **Per-variant scoring:** <10 ms

### Batch Scoring (CLI)
- **1,000 variants:** ~3 seconds (including I/O)
- **10,000 variants:** ~20 seconds
- **Throughput:** ~5,000 variants/second

### Streamlit Web App
- **Page load:** ~5 seconds (first time)
- **Single variant prediction:** ~1 second (after loading)
- **Batch upload processing:** ~2 seconds per 100 variants

---

## Next Steps

1. **Test both interfaces** with sample variants
2. **Verify reproducibility** (same inputs → same outputs)
3. **Review performance** against Final Report metrics
4. **Deploy to production** (see deployment section in `docs/DEPLOYMENT_INTERFACES.md`)
5. **Integrate into workflows** (use CLI for automation)

---

## Support

For issues or questions:
1. Check `docs/DEPLOYMENT_INTERFACES.md` for detailed documentation
2. Review error messages and troubleshooting section
3. Check `scripts/score_variants.py --help` for CLI usage
4. Contact: Angel Morenu (angelmorenu@ufl.edu)
