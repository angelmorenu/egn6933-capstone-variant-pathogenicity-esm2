# System Architecture - Regulatory Variant Interpretation ML System

**Project:** End-to-End ML System for Non-Coding Variant Interpretation  
**Date:** January 7, 2026  
**Version:** 0.1 (Planning Phase)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES (Public)                        │
├─────────────────────────────────────────────────────────────────────┤
│  ENCODE    │  GTEx    │  DeepSEA    │  1000 Genomes  │  Annotations│
│  (Chrom.)  │  (eQTL)  │  (Benchmark)│   (Variants)   │  (BED/VCF)  │
└──────┬─────────────────────────────────────────────────────┬─────────┘
       │                                                       │
       v                                                       v
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION & PREPROCESSING                   │
├─────────────────────────────────────────────────────────────────────┤
│  • Variant extraction & annotation                                   │
│  • DNA sequence retrieval (reference genome)                         │
│  • Functional signal alignment                                       │
│  • Quality control & filtering                                       │
│  • Train/Val/Test splitting                                          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING                           │
├─────────────────────────────────────────────────────────────────────┤
│  • DNA sequence encoding (one-hot / k-mer)                           │
│  • Functional signal normalization                                   │
│  • Context window selection                                          │
│  • Augmentation strategies                                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING                               │
├─────────────────────────────────────────────────────────────────────┤
│  Baseline Models           │  Deep Learning Models                   │
│  • Logistic Regression     │  • CNN (DeepSEA-style)                  │
│  • Random Forest           │  • Transformer (DNABERT-style)          │
│  • Gradient Boosting       │  • Hybrid architectures                 │
│                            │                                         │
│  Experiment Tracking: MLflow / Weights & Biases                     │
│  Hyperparameter Tuning: Optuna / Ray Tune                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL EVALUATION                             │
├─────────────────────────────────────────────────────────────────────┤
│  • Performance metrics (AUROC, AUPRC, F1)                            │
│  • Cross-context generalization                                      │
│  • Calibration analysis                                              │
│  • Error analysis & failure cases                                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────────────┐
│                       INTERPRETABILITY                               │
├─────────────────────────────────────────────────────────────────────┤
│  • Saliency maps / Attention visualization                           │
│  • In silico mutagenesis                                             │
│  • Motif discovery                                                   │
│  • Feature importance analysis                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE & DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────────┤
│  CLI Tool                    │  Web Interface (Streamlit/Gradio)     │
│  • Batch variant scoring     │  • Interactive variant input          │
│  • VCF file processing       │  • Real-time prediction               │
│  • Output generation         │  • Visualization dashboard            │
│                              │  • Downloadable reports               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Data Pipeline

**Purpose:** Ingest, preprocess, and prepare data for model training

**Inputs:**
- Raw genomic data files (FASTA, BED, VCF, BigWig)
- Functional genomics signals
- Variant annotations

**Processing Steps:**
```python
class DataPipeline:
    def __init__(self, config):
        self.reference_genome = load_reference()
        self.functional_data = load_functional_signals()
        
    def extract_variants(self, vcf_file):
        """Extract variants from VCF, filter non-coding"""
        pass
        
    def get_sequence_context(self, variant, window=1000):
        """Retrieve DNA sequence around variant"""
        pass
        
    def align_functional_signals(self, variant):
        """Get chromatin/TF signals at variant locus"""
        pass
        
    def create_dataset(self):
        """Generate train/val/test datasets"""
        pass
```

**Outputs:**
- Preprocessed datasets (HDF5 or PyTorch tensors)
- Metadata files (variant IDs, positions, labels)
- Statistics and QC reports

---

### 2. Feature Engineering

**DNA Sequence Encoding:**
```python
# One-hot encoding
# A = [1,0,0,0], C = [0,1,0,0], G = [0,0,1,0], T = [0,0,0,1]
sequence_tensor = one_hot_encode(dna_sequence)  # Shape: (4, seq_length)

# K-mer encoding (alternative)
kmer_features = extract_kmers(dna_sequence, k=6)
```

**Functional Signal Integration:**
```python
# Normalize chromatin accessibility scores
normalized_signals = (signals - mean) / std

# Multi-track integration
combined_features = concatenate([
    sequence_features,
    chromatin_features,
    tf_binding_features
])
```

---

### 3. Model Architectures

#### Baseline Models

```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
baseline = LogisticRegression(max_iter=1000)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
```

#### CNN Architecture (DeepSEA-inspired)

```python
import torch.nn as nn

class RegulatoryVariantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(4, 320, kernel_size=8)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(320, 480, kernel_size=8)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(480, 960, kernel_size=8)
        self.pool3 = nn.MaxPool1d(4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(960 * L, 925)  # L depends on input length
        self.fc2 = nn.Linear(925, num_tasks)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

#### Transformer Architecture (DNABERT-inspired)

```python
from transformers import BertModel, BertConfig

class DNATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(
            vocab_size=4096,  # For k-mer vocabulary
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=512
        )
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, num_tasks)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
```

---

### 4. Training Pipeline

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                          lr=config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5
        )
        
    def train_epoch(self):
        self.model.train()
        for batch in self.train_loader:
            sequences, labels = batch
            predictions = self.model(sequences)
            loss = self.criterion(predictions, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            # Calculate validation metrics
            auroc = calculate_auroc(predictions, labels)
            auprc = calculate_auprc(predictions, labels)
        return auroc, auprc
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch()
            auroc, auprc = self.validate()
            self.scheduler.step(auroc)
            
            # Log to MLflow
            mlflow.log_metrics({
                'auroc': auroc,
                'auprc': auprc
            }, step=epoch)
```

---

### 5. Evaluation Framework

```python
class Evaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        
    def compute_metrics(self, predictions, labels):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        metrics = {
            'auroc': roc_auc_score(labels, predictions),
            'auprc': average_precision_score(labels, predictions),
            'accuracy': accuracy_score(labels, predictions > 0.5),
            'f1': f1_score(labels, predictions > 0.5)
        }
        return metrics
        
    def cross_context_evaluation(self, cell_types):
        """Test generalization across cell types"""
        results = {}
        for cell_type in cell_types:
            test_data = load_cell_type_data(cell_type)
            metrics = self.compute_metrics(predictions, labels)
            results[cell_type] = metrics
        return results
        
    def calibration_analysis(self):
        """Check if predicted probabilities are well-calibrated"""
        from sklearn.calibration import calibration_curve
        return calibration_curve(labels, predictions, n_bins=10)
```

---

### 6. Interpretability Module

```python
class Interpreter:
    def __init__(self, model):
        self.model = model
        
    def saliency_map(self, sequence):
        """Compute gradient-based saliency"""
        sequence.requires_grad = True
        output = self.model(sequence)
        output.backward()
        return sequence.grad.abs()
        
    def in_silico_mutagenesis(self, sequence):
        """Test effect of all possible mutations"""
        original_pred = self.model(sequence)
        mutation_effects = {}
        
        for pos in range(len(sequence)):
            for base in ['A', 'C', 'G', 'T']:
                mutated_seq = mutate_sequence(sequence, pos, base)
                mutated_pred = self.model(mutated_seq)
                mutation_effects[(pos, base)] = mutated_pred - original_pred
                
        return mutation_effects
        
    def attention_visualization(self, sequence):
        """Extract and visualize attention weights"""
        # For transformer models
        outputs = self.model(sequence, output_attentions=True)
        return outputs.attentions
```

---

### 7. Inference Interface

#### CLI Tool

```python
# cli.py
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Score regulatory variants'
    )
    parser.add_argument('--vcf', required=True, 
                       help='Input VCF file')
    parser.add_argument('--model', required=True,
                       help='Path to trained model')
    parser.add_argument('--output', required=True,
                       help='Output file')
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    # Load model and process variants
    model = load_model(args.model)
    variants = load_vcf(args.vcf)
    scores = model.predict(variants)
    save_results(scores, args.output)

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
python cli.py --vcf variants.vcf --model best_model.pt --output scores.csv
```

#### Web Interface (Streamlit)

```python
# app.py
import streamlit as st

st.title("Regulatory Variant Impact Predictor")

# Input section
st.header("Enter Variant Information")
chromosome = st.selectbox("Chromosome", [f"chr{i}" for i in range(1, 23)])
position = st.number_input("Position", min_value=1)
ref_allele = st.text_input("Reference Allele")
alt_allele = st.text_input("Alternative Allele")

if st.button("Predict Impact"):
    # Process variant
    variant = create_variant(chromosome, position, ref_allele, alt_allele)
    
    # Get prediction
    with st.spinner("Analyzing variant..."):
        score = model.predict(variant)
        saliency = interpreter.saliency_map(variant)
    
    # Display results
    st.header("Prediction Results")
    st.metric("Regulatory Impact Score", f"{score:.3f}")
    
    # Visualization
    st.header("Sequence Importance")
    fig = plot_saliency_map(saliency)
    st.pyplot(fig)
    
    # Download results
    st.download_button(
        "Download Report",
        data=generate_report(variant, score, saliency),
        file_name="variant_report.pdf"
    )
```

---

## Technology Stack

### Core Libraries

**Data Processing:**
- `pysam` - BAM/VCF file handling
- `pybedtools` - Genomic interval operations
- `biopython` - Sequence manipulation
- `pandas`, `numpy` - Data manipulation

**Machine Learning:**
- `PyTorch` - Deep learning framework
- `scikit-learn` - Baseline models, metrics
- `transformers` - Pretrained models

**Experiment Tracking:**
- `MLflow` - Experiment management
- `Weights & Biases` - Alternative tracking

**Visualization:**
- `matplotlib`, `seaborn` - Static plots
- `plotly` - Interactive visualizations

**Deployment:**
- `Streamlit` or `Gradio` - Web interface
- `FastAPI` - REST API (optional)
- `Docker` - Containerization

---

## File Structure

```
regulatory-variant-ml/
├── README.md
├── requirements.txt
├── setup.py
├── environment.yml
│
├── configs/                  # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── data/                     # Data directory (gitignored)
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── augmentation.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── cnn.py
│   │   ├── transformer.py
│   │   └── ensemble.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── optimizer.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   ├── interpretation/
│   │   ├── __init__.py
│   │   ├── saliency.py
│   │   └── mutagenesis.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── visualization.py
│
├── scripts/                  # Standalone scripts
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── inference.py
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_results_analysis.ipynb
│
├── tests/                    # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── app/                      # Web interface
│   ├── streamlit_app.py
│   └── assets/
│
├── models/                   # Saved models (gitignored)
│   └── best_model.pt
│
├── results/                  # Outputs (gitignored)
│   ├── figures/
│   ├── tables/
│   └── predictions/
│
└── docs/                     # Documentation
    ├── data_schema.md
    ├── model_architecture.md
    └── api_reference.md
```

---

## Deployment Workflow

```
┌──────────────┐
│ Development  │
│  (Local)     │
└──────┬───────┘
       │
       v
┌──────────────┐
│   Testing    │
│  (Unit +     │
│  Integration)│
└──────┬───────┘
       │
       v
┌──────────────┐
│  Packaging   │
│  (Docker +   │
│  Requirements)│
└──────┬───────┘
       │
       v
┌──────────────┐
│ Demo Deploy  │
│ (Streamlit   │
│  Cloud)      │
└──────────────┘
```

---

## Computational Requirements

**Training:**
- GPU: NVIDIA Tesla T4 or better (16GB VRAM minimum)
- CPU: 8+ cores for data preprocessing
- RAM: 32GB minimum, 64GB recommended
- Storage: 100GB for datasets + models

**Inference:**
- CPU-only acceptable for demo
- GPU recommended for batch processing
- RAM: 16GB sufficient
- Storage: 10GB for model + reference data

**Estimated Costs:**
- Google Colab Pro: ~$10/month (sufficient for project)
- AWS p3.2xlarge: ~$3/hour (if needed for large experiments)
- Total budget: $50-100 for semester

---

## Success Metrics

**Technical Metrics:**
- AUROC ≥ 0.85 on test set
- AUPRC ≥ 0.70 on test set
- Cross-context AUROC ≥ 0.80
- Inference speed: <1 second per variant

**System Metrics:**
- Code coverage ≥ 70%
- Documentation completeness: 100%
- Reproducibility: All results reproducible from seed

**Deliverable Metrics:**
- GitHub stars/forks (community interest)
- Demo uptime ≥ 95%
- User-friendly interface (qualitative)

---

This architecture provides a clear roadmap for implementation while maintaining flexibility for adjustments based on advisor feedback and experimental results.
