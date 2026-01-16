# System Architecture - Coding Variant Pathogenicity Classifier (ESM2)

**Project:** End-to-End ML System for Coding Variant Pathogenicity Classification  
**Date:** January 15, 2026  
**Version:** 0.2 (Coding-Variant Scope)

---

## High-Level Architecture (Current)

```
┌──────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                             │
├──────────────────────────────────────────────────────────────────────┤
│  Public curated coding-variant dataset (labels + ESM2 embeddings)     │
│  Optional: ClinVar + VEP for validation/standardization               │
└───────────────┬───────────────────────────────────────────┬──────────┘
                │                                           │
                v                                           v
┌──────────────────────────────────────────────────────────────────────┐
│                     INGESTION & QUALITY CONTROL                       │
├──────────────────────────────────────────────────────────────────────┤
│  • Load raw dataset (PKL/CSV/Parquet)                                 │
│  • Normalize variant identifiers / metadata                            │
│  • Apply label policy (strict vs relaxed)                              │
│  • Validate embeddings (shape, NaNs, dtype)                            │
│  • Write versioned Parquet artifacts                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌──────────────────────────────────────────────────────────────────────┐
│                         FEATURE / DATA STORE                          │
├──────────────────────────────────────────────────────────────────────┤
│  data/processed/*.parquet     data/embeddings/* (optional cache)      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌──────────────────────────────────────────────────────────────────────┐
│                  SPLITTING (LEAKAGE-AWARE) & TRAINING                 │
├──────────────────────────────────────────────────────────────────────┤
│  • Gene/protein-aware train/val/test splits                            │
│  • Baselines: Logistic Regression, Random Forest, optional MLP         │
│  • Class weighting + threshold tuning                                  │
│  • Calibration (Platt / isotonic)                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌──────────────────────────────────────────────────────────────────────┐
│                       EVALUATION & REPORTING                          │
├──────────────────────────────────────────────────────────────────────┤
│  • AUROC / AUPRC + confidence intervals                                │
│  • Paired comparisons (e.g., DeLong for AUROC)                         │
│  • Error analysis + interpretability (feature importance / SHAP)       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               v
┌──────────────────────────────────────────────────────────────────────┐
│                       INFERENCE & DEPLOYMENT                           │
├──────────────────────────────────────────────────────────────────────┤
│  CLI scoring tool                  Streamlit web app                   │
│  • Batch scoring                   • Interactive variant scoring       │
│  • CSV/VCF-derived tables          • Probabilities + explanation       │
└──────────────────────────────────────────────────────────────────────┘
```

### Implementation Map (Repo)

- Ingestion/QC: `scripts/` (e.g., PKL→Parquet ingestion)
- Core ML pipeline: `src/variant_classifier/`
- Embedding utilities: `src/variant_embeddings/`
- Config: `config/`
- Tests: `tests/`
- Proposal/Docs: `project-proposal/`, `docs/`

---
