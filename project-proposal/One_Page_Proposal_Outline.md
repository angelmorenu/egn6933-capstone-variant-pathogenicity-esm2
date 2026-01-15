# One-Page Proposal Outline (EGN 6933)

**Project:** Machine Learning Classification of Pathogenic vs. Benign Non-Coding Genetic Variants Using DNA Sequence Embeddings  
**Student:** Angel Morenu  
**Date:** January 13, 2026

## Problem & Impact
Non-coding variants are difficult to interpret experimentally, limiting rare disease and precision-medicine workflows. This project builds a prediction-first ML pipeline to prioritize non-coding variants as likely pathogenic vs. benign.

## Data
- **Primary source:** ClinVar public releases (VCF/variant summaries)
- **Labels:** Pathogenic/Likely pathogenic vs. Benign/Likely benign; exclude VUS/conflicting
- **Subset:** non-coding variants via Ensembl VEP consequence annotation
- **Features:** fixed-length DNA sequence windows around each variant
- **Collaboration (benchmark/augmentation):** Dylan Tan’s pathogenicity dataset with precomputed ESM2 embedding features on UF HiPerGator
	- Dataset location: `/blue/xiaofan/dtan1/Output_Folder/ESM2_PrimateAI_Scripts/` (Baseline/ and CoordsData/)
	- My HiPerGator working path: `/blue/xiaofan/angel.morenu` (staging outputs/logs when running on HiPerGator)
	- Local standardized ingestion: `scripts/ingest_esm2_primateai.py` → `data/processed/esm2_primateai/*.parquet`

## Method (Technical Plan)
- Compute **pretrained DNA embedding** features (DNABERT-style) for each window
- Train embedding-based classifiers: **Logistic Regression**, **Random Forest**, optional **shallow MLP**
- Package as a reproducible pipeline (saved splits, configs, seeds)

## Evaluation (Statistical Rigor)
- Metrics: **AUROC**, **AUPRC** (primary), plus precision/recall/F1
- Leakage control: **chromosome holdout** train/val/test split
- Uncertainty: **bootstrapped 95% CIs** for AUROC/AUPRC
- Model comparison: **DeLong test** (AUROC) + paired bootstrap/permutation (AUPRC)
- Calibration check (supporting): reliability curve + **Brier score** (optional Platt/isotonic if needed)

## Deployment (“The App”)
- Streamlit app + CLI to score a user-provided variant (or CSV) and return a calibrated pathogenic probability and predicted label.

## Timeline (15 weeks)
- Weeks 1–4: data download + filters + VEP annotation + EDA + split/window decisions
- Weeks 5–8: embeddings + LR/RF baselines + first evaluation
- Weeks 9–12: optional MLP + final evaluation + statistical testing + error analysis
- Weeks 13–15: deployment + documentation + final report + demo
