# EGN 6933 – Project in Applied Data Science

**Student:** Angel Morenu 
**Semester:** Spring 2026  
**Instructor:** Dr. Edwin Marte Zorrilla  
**Faculty Advisor:** [TBD - Target: Bioinformatics/Computational Biology/Genomics faculty]

## Course Overview

This course focuses on designing and developing an end-to-end capstone project in applied data science and AI. The goal is to build impactful, data-driven solutions to real-world problems through careful planning, iterative development, and rigorous evaluation.

## Project Information

**Project Title:** Machine Learning Classification of Pathogenic vs. Benign Non-Coding Genetic Variants Using DNA Sequence Embeddings

**Project Type:** Individual  
**Status:** Planning Phase

### Project Abstract

This project builds a prediction-first machine learning pipeline to classify **non-coding ClinVar variants** as pathogenic vs. benign using **pretrained DNA sequence embeddings** (DNABERT-style) and simple classifiers (logistic regression, random forest, optional shallow MLP). Deliverables include a reproducible pipeline (data regeneration + splits + cached features), rigorous evaluation under class imbalance (AUPRC/AUROC + CIs + paired tests), and a user-facing Streamlit + CLI scoring interface.

**Key Features:**
- Single primary dataset: ClinVar
- Single task: binary classification (pathogenic vs benign) on high-confidence labels
- Transfer learning via DNA foundation-model embeddings
- Leakage-aware evaluation (chromosome holdout) + statistical testing
- Deployable scoring app (Streamlit) + CLI
- Reproducible engineering practices (conda/Docker, cached artifacts, tests)

## Repository Structure

```
.
├── README.md                 # This file
├── project-proposal/         # Proposal documents
├── research/                 # Literature review, related work
├── milestones/              # Milestone deliverables and progress
├── code/                    # Source code and notebooks
└── documentation/           # Technical documentation
```

## Key Milestones

- [ ] **Pre-Class:** Develop preliminary project idea
- [ ] **Pre-Class:** Secure faculty advisor
- [ ] **Class 1:** Present preliminary project idea
- [x] Finalize project scope and proposal
- [ ] Literature review and related work
- [ ] Data collection and preprocessing
- [ ] Model development and implementation
- [ ] Evaluation and testing
- [ ] Final deliverables and presentation

## Contact Information

**Course Instructor:**  
Dr. Edwin Marte Zorrilla  
Email: emartezorrilla@ufl.edu  
Phone: 352-392-0638  
Office: Nuclear Sciences Bldg., Rm 329

**Faculty Advisor:**  
[Name]  
[Email]  
[Office/Contact Info]

## Important Notes

- First class meeting: Be prepared with a preliminary project idea
- Faculty advisor should be secured (or in progress) before first class
- Group projects require expanded scope and clear rationale
- Canvas course site is under development - check regularly for updates
