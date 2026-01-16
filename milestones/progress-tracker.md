# Project Milestones & Progress Tracker

**Project Title:** Machine Learning Classification of Pathogenic vs. Benign Coding Genetic Variants Using Protein Language Model Embeddings  
**Student Name:** Angel Morenu  
**Faculty Advisor:** Dr. Fan  
**Last Updated:** January 15, 2026

---

## Pre-Course Checklist

- [x] Develop preliminary project idea (coding-variant pathogenicity classification)
- [x] Complete project abstract (half-page)
- [ ] Research potential faculty advisors (Bioinformatics, Computational Biology, Genomics)
- [ ] Contact faculty advisors (aim for 3-5 professors)
- [ ] Secure faculty advisor agreement
- [ ] Prepare presentation for first class

---

## Course Milestones

### Milestone 1: Project Planning & Setup
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Present preliminary project idea in class
- [ ] Submit formal project proposal
- [ ] Confirm faculty advisor
- [ ] Establish project scope (individual vs. group justification if applicable)

**Notes:**


---

### Milestone 2: Literature Review & Related Work
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Comprehensive literature review
- [ ] Related work analysis
- [ ] Identification of gaps and opportunities
- [ ] Refined problem statement

**Notes:**


---

### Milestone 3: Data Collection & Exploration
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Dataset acquired and documented
- [ ] Exploratory data analysis (EDA)
- [ ] Data quality assessment
- [ ] Preprocessing pipeline

**Notes:**


---

### Milestone 4: Methodology & Implementation Plan
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Detailed methodology document
- [ ] Technical architecture design
- [ ] Implementation timeline
- [ ] Risk assessment and mitigation plan

**Notes:**


---

### Milestone 5: Initial Development
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Baseline model/system implementation
- [ ] Initial results and metrics
- [ ] Code repository with documentation
- [ ] Progress presentation

**Notes:**


---

### Milestone 6: Refinement & Optimization
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Improved model/system
- [ ] Comparative analysis
- [ ] Performance optimization
- [ ] Validation results

**Notes:**


---

### Milestone 7: Evaluation & Testing
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Comprehensive evaluation framework
- [ ] Testing results and analysis
- [ ] Error analysis and limitations
- [ ] Documentation updates

**Notes:**


---

### Milestone 8: Final Deliverables
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Final project report
- [ ] Complete code repository with README
- [ ] Final presentation slides
- [ ] Project demonstration/demo
- [ ] Poster (if required)

**Notes:**


---

## Weekly Progress Log

### Week 1 (January 2026)
**Date:** January 7–15, 2026

**Accomplished:**
- Received course welcome and syllabus
- Set up project workspace and documentation structure
- ✅ Finalized project scope: coding-variant pathogenicity classification using ESM2 embeddings
- ✅ Updated proposal/README/architecture docs to match the coding-variant scope
- ✅ Renamed repository to match the updated scope and updated local remote

**Next Steps:**
- Obtain the curated coding-variant dataset details (location/format/schema)
- Implement ingestion + QC checks and create a versioned processed dataset artifact (Parquet)
- Decide and implement a gene/protein-aware split strategy to prevent leakage
- Begin baseline model training on embeddings (logistic regression / random forest)

**Blockers/Questions:**
- Confirm how gene/protein identifiers are represented in the curated dataset (gene symbol vs transcript vs protein ID)
- Confirm expected embedding format (vector dim; storage as columns vs arrays)

---

### Week 2

**Date:** 

**Accomplished:**


**Next Steps:**


**Blockers/Questions:**


---

## Advisor Meeting Notes

### Meeting 1
**Date:**  
**Attendees:**  
**Topics Discussed:**


**Action Items:**


**Next Meeting:**
### Key Papers
1. ESM-2 (protein language model embeddings) — primary representation approach
2. Gene/protein-aware evaluation and leakage control for variant prediction
3. Baseline classifiers for tabular embeddings (logistic regression, random forest)

### Datasets
1. Public curated coding-variant dataset (provided by Dylan)
2. ClinVar (optional validation/standardization): https://www.ncbi.nlm.nih.gov/clinvar/
3. Ensembl VEP (optional consequence annotation): https://www.ensembl.org/info/docs/tools/vep/

### Tools & Frameworks

1. PyTorch / TensorFlow (deep learning)
2. scikit-learn (baseline models, evaluation)
3. pandas, numpy, pyarrow (data processing)
4. MLflow / Weights & Biases (experiment tracking)
5. Streamlit / Gradio (demo interface)
6. HiPerGator (UF Research Computing) or Google Colab Pro

### Useful Links

1. ESM / ESM-2 project resources (overview + model cards)
2. ClinVar documentation and release notes
3. Ensembl VEP docs (if used for optional annotation)

---

## Notes & Ideas

[Use this section for quick notes, ideas, or things to remember]

