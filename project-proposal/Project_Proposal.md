# M.S. Applied Data Science: Capstone Project Proposal

**Due Date:** January 25, 2025  
**Student Name:** Angel Morenu  
**Course:** EGN 6933 – Project in Applied Data Science  
**Project Type:** Individual

---

## 1. Project Title & Team Members

**Project Name:** Machine Learning Classification of Pathogenic vs. Benign Coding Genetic Variants Using Protein Language Model Embeddings

**Team Lead:** Angel Morenu (Individual Project)

**Collaboration:** Dylan Tan has provided access to a public curated pathogenicity dataset of coding variants with embedding-style features (ESM2 protein language model embeddings), as well as guidance and reference code on handling these feature datasets.

---

## 2. Problem Statement & Impact

Interpreting the clinical impact of human genetic variation is a central challenge in precision medicine. In particular, coding variants that alter protein sequence (e.g., missense and loss-of-function variants) can disrupt protein stability, function, and interactions, and are frequently implicated in rare disease and inherited disorders. However, experimentally validating variant effects at scale remains costly and time-intensive, motivating computational methods that can prioritize variants for downstream laboratory follow-up.

This capstone focuses on a prediction-focused, semester-feasible supervised learning task: binary classification of coding genetic variants as pathogenic versus benign using pretrained protein language model embeddings. The objective is not to resolve the complete mechanistic basis of variant pathogenicity, but rather to construct a reproducible machine learning pipeline that can effectively prioritize coding variants for downstream experimental validation and clinical follow-up.

The stakeholders for this work include rare disease research teams seeking to accelerate variant interpretation, clinical genomics analysts requiring computational decision-support tools, and computational genomics researchers developing scalable variant annotation workflows. The work is positioned within the broader precision medicine context, where accurate computational variant classification can reduce the burden of manual expert review and allow limited experimental resources to be focused on the most promising research directions.

From a societal and ethical perspective, this project operates with publicly available, de-identified genetic variant data (Landrum et al., 2018). The project treats variant data responsibly and explicitly notes that model predictions are not intended for clinical diagnosis without independent laboratory confirmation and expert medical interpretation. The project will address class imbalance and label uncertainty explicitly by excluding ambiguous records (e.g., variants of uncertain significance, records with conflicting clinical interpretations) and by reporting performance metrics that remain valid under class imbalance.

This project integrates applied data science—encompassing machine learning, rigorous statistical evaluation, and reproducible model deployment—with computational genomics and healthcare workflows, demonstrating how variant prioritization connects practical machine learning methods to clinically relevant research applications.

---

## 3. Data Acquisition & Viability

The primary data source for this project is a public curated pathogenicity dataset of coding variants assembled by Dr. Fan’s group and provided via Dylan Tan. The dataset contains coding variants with associated pathogenicity labels and precomputed embedding-style features derived from protein language models (ESM2). This dataset is suitable for a semester capstone because it provides a clean starting point for machine learning experimentation while still requiring rigorous data handling, leakage-aware evaluation, and reproducible deployment.

As a secondary validation pathway (and to ensure full reproducibility), the project can additionally download and parse ClinVar (https://www.ncbi.nlm.nih.gov/clinvar/) and optionally apply consequence annotation using Ensembl VEP (McLaren et al., 2016) to confirm coding consequences and enable independent reconstruction of comparable training tables.

Clinical significance labels will be strictly defined to maximize confidence in the training signal. Variants labeled as Pathogenic or Likely Pathogenic will be assigned to the pathogenic class, while variants labeled as Benign or Likely Benign will be assigned to the benign class. Variants classified as Uncertain Significance or those with conflicting clinical interpretations across submitters will be excluded from model training and evaluation to reduce label noise and maintain high-confidence positive and negative examples.

Coding variants will be defined as variants predicted to alter protein-coding transcripts (e.g., missense, nonsense, frameshift, and other coding consequences). If needed, Ensembl VEP (McLaren et al., 2016) will be used to confirm and standardize consequence categories across variants.

Feature representation will be based on protein language model embeddings. Specifically, ESM2 embeddings provided in the curated dataset will be used as fixed-dimensional feature vectors suitable for classical machine learning models. If any variants require missing feature generation, the project will generate embeddings from relevant protein sequence context using the ESM2 model and cache them in standardized formats to ensure reproducibility.

Data curation and versioning will follow software engineering best practices. The curated labeled dataset will be stored as versioned Parquet files that explicitly record the ClinVar release date, consequence annotation parameters, window size, and label filtering criteria applied during construction. All dataset processing steps will be deterministic and fully scripted, ensuring that researchers can reconstruct the labeled dataset deterministically from the original ClinVar public release. Embedding features will be cached to disk in standardized formats to enable reproducible model training without requiring re-computation of expensive embedding operations.

The curated dataset and ClinVar are public and de-identified, and the project does not involve participant recruitment or private health information. The project will be executed in accordance with university research and data-handling best practices. The computational workflow will be thoroughly documented and sufficient implementation detail will be provided that other researchers can reproduce the analysis using public data sources. The project is structured as a research proof-of-concept and explicitly disclaims clinical utility; model predictions are intended for research-driven variant prioritization rather than clinical diagnostic use.

---

## 4. Technical Execution & Complexity

The end-to-end pipeline comprises seven major computational phases. First, the curated coding-variant dataset is obtained and parsed, variant representations are normalized, and clinical significance labels are applied and filtered. Second, coding consequences are standardized (optionally via Ensembl VEP) and basic quality-control checks are performed. Third, ESM2 embedding features are loaded from the curated dataset and/or generated when needed, then cached to disk in standardized formats. Fourth, train, validation, and test splits are created using gene- or protein-aware stratification to prevent data leakage (ensuring that variants from the same gene/protein do not appear in both training and test sets). Fifth, multiple classifier models are trained on the embedding features and compared using rigorous statistical tests. Sixth, the best-performing model is calibrated and validated on held-out data. Seventh, the final model is deployed as both a web application and command-line tool.

The project will implement three complementary classifier architectures, all trained on the same embedding features. Logistic Regression will serve as an interpretable, well-behaved baseline model. Random Forest will provide a nonlinear, ensemble-based alternative that can capture complex patterns in the embedding space. Optionally, a shallow Multi-Layer Perceptron will be implemented as a neural network baseline to assess whether additional model complexity provides incremental improvements in predictive performance.

This project addresses several technically sophisticated challenges at the Master's level. First, it implements transfer learning by applying pretrained protein language models (ESM2-style embeddings) to a variant pathogenicity classification task, requiring understanding of representation learning and its application to biological sequence data. Second, the project employs leakage-aware experimental design using gene/protein-aware splitting to prevent overoptimistic performance estimates that would arise if variants from the same gene/protein appeared in both training and test sets. Third, the project implements rigorous statistical testing (DeLong tests for AUROC comparison, paired bootstrap and permutation tests for AUPRC comparison) to distinguish genuine performance differences from random variation. Fourth, the project delivers a complete machine learning system suitable for production deployment, including model serialization, reproducible computation environments (via conda and optional Docker containerization), and user-facing interfaces (interactive web application and command-line tool).

Reproducibility is central to the project's design. All random seeds are fixed at every stage (embedding computation, train/test splitting, model training) to enable deterministic replication of results. Hyperparameters and dataset filtering criteria are specified in tracked configuration files, allowing different analysis variants to be reproduced by simply changing configuration parameters. Train, validation, and test splits are explicitly saved to disk so that identical data partitions can be used for future analyses or by other researchers. The computational environment will be captured via conda environment files, documenting exact package versions and dependencies. The codebase will follow professional Python standards, including type annotations, modular organization into separate modules for data loading, feature engineering, model training, evaluation, and deployment, and formatting/linting via tools such as Black and Ruff. Unit tests will be implemented for critical pipeline steps to catch regressions and ensure long-term maintainability.

---

## 5. Deployment Plan: "The App"

The final deliverable includes a user-facing application that scores coding genetic variants and returns calibrated pathogenicity probabilities. The application will be implemented as a Streamlit web application, accepting as input either a single coding variant (specified by chromosome, genomic position, reference allele, alternate allele, and genome assembly, and/or a transcript/protein identifier if available) or a small CSV file containing multiple variants. The application returns for each variant a calibrated probability of pathogenicity and a predicted class label (pathogenic or benign) based on a documented decision threshold.

Additionally, a command-line interface will be provided to enable batch scoring of larger variant datasets provided as CSV or VCF-derived tables. This command-line tool allows seamless integration into automated variant interpretation pipelines and bioinformatics workflows used by computational research laboratories.

The complete set of end-to-end project deliverables includes: (1) a reproducible pipeline with associated configuration files that can rebuild the dataset and models from the original public ClinVar release; (2) trained model artifacts (serialized scikit-learn or PyTorch models) accompanied by an evaluation report documenting performance metrics, statistical tests, and detailed error analysis; (3) a Streamlit web application for interactive variant scoring; and (4) a command-line tool for batch processing of variant sets.

---

## 6. Statistical Analysis & Evaluation

Classification performance will be evaluated using two primary metrics selected for their appropriateness to imbalanced binary classification problems. Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) are both threshold-independent metrics that remain valid across different class imbalance ratios. Secondary metrics will include precision, recall, F1 score, and balanced accuracy to provide a comprehensive view of model behavior on both the majority and minority classes.

The experimental design incorporates leakage-aware splitting to avoid artificially inflated performance estimates. Specifically, the dataset will be divided into training, validation, and test subsets using gene- or protein-aware stratification, where all variants mapping to the same gene/protein are assigned to a single split. This strategy reduces the risk of leakage wherein highly related variants share protein context and embeddings across train and test, inflating apparent performance. As a sensitivity analysis, the gene/protein assignment strategy will be varied across multiple random seeds to test whether performance estimates remain stable across different partitions.

The project will train and compare three complementary models—Logistic Regression, Random Forest, and optionally a shallow Multi-Layer Perceptron—all using the same embedding feature sets. Performance comparisons will employ rigorous statistical tests appropriate for paired predictions on the same test set. The DeLong test will be used to compare AUROC values between models, while paired bootstrap and permutation tests will compare AUPRC values. Bootstrapped 95% confidence intervals will be constructed for AUROC and AUPRC on the held-out test set, providing not merely point estimates but ranges of plausible performance values.

Because the dataset is expected to exhibit class imbalance (often more benign than pathogenic variants), the project will employ class weighting during model training to penalize misclassification of the minority (pathogenic) class more heavily. Decision thresholds will be tuned on the validation set to optimize a desired trade-off between sensitivity and specificity. Performance will be reported as precision-recall curves to clearly visualize the trade-off between precision and recall across different operating points.

The final trained model will be calibrated using the validation set (via Platt scaling or isotonic regression) to ensure that predicted probabilities reflect true conditional class probabilities. Calibration will be assessed using calibration curves and the Brier score, enabling downstream users to interpret predicted probabilities as genuine probability estimates rather than arbitrary model scores.

Supporting interpretability analyses will provide insights into model predictions and feature importance. For tree-based models, feature importance scores will identify which embedding dimensions most strongly influence predictions. For all models, calibrated probability outputs will serve as risk scores for downstream decision-making. Lightweight attribution analysis will be performed for a small set of representative variants, highlighting which parts of the protein sequence context and/or embedding dimensions most influence model predictions. If time permits, SHAP (SHapley Additive exPlanations) analysis will provide model-agnostic, local and global explanations of model behavior within the embedding feature space.

If, during exploratory analysis, the positive class (pathogenic variants) is discovered to be too small after strict filtering to support reliable model training, the filtering criteria will be relaxed iteratively (e.g., expanding the label set to include Likely Pathogenic variants) while maintaining high confidence by continuing to exclude ambiguous records (VUS, conflicting interpretations).

---

## 7. Project Timeline & Milestones

The project is organized into seven phases across fifteen weeks.

During weeks 1–4, the focus is on data acquisition and preparation. Activities include obtaining and validating the curated coding-variant dataset, defining and applying label filtering rules, optionally standardizing consequences via Ensembl VEP, conducting exploratory data analysis, and designing the gene/protein-aware split strategy.

Weeks 5–8 focus on feature engineering and baseline model development. Embedding vectors will be generated for all variants. Logistic Regression and Random Forest classifiers will be trained on the embedding features, and initial performance estimates using AUROC and AUPRC will be computed. Iterative analysis of class imbalance effects will inform threshold tuning and the selection of decision operating points.

Weeks 9–12 are dedicated to model refinement and rigorous statistical evaluation. If time permits, a shallow Multi-Layer Perceptron will be implemented as an optional extension. Final model evaluation will include bootstrapped confidence intervals, paired statistical tests comparing model performance, and detailed error analysis to understand which types of variants are misclassified and why.

Weeks 13–15 focus on deployment, documentation, and presentation. The Streamlit web application and command-line interface will be implemented. Project documentation, inline code comments, and the final capstone report will be completed. A project presentation and interactive demo will be prepared for peer and instructor review.

This is an individual capstone project. Feedback will be iteratively solicited from the instructor and advisor throughout the semester, and a final project presentation will be delivered to peers for collaborative review.

---

## 8. New Knowledge Acquisition

This project offers hands-on experience with a number of modern computational genomics and machine learning methods. It shows how pretrained protein language models (ESM2-style embeddings) may be used practically for feature extraction in a variant pathogenicity prediction task, relating fundamental transfer learning ideas to domain-specific bioinformatics problems. Second, the initiative promotes fluency with common genomic variant representations and workflows by developing literacy in biological data sources and annotation tools (e.g., ClinVar and Ensembl VEP, when used for validation and standardization). Lastly, the project exemplifies professional software engineering techniques (typed code, reproducible environments, modular architecture, unit testing) applied to a machine learning context, preparing the student for collaborative research environments and industry-scale data science roles.

---

## References

Landrum, M. J., Lee, J. M., Benson, M., Brown, G. R., Chao, C., Chitipiralla, S., and others (2018). "ClinVar: improving access to variant interpretations and supporting evidence." *Nucleic Acids Research*, 46(D1), D1062–D1067.

McLaren, W., Gil, L., Hunt, S. E., Riat, H. S., Ritchie, G. R., Thormann, A., and others (2016). "The Ensembl Variant Effect Predictor." *Genome Biology*, 17(1), 122.

---

## Resources

**Data sources:**
- Public curated coding-variant pathogenicity dataset (Dr. Fan’s group; shared via Dylan Tan)
- ClinVar (optional upstream/validation): https://www.ncbi.nlm.nih.gov/clinvar/

**Tools and libraries:**
- scikit-learn: https://scikit-learn.org/ (machine learning models and evaluation)
- Streamlit: https://streamlit.io/ (web application framework)
- PyTorch: https://pytorch.org/ (neural network implementation, optional)
- Ensembl VEP (optional standardization/validation): https://www.ensembl.org/info/docs/tools/vep/
- ESM (protein language models, including ESM2): https://github.com/facebookresearch/esm
