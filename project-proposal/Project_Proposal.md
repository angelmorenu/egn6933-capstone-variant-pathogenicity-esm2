---
geometry: margin=1in
fontsize: 12pt
linestretch: 1.15
header-includes:
	- "\\usepackage{ragged2e}"
	- "\\justifying"
---

# M.S. Applied Data Science: Capstone Project Proposal

**Due Date:** January 25, 2026  
**Student Name:** Angel Morenu  
**Course:** EGN 6933 – Project in Applied Data Science  
**Project Type:** Individual

---

## 1. Project Title & Team Members

**Project Name:** Machine Learning Classification of Pathogenic vs. Benign Missense Variants Using Protein Language Model Embeddings

**Team Lead:** Angel Morenu (Individual Project)

---

## 2. Problem Statement & Impact

One of the main challenges in precision medicine is comprehending the impact of human genetic variation on health. Single amino acid mutations in protein-coding areas, or missense variations, can change the stability of proteins, interfere with molecular interactions, and disturb biological processes. These variations are frequently associated with inherited problems and uncommon diseases. However, it is costly and time-consuming to assess the impact of these genetic modifications in the laboratory. The necessity for computational tools that can assist prioritize which varieties need more research has been prompted by this.

This capstone project focuses on a practical, prediction-based machine learning task. It involves building a binary classifier that labels missense variants as either pathogenic or benign. The model will use embeddings from pretrained protein language models as input. The goal is not to uncover the full biological mechanism behind each variant, but to develop a reproducible machine learning workflow that can effectively guide future lab experiments and clinical evaluations.

The stakeholders for this work include rare disease researchers looking to speed up variant interpretation, clinical genomics professionals who need computational tools to assist with decision-making, and computational biologists building scalable variant analysis pipelines. The project contributes to the broader mission of precision medicine by helping reduce the manual effort involved in variant classification and ensuring that limited lab resources are directed toward the most promising leads.

From an ethical and societal standpoint, the project uses only public, de-identified genetic data from ClinVar (Landrum et al., 2018). It makes clear that the model’s predictions are not to be used for clinical diagnosis without further lab testing and expert review, and that clinical pathogenicity categories follow widely used standards (Richards et al., 2015). The project also addresses common data challenges, such as class imbalance and uncertain labels, by removing ambiguous cases and using performance metrics that remain meaningful even when classes are uneven.

This work combines applied data science methods, including machine learning, statistical analysis, and reproducible coding practices, with genomics and healthcare workflows. It demonstrates how computational tools can play a meaningful role in supporting real-world clinical research and decision-making.

---

## 3. Data Acquisition & Viability

The primary data source for this project is ClinVar (https://www.ncbi.nlm.nih.gov/clinvar/), a public archive of variant interpretations and supporting evidence (Landrum et al., 2018). The project will download ClinVar releases and construct a labeled training table by applying a strict label policy based on ClinVar clinical significance categories and filtering decisions that are fully documented and reproducible.

To restrict scope and reduce heterogeneity, the dataset will include missense variants only. Variant consequences will be derived using Ensembl Variant Effect Predictor (VEP) (McLaren et al., 2016), and only variants annotated as missense (e.g., VEP consequence `missense_variant`) will be retained.

To ensure a strong training signal, clinical labels will be clearly defined. Variants labeled as Pathogenic or Likely Pathogenic will be grouped as pathogenic, while those labeled Benign or Likely Benign will be assigned to the benign class. Variants labeled as Uncertain Significance or those with conflicting submissions will be excluded to avoid label noise and maintain a clear separation between positive and negative classes.

Only missense coding variants will be included. Ensembl VEP will be used to verify and standardize this consequence definition across the dataset.

The features used for model training will come from protein language model embeddings. The project will compute and cache ESM2-style embeddings from protein sequence context to produce fixed-length vectors suitable for use with traditional machine learning models (Rives et al., 2021; Lin et al., 2023).

Data handling will follow software engineering best practices. The labeled dataset will be saved in versioned Parquet files that record key metadata such as the ClinVar release date, VEP annotation settings, the embedding model/version and dimensionality, split seeds/grouping keys, and label filtering choices. All processing steps will be fully scripted and deterministic, ensuring that others can recreate the dataset exactly from the original ClinVar source. Embedding files will be saved in standardized formats to support efficient and reproducible model training without needing to recompute embeddings every time.

ClinVar is public and de-identified, and the project does not involve participant recruitment or private health information. The project will be executed in accordance with university research and data-handling best practices. The computational workflow will be thoroughly documented and sufficient implementation detail will be provided that other researchers can reproduce the analysis using public data sources. The project is structured as a research proof-of-concept and explicitly disclaims clinical utility; model predictions are intended for research-driven variant prioritization rather than clinical diagnostic use.

---

## 4. Technical Execution & Complexity

The complete pipeline for this project consists of seven key computational steps. First, ClinVar data is downloaded and processed into a consistent variant representation, and clinical labels are applied and filtered. Second, VEP is used to annotate consequences and enforce a missense-only dataset, and basic quality checks are conducted. Third, protein language model embedding features (ESM2-style) are generated and cached, then saved to disk in a consistent format. Fourth, the dataset is split into training, validation, and test sets using gene- or protein-aware stratification to avoid data leakage. This ensures that variants from the same gene or protein are not shared across the splits; when multiple identifiers exist, grouping will prioritize protein or gene IDs and fall back to transcript IDs when necessary. Fifth, several classification models are trained and compared using robust statistical testing. Sixth, the best-performing model is selected, calibrated, and validated on held-out data. Seventh, the final model is deployed as both an interactive web application and a command-line tool.

The project explores three different types of classifiers, all trained on the same set of embeddings. Logistic Regression will be used as a simple and interpretable baseline. Random Forest will offer a more flexible, ensemble-based alternative that can model complex patterns. Additionally, a shallow Multi-Layer Perceptron will serve as a neural network baseline, helping evaluate whether increased model complexity leads to better predictions.

This project addresses a number of complex technical issues suitable for a capstone at the master's level. It utilizes pre-trained protein language models (ESM2) to classify the pathogenicity of genetic variations through transfer learning, requiring a solid understanding of representation learning and its application to biological data. Additionally, it employs an evaluation technique that prevents overly optimistic performance findings by carefully avoiding data leakage, ensuring gene- or protein-level isolation between the training and testing sets. Additionally, this study uses stringent statistical techniques to verify whether performance differences are statistically significant, such as bootstrap or permutation tests for AUPRC and DeLong tests for comparing AUROC values. A complete machine learning system that is prepared for practical implementation is the end result. This includes serializing the trained model, managing the computing environment through Conda (and optionally Docker), and providing interfaces for users through both a web app and command-line tool.

Reproducibility is a central focus throughout the project. All random seeds are fixed to ensure that results can be exactly replicated. Configuration files track all hyperparameters and filtering criteria so that alternative analyses can be reproduced just by changing configuration values. The train, validation, and test splits are saved to disk, allowing for consistent reuse in future experiments or by other researchers. The computing environment is captured in Conda environment files, which document precise package versions and dependencies. The codebase is structured professionally, with typed Python modules for each major step: data loading, feature engineering, model training, evaluation, and deployment. Code formatting and linting are enforced using tools like Black and Ruff. To support long-term reliability, key parts of the pipeline are covered by unit tests that help catch bugs and prevent future regressions.

---

## 5. Deployment Plan: "The App"

The final deliverable includes a user-facing application that scores missense variants and returns calibrated pathogenicity probabilities. The application will be implemented as a Streamlit web application and will support two primary workflows: (1) single-variant scoring via a simple input form (e.g., variant identifier fields used to match a record with a precomputed embedding, and/or transcript/protein identifier when available) and (2) batch scoring via CSV upload. For batch input, the app will return a ranked “variant prioritization” table (sorted by predicted pathogenicity probability), with downloadable results suitable for downstream review.

To support transparent benchmarking, the Streamlit app will also include a “Model Performance” view that displays precomputed evaluation artifacts from the held-out gene/protein-disjoint test set (ROC and precision-recall curves with AUROC/AUPRC values and selected operating thresholds). These performance plots will be generated during the offline evaluation phase and bundled with the trained model release, ensuring that the displayed metrics are reproducible and not influenced by user-supplied inputs.

Additionally, a command-line interface will be provided for batch scoring of larger variant tables (CSV or VCF-derived tabular exports). The CLI will output a scored, ranked file and optionally export evaluation plots for reporting. The complete set of end-to-end project deliverables includes: (1) a reproducible pipeline with associated configuration files that can rebuild the processed dataset artifacts and models from public ClinVar and VEP (when used for consequence filtering); (2) trained model artifacts (serialized scikit-learn or PyTorch models) accompanied by an evaluation report documenting performance metrics, statistical tests, and error analysis; (3) a Streamlit web application for interactive scoring and demonstration; and (4) a command-line tool for batch processing and integration into automated workflows.

---

## 6. Statistical Analysis & Evaluation

Classification performance will be evaluated using two primary metrics selected for their appropriateness to imbalanced binary classification problems. Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) are both threshold-independent metrics that remain valid across different class imbalance ratios. Secondary metrics will include precision, recall, F1 score, and balanced accuracy to provide a comprehensive view of model behavior on both the majority and minority classes.

The experimental design incorporates leakage-aware splitting to avoid artificially inflated performance estimates. Specifically, the dataset will be divided into training, validation, and test subsets using gene- or protein-aware stratification, where all variants mapping to the same gene/protein are assigned to a single split. This strategy reduces the risk of leakage wherein highly related variants share protein context and embeddings across train and test, inflating apparent performance. As a sensitivity analysis, the gene/protein assignment strategy will be varied across multiple random seeds to test whether performance estimates remain stable across different partitions.

Using the same embedding feature sets, the project will train and compare three complementary models: Random Forest, Logistic Regression, and possibly a shallow Multi-Layer Perceptron. Performance comparisons will employ rigorous statistical tests appropriate for paired predictions on the same test set. The DeLong test will be used to compare AUROC values between models, while paired bootstrap and permutation tests will compare AUPRC values. Bootstrapped 95% confidence intervals will be constructed for AUROC and AUPRC on the held-out test set, providing not merely point estimates but ranges of plausible performance values.

Because the dataset is expected to exhibit class imbalance (often more benign than pathogenic variants), the project will employ class weighting during model training to penalize misclassification of the minority (pathogenic) class more heavily. Decision thresholds will be tuned on the validation set to optimize a desired trade-off between sensitivity and specificity. Performance will be reported as precision-recall curves to clearly visualize the trade-off between precision and recall across different operating points.

The final trained model will be calibrated using the validation set (via Platt scaling or isotonic regression) to ensure that predicted probabilities reflect true conditional class probabilities. Calibration will be assessed using calibration curves and the Brier score, enabling downstream users to interpret predicted probabilities as genuine probability estimates rather than arbitrary model scores.

Supporting model interpretability analyses will provide insights into model predictions and feature importance. For tree-based models, feature importance scores will identify which embedding dimensions most strongly influence predictions. For all models, calibrated probability outputs will serve as risk scores for downstream decision-making. Lightweight attribution analysis will be performed for a small set of representative variants, highlighting which parts of the protein sequence context and/or embedding dimensions most influence model predictions. If time permits, SHAP (SHapley Additive exPlanations) analysis will provide model-agnostic, local and global explanations of model behavior within the embedding feature space.

If, during exploratory analysis, the positive class (pathogenic variants) is discovered to be too small after strict filtering to support reliable model training, the filtering criteria will be relaxed iteratively (e.g., expanding the label set to include Likely Pathogenic variants) while maintaining high confidence by continuing to exclude ambiguous records (VUS, conflicting interpretations).

---

## 7. Project Timeline & Milestones

The project is organized into seven phases across fifteen weeks.

During weeks 1–4, the focus is on data acquisition and preparation. Activities include downloading ClinVar, defining and applying label filtering rules, annotating and filtering to missense variants via Ensembl VEP, conducting exploratory data analysis, and designing the gene/protein-aware split strategy.

Weeks 5–8 focus on feature engineering and baseline model development. Embedding vectors will be generated for all variants. Logistic Regression and Random Forest classifiers will be trained on the embedding features, and initial performance estimates using AUROC and AUPRC will be computed. Iterative analysis of class imbalance effects will inform threshold tuning and the selection of decision operating points.

Weeks 9–12 are dedicated to model refinement and rigorous statistical evaluation. If time permits, a shallow Multi-Layer Perceptron will be implemented as an optional extension. Final model evaluation will include bootstrapped confidence intervals, paired statistical tests comparing model performance, and detailed error analysis to understand which types of variants are misclassified and why.

Weeks 13–15 focus on deployment, documentation, and presentation. The Streamlit web application and command-line interface will be implemented. Project documentation, inline code comments, and the final capstone report will be completed. A project presentation and interactive demo will be prepared for peer and instructor review.

This is an individual capstone project. Feedback will be iteratively solicited from the instructor and advisor throughout the semester, and a final project presentation will be delivered to peers for collaborative review.

---

## 8. New Knowledge Acquisition

This project offers hands-on experience with a number of modern computational genomics and machine learning methods. It shows how pretrained protein language models (ESM2-style embeddings) may be used practically for feature extraction in a variant pathogenicity prediction task, relating fundamental transfer learning ideas to domain-specific bioinformatics problems. Second, the initiative promotes fluency with common genomic variant representations and workflows by developing literacy in biological data sources and annotation tools (ClinVar and Ensembl VEP, when used for validation and standardization). Lastly, the project exemplifies professional software engineering techniques (typed code, reproducible environments, modular architecture, unit testing) applied to a machine learning context, preparing the student for collaborative research environments and industry-scale data science roles.

---

## References

Landrum, M. J., Lee, J. M., Benson, M., Brown, G. R., Chao, C., Chitipiralla, S., and others (2018). "ClinVar: improving access to variant interpretations and supporting evidence." *Nucleic Acids Research*, 46(D1), D1062–D1067.

McLaren, W., Gil, L., Hunt, S. E., Riat, H. S., Ritchie, G. R., Thormann, A., and others (2016). "The Ensembl Variant Effect Predictor." *Genome Biology*, 17(1), 122.

Richards, S., Aziz, N., Bale, S., Bick, D., Das, S., Gastier-Foster, J., Grody, W. W., Hegde, M., Lyon, E., Spector, E., Voelkerding, K., Rehm, H. L., and others (2015). "Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology." *Genetics in Medicine*, 17(5), 405–424.

Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., and others (2021). "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." *Proceedings of the National Academy of Sciences*, 118(15), e2016239118.

Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., and others (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 379(6637), 1123–1130.

---

## Acknowledgements

This work benefits from guidance and reference materials shared by Dylan Tan during early exploration and prototyping.

---

## Resources

**Data sources:**
- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- Ensembl VEP (missense consequence annotation): https://www.ensembl.org/info/docs/tools/vep/

**Tools and libraries:**
- scikit-learn: https://scikit-learn.org/ (machine learning models and evaluation)
- Streamlit: https://streamlit.io/ (web application framework)
- PyTorch: https://pytorch.org/ (neural network implementation, optional)
- Ensembl VEP (optional standardization/validation): https://www.ensembl.org/info/docs/tools/vep/
- ESM (protein language models, including ESM2): https://github.com/facebookresearch/esm
