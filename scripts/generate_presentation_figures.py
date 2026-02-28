"""
Generate presentation-ready figures: ROC curves for baseline models.
This script creates ROC curves for LogReg and RF test performance.

Usage:
conda run -n egn6933-variant-embeddings python scripts/generate_presentation_figures.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Load data
data_path = _REPO_ROOT / "data/processed/week4_curated_dataset.parquet"
df = pd.read_parquet(data_path)

print(f"Loaded data with shape {df.shape}")

# Extract features, labels, and splits
embeddings = np.array(df['embedding'].tolist())
labels = df['label'].values
splits = df['split'].values

# Create split indices
idx_train = np.where(splits == 'train')[0]
idx_val = np.where(splits == 'val')[0]
idx_test = np.where(splits == 'test')[0]

print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

# ============ Train Logistic Regression ============
logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=0, class_weight='balanced'))
])
logreg.fit(embeddings[idx_train], labels[idx_train])
scores_test_logreg = logreg.predict_proba(embeddings[idx_test])[:, 1]
auc_test_logreg = roc_auc_score(labels[idx_test], scores_test_logreg)

# ============ Train Random Forest ============
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=0,
    class_weight='balanced',
    n_jobs=-1
)
rf.fit(embeddings[idx_train], labels[idx_train])
scores_test_rf = rf.predict_proba(embeddings[idx_test])[:, 1]
auc_test_rf = roc_auc_score(labels[idx_test], scores_test_rf)

print(f"Test AUROC: LogReg={auc_test_logreg:.4f}, RF={auc_test_rf:.4f}")

# ============ Compute ROC curves ============
fpr_logreg, tpr_logreg, _ = roc_curve(labels[idx_test], scores_test_logreg)
fpr_rf, tpr_rf, _ = roc_curve(labels[idx_test], scores_test_rf)

# ============ Create figure ============
fig, ax = plt.subplots(figsize=(9, 7))

# Plot ROC curves with model names in legend
ax.plot(fpr_logreg, tpr_logreg, label=f"Logistic Regression: AUC={auc_test_logreg:.4f}", linewidth=2.5, color='#1f77b4')
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (shallow): AUC={auc_test_rf:.4f}", linewidth=2.5, color='#ff7f0e')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random classifier')

ax.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
ax.set_title("ROC Curves: Baseline Model Comparison (Test Set)", fontsize=13, fontweight='bold')
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout()

# Save figure
output_path = _REPO_ROOT / "results/presentation_roc_baselines.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
print(f"Saved ROC curve to {output_path}")

plt.close()

print("Done!")
