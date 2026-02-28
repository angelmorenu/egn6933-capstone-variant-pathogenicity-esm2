"""
Generate presentation-ready figures: ROC curves with calibrated and uncalibrated predictions.
This script creates ROC curves for the RF model showing both calibrated and uncalibrated approaches
on validation and test sets.

Usage:
conda run -n egn6933-variant-embeddings python scripts/generate_presentation_figures.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, roc_auc_score
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

# Train RF model (matching Week 5 baseline configuration)
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

# Get uncalibrated predictions
scores_val_uncal = rf.predict_proba(embeddings[idx_val])[:, 1]
scores_test_uncal = rf.predict_proba(embeddings[idx_test])[:, 1]

# Calibrate on validation set, apply to test
calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
calibrated_rf.fit(embeddings[idx_val], labels[idx_val])

# Get calibrated predictions (on validation and test, fitted on val)
scores_val_cal = calibrated_rf.predict_proba(embeddings[idx_val])[:, 1]
scores_test_cal = calibrated_rf.predict_proba(embeddings[idx_test])[:, 1]

# Compute ROC curves
fpr_v_u, tpr_v_u, _ = roc_curve(labels[idx_val], scores_val_uncal)
fpr_v_c, tpr_v_c, _ = roc_curve(labels[idx_val], scores_val_cal)
fpr_t_u, tpr_t_u, _ = roc_curve(labels[idx_test], scores_test_uncal)
fpr_t_c, tpr_t_c, _ = roc_curve(labels[idx_test], scores_test_cal)

# Compute AUCs
auc_v_u = roc_auc_score(labels[idx_val], scores_val_uncal)
auc_v_c = roc_auc_score(labels[idx_val], scores_val_cal)
auc_t_u = roc_auc_score(labels[idx_test], scores_test_uncal)
auc_t_c = roc_auc_score(labels[idx_test], scores_test_cal)

print(f"Validation AUROC: Uncal={auc_v_u:.4f}, Cal={auc_v_c:.4f}")
print(f"Test AUROC: Uncal={auc_t_u:.4f}, Cal={auc_t_c:.4f}")

# Create figure
fig, ax = plt.subplots(figsize=(9, 7))

# Plot ROC curves
ax.plot(fpr_v_u, tpr_v_u, label=f"val (uncal): AUC={auc_v_u:.4f}", linewidth=2)
ax.plot(fpr_v_c, tpr_v_c, label=f"val (cal): AUC={auc_v_c:.4f}", linewidth=2)
ax.plot(fpr_t_u, tpr_t_u, label=f"test (uncal): AUC={auc_t_u:.4f}", linewidth=2, linestyle='--')
ax.plot(fpr_t_c, tpr_t_c, label=f"test (cal): AUC={auc_t_c:.4f}", linewidth=2, linestyle='--')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='random')

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves (Calibrated vs Uncalibrated)", fontsize=13, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout()

# Save figure
output_path = _REPO_ROOT / "results/presentation_roc_cal_vs_uncal.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
print(f"Saved ROC curve to {output_path}")

plt.close()

print("Done!")
