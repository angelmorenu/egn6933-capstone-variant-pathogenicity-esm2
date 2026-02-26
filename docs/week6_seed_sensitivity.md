# Week 6 — Split Seed Sensitivity (RF baseline)

**Setup** 
- Dataset: `data/processed/week4_curated_dataset_seed{13,37}.parquet` (n=5000)
- Model: Random Forest (`--rf-max-depth 4 --rf-n-estimators 200`)
- Evaluation: gene-disjoint splits; metrics reported for train/val/test
- Bootstrap: 1000 iterations on **test**
- Split breakdown (rows): train=4000 (80%), val=500 (10%), test=500 (10%)

## Seed 13 
- Split sizes: train=4000, val=500, test=500 (80/10/10)
- Test prevalence: 0.582
- Test AUROC: 0.9344 (bootstrap CI: [0.9112, 0.9545])
- Test AUPRC: 0.9502 (bootstrap CI: [0.9299, 0.9677])
- Test Brier: bootstrap CI: [0.1271, 0.1443]
- Test log loss: bootstrap CI: [0.4259, 0.4665]
- Report: `results/baseline_rf_seed13_bootstrap.json`

## Seed 37 
- Split sizes: train=4000, val=500, test=500 (80/10/10)
- Test prevalence: 0.588
- Test AUROC: 0.9299 (bootstrap CI: [0.9074, 0.9496])
- Test AUPRC: 0.9473 (bootstrap CI: [0.9253, 0.9653])
- Test Brier: bootstrap CI: [0.1270, 0.1458]
- Test log loss: bootstrap CI: [0.4253, 0.4676]
- Report: `results/baseline_rf_seed37_bootstrap.json`

## Summary 
- Seed effect on test AUROC: 0.9344 → 0.9299 (Δ = -0.0045)
- Seed effect on test AUPRC: 0.9502 → 0.9473 (Δ = -0.0029)
- The two seeds produce very similar test performance; bootstrap CIs overlap substantially.

## Interpretation (what this means) 
- **What “split seed sensitivity” measures:** The split seed changes which *genes* land in train vs. test (while still keeping genes disjoint). If performance changes a lot across seeds, the reported test score is fragile and may depend heavily on which genes happened to be held out.
- **What these results indicate:** Across two different gene-disjoint splits (seed 13 vs. 37), the RF’s test AUROC/AUPRC point estimates change only slightly (ΔAUROC≈-0.0045; ΔAUPRC≈-0.0029), and the bootstrap confidence intervals overlap. This suggests the baseline ranking performance is **reasonably stable** to the particular gene partition (at least for these two seeds on this 5k pilot).
- **Why this matters:** A stable score across seeds increases confidence that the baseline is capturing signal that generalizes to *unseen genes*, rather than exploiting quirks of a single lucky split.

## Importance / caveats
- **Why AUROC is primary here:** AUROC measures ranking quality and is less sensitive to the exact decision threshold; it is commonly used when class balance can vary across splits (as it does here due to gene-disjoint partitioning).
- **Why still report AUPRC:** Because this dataset is curated and prevalence is not population-realistic, AUPRC is best treated as a complementary ranking summary rather than a direct estimate of real-world PPV.
- **What this does *not* prove:** Two seeds is a quick robustness check, not a full uncertainty analysis over all plausible gene partitions. If needed, extend to more seeds (e.g., 10+) and summarize mean±sd (or quantiles) of test AUROC.
