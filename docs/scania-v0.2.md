# Scania APS v0.2 Evaluation Note

## Scope

v0.2 upgrades the repository from a deterministic synthetic evaluation fixture to the public
**APS Failure at Scania Trucks** benchmark from the UCI Machine Learning Repository (dataset 421,
DOI `10.24432/C51S51`). The UCI description states that the data was collected from heavy Scania
trucks in everyday usage.

The release is designed to demonstrate defensible offline model-selection mechanics. It does not
claim a deployed predictive-maintenance system.

## Data contract

The official release used by CI contains:

- 60,000 training examples;
- 16,000 official test examples;
- one binary class column (`neg` / `pos`);
- 170 anonymized numeric operational features;
- missing values encoded as `na` in the source CSVs.

The downloader verifies the exact source files before parsing:

```text
aps_failure_training_set.csv
bb484302e3a3a1c8ef5e1f0129c4dc7cbd58f350867f95b575461ca21ab6b9da

aps_failure_test_set.csv
2cdf6f7661c7b4c63333c93cdec36a3a82350176b604a2312cf82799fb2712f3
```

The raw files are cached locally/inside CI but are not committed to the repository.

## Leakage controls

The official test set is never used for fitting, calibration, threshold selection, or promotion-rule
construction. The official training set is divided with a deterministic stratified split:

| Partition | Rows | Purpose |
|---|---:|---|
| Fit | 42,000 | fit model parameters |
| Calibration | 9,000 | fit sigmoid/Platt calibration |
| Development | 9,000 | select the cost-minimizing decision threshold |
| Official test | 16,000 | final frozen evaluation only |

Seed: `42`.

The UCI files do not expose reliable event timestamps, so a chronological split would be an
unsupported claim. Stratification is used instead to preserve the rare positive class during the
three-way split of the official training data.

## Models

### Logistic Regression baseline

- median imputation;
- missingness indicators;
- standardization;
- balanced class weights;
- `C=0.2`;
- `liblinear` solver.

### HistGradientBoosting candidate

- native missing-value handling;
- balanced class weights;
- learning rate `0.08`;
- maximum 140 boosting iterations;
- 31 maximum leaf nodes;
- minimum 40 samples per leaf;
- L2 regularization `1.0`;
- early stopping enabled.

Both models receive independent one-dimensional sigmoid/Platt calibration fitted only on the
calibration partition.

## Threshold selection and cost

Thresholds are not fixed at 0.5. Each model's threshold is selected only on the development
partition by minimizing the original challenge cost:

```text
cost = 10 * false_positives + 500 * false_negatives
```

This makes missed APS failures fifty times more expensive than unnecessary inspections.

The threshold search evaluates every unique calibrated development probability and chooses the
minimum-cost threshold, breaking equal-cost ties toward higher recall.

## Frozen promotion rule

The candidate is promoted only when all three conditions hold:

1. candidate test challenge cost is at most 95% of baseline test challenge cost;
2. candidate test PR-AUC is at least baseline PR-AUC;
3. candidate test recall is at least baseline recall.

The rule is part of the code and the frozen JSON evidence rather than being rewritten after observing
the result.

## Measured result

| Model | PR-AUC | ROC-AUC | Brier | Recall | Precision | FP | FN | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.793775 | 0.980954 | 0.009523 | 0.880000 | 0.552764 | 267 | 45 | 25,170 |
| HistGradientBoosting | 0.871066 | 0.991976 | 0.007348 | 0.925333 | 0.524962 | 314 | 28 | 17,140 |

Decision: **PROMOTE HistGradientBoosting**.

The challenge cost falls by approximately **31.9%**. The candidate produces 47 more false positives
but 17 fewer false negatives; the asymmetric cost makes that trade favorable.

Exact evidence lives in `evals/results/v0.2_scania_aps.json`.

## Reproducibility contract

CI pins the Python and benchmark dependency versions, downloads the hash-pinned UCI files, executes
the benchmark, and performs a byte-level diff between the generated JSON and the committed evidence.
A change in dataset bytes, preprocessing, model behavior, thresholding, metrics, runtime versions, or
promotion output therefore causes the benchmark job to fail until the evidence is deliberately
reviewed and updated.

## Limitations

- Feature names are anonymized by the source dataset, so engineering interpretation is limited.
- This is a static offline benchmark and does not establish temporal generalization.
- The cost weights are the published challenge weights, not a measured maintenance budget.
- The benchmark does not establish real-world downtime reduction, ROI, latency, availability, or
  alert quality in a specific fleet.
- A production milestone would require live/representative telemetry, drift monitoring, serving,
  alert integration, maintenance feedback, SLOs, and environment-backed evidence.

## Attribution

Dataset: **APS Failure at Scania Trucks**, UCI Machine Learning Repository, dataset 421,
DOI `10.24432/C51S51`.

The UCI dataset page lists the dataset license as **CC BY 4.0**. Dataset files are not redistributed
inside this repository; the benchmark retrieves them from UCI and preserves their independent
licensing.
