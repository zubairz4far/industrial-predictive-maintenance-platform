# Industrial Predictive Maintenance Platform

[![CI](https://github.com/zubairz4far/industrial-predictive-maintenance-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/industrial-predictive-maintenance-platform/actions/workflows/ci.yml)

Evaluated predictive-maintenance ML with explicit cost-sensitive model promotion, probability
calibration, immutable benchmark evidence, dataset integrity checks, and CI reproduction.

## Status

**v0.2 — real-data Scania APS benchmark.**

The current release evaluates failure classification on the UCI **APS Failure at Scania Trucks**
dataset. The data was collected from heavy Scania trucks in everyday usage and contains an official
60,000-row training set and 16,000-row test set with 170 anonymized operational features and missing
values.

This is a rigorous offline benchmark, not a claim of a live factory or fleet deployment.

## Headline result

The official 16,000-row test set is held out until the final evaluation. The nonlinear candidate is
promoted because it satisfies the predeclared promotion rule and materially reduces the original
challenge cost.

| Model | PR-AUC ↑ | ROC-AUC ↑ | Brier ↓ | Recall ↑ | Precision ↑ | FP | FN | Challenge cost ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.794 | 0.981 | 0.00952 | 0.880 | **0.553** | **267** | 45 | 25,170 |
| HistGradientBoosting | **0.871** | **0.992** | **0.00735** | **0.925** | 0.525 | 314 | **28** | **17,140** |

**Promotion decision: PROMOTE.** HistGradientBoosting reduces the configured challenge cost by
**31.9%**, improves PR-AUC by **0.077**, and improves recall by **4.5 percentage points**. It accepts
47 additional false positives to avoid 17 additional false negatives; under the original challenge
costs, a false negative costs 500 while a false positive costs 10.

Machine-readable evidence:
[`evals/results/v0.2_scania_aps.json`](evals/results/v0.2_scania_aps.json).

## Evaluation protocol

The UCI release does not provide trustworthy event timestamps, so v0.2 does **not** pretend to use a
chronological split. Instead:

```text
official training set: 60,000 rows
|---------- fit 42,000 ----------|-- calibration 9,000 --|-- dev 9,000 --|

untouched official test set: 16,000 rows
|----------------------------- final test -----------------------------|
```

The protocol is fixed before test scoring:

1. stratified 70/15/15 split of the official training set with seed 42;
2. fit each model only on the 42,000-row fit partition;
3. fit sigmoid/Platt probability calibration only on the 9,000-row calibration partition;
4. select each decision threshold only on the 9,000-row development partition using challenge cost;
5. score the untouched official 16,000-row test set once with the frozen model/calibration/threshold;
6. promote only if candidate cost is at most 95% of baseline cost, PR-AUC is no worse, and recall is
   no worse.

The original challenge cost used here is:

- false positive / unnecessary inspection: **10**;
- false negative / missed APS failure: **500**.

See [`docs/scania-v0.2.md`](docs/scania-v0.2.md) for the full methodology and limitations.

## Dataset integrity

The raw UCI CSV files are not committed to this repository. The evaluation command downloads them
from UCI when requested and verifies the exact files before parsing:

```text
training SHA256  bb484302e3a3a1c8ef5e1f0129c4dc7cbd58f350867f95b575461ca21ab6b9da
test SHA256      2cdf6f7661c7b4c63333c93cdec36a3a82350176b604a2312cf82799fb2712f3
```

Measured training-set missing cells: **850,015**. Positive rate is **1.67%** in the official training
set and **2.34%** in the official test set, so PR-AUC and the asymmetric business cost are emphasized
rather than accuracy.

## Reproduce v0.2

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements-ci.txt
pip install -e . --no-deps

ruff check .
pytest -q

predictive-maintenance evaluate-scania \
  --download \
  --data-dir .cache/scania \
  --output /tmp/v0.2_scania_aps.json

diff -u evals/results/v0.2_scania_aps.json /tmp/v0.2_scania_aps.json
```

CI uses Python 3.12.14 and pinned benchmark dependencies. It regenerates the full Scania benchmark
and fails if the generated JSON differs from the frozen v0.2 evidence.

## Repository structure

```text
src/predictive_maintenance/
  data.py                 deterministic v0.1 synthetic fixture
  evaluation.py           v0.1 evaluation
  scania.py               UCI download, hash validation, parsing
  scania_evaluation.py    v0.2 calibration, thresholding, evaluation
  cli.py                  reproducible v0.1/v0.2 commands

tests/
  test_data.py
  test_evaluation.py
  test_scania.py

evals/results/
  v0.1_synthetic.json
  v0.2_scania_aps.json

docs/
  scania-v0.2.md

requirements-ci.txt
.github/workflows/ci.yml
```

## Previous milestone: v0.1

v0.1 remains in the repository as a deterministic CI-friendly synthetic baseline. It used 7,200
telemetry rows from 40 simulated machines, chronological splitting, and a predeclared promotion
rule. The nonlinear candidate was correctly **REJECTED** because it missed the required recall floor,
even though it improved several other metrics.

Evidence: [`evals/results/v0.1_synthetic.json`](evals/results/v0.1_synthetic.json).

Reproduce it with:

```bash
predictive-maintenance evaluate-synthetic --output /tmp/v0.1_synthetic.json
```

## Limitations

- The Scania features are anonymized, limiting domain-specific feature interpretation.
- v0.2 is an offline fixed benchmark; it does not measure temporal drift or fleet-specific drift.
- No live telemetry ingestion, model serving, alert workflow, maintenance integration, latency SLO,
  or production availability is claimed.
- The measured cost reduction is benchmark cost reduction under the published challenge weights,
  not measured monetary savings or downtime reduction.

## Dataset attribution and license

The Scania APS dataset is obtained separately from the UCI Machine Learning Repository, dataset 421,
DOI **10.24432/C51S51**. The UCI dataset page lists the dataset under **CC BY 4.0**. Raw dataset files
are intentionally not vendored here.

Repository code is MIT licensed. Third-party data and libraries retain their own licenses.
