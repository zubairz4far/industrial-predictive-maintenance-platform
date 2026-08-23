# Industrial Predictive Maintenance Platform

[![CI](https://github.com/zubairz4far/industrial-predictive-maintenance-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/industrial-predictive-maintenance-platform/actions/workflows/ci.yml)

Leakage-aware predictive-maintenance ML project for machine telemetry. The v0.1 milestone builds a deterministic synthetic telemetry fixture, performs chronological train/dev/test evaluation, freezes decision thresholds on development data, and compares a transparent logistic baseline with a nonlinear gradient-boosting candidate under an explicit failure-cost model.

## Status

**v0.1 — evaluated failure-risk baseline.**

This repository currently demonstrates evaluation mechanics and software quality. It does **not** claim real factory accuracy, live sensor integration, downtime reduction, or production deployment.

## Problem

Predict whether a machine will fail in the next 24 hours from dispatch-time telemetry:

- temperature;
- vibration;
- pressure;
- RPM;
- load;
- lubricant quality;
- hours since maintenance;
- machine age.

The forward-looking target (`failure_next_24h`) is never included in the feature set.

## Evaluation design

The synthetic fixture contains **7,200 telemetry rows from 40 machines** with a **6.10% positive rate**. Data is split by timestamp, not randomly:

```text
oldest                                                    newest
|---------------- train ----------------|--- dev ---|--- test ---|
                 4,320                    1,440       1,440
```

Each model selects its decision threshold on the **development set only**. The threshold is then frozen before test evaluation.

Business cost:

- false negative: **12 units**;
- false positive: **1 unit**.

Promotion rule: candidate business cost must be no worse than baseline, PR-AUC must be at least baseline, and test recall must be at least 0.60.

## Measured v0.1 synthetic result

| Model | PR-AUC ↑ | ROC-AUC ↑ | Brier ↓ | Recall ↑ | Precision ↑ | Cost ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.151 | **0.757** | 0.138 | **0.486** | 0.099 | 391 |
| HistGradientBoosting | **0.187** | 0.748 | **0.054** | 0.405 | **0.115** | **380** |

**Promotion decision: REJECT.** The nonlinear candidate improved PR-AUC, calibration, precision, and configured cost, but missed the predeclared 0.60 recall floor. The project intentionally preserves that rejection rather than changing the rule after seeing test results.

Machine-readable evidence: [`evals/results/v0.1_synthetic.json`](evals/results/v0.1_synthetic.json).

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"

ruff check .
pytest -q
predictive-maintenance evaluate --output evals/results/v0.1_synthetic.json
```

The evaluation is deterministic with seed 42.

## Repository structure

```text
src/predictive_maintenance/
  data.py          deterministic telemetry + chronological split
  evaluation.py    models, threshold selection, metrics, promotion rule
  cli.py           reproducible evaluation command

tests/
  test_data.py
  test_evaluation.py

evals/results/
  v0.1_synthetic.json

.github/workflows/ci.yml
```

## CI contract

GitHub Actions installs the project on Python 3.12, runs Ruff, executes the test suite, and regenerates a CI evaluation report. Local validation before the initial push passed **4/4 tests**; Ruff was not available in the local execution environment, so the repository does not claim local lint evidence.

## Next milestone

v0.2 should move from a deterministic synthetic fixture to a public predictive-maintenance dataset, add time-aware feature windows, calibrate probabilities, compare cost-sensitive models, and preserve an immutable evaluation report. Production serving should come only after the offline evidence is credible.

## License

MIT. Third-party datasets and libraries retain their own licenses.
