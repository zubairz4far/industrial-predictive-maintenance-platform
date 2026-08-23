from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET, chronological_split, generate_synthetic_telemetry

FALSE_NEGATIVE_COST = 12
FALSE_POSITIVE_COST = 1


@dataclass(frozen=True)
class Metrics:
    pr_auc: float
    roc_auc: float
    brier: float
    recall: float
    precision: float
    threshold: float
    false_positives: int
    false_negatives: int
    business_cost: int


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Metrics:
    predictions = probabilities >= threshold
    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return Metrics(
        pr_auc=float(average_precision_score(y_true, probabilities)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        brier=float(brier_score_loss(y_true, probabilities)),
        recall=float(recall),
        precision=float(precision),
        threshold=float(threshold),
        false_positives=fp,
        false_negatives=fn,
        business_cost=int(FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp),
    )


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.03, 0.60, 58)
    scored = [_metrics(y_true, probabilities, float(t)) for t in candidates]
    best = min(scored, key=lambda m: (m.business_cost, -m.recall, m.threshold))
    return best.threshold


def evaluate(seed: int = 42) -> dict[str, object]:
    frame = generate_synthetic_telemetry(seed=seed)
    train, dev, test = chronological_split(frame)

    x_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    x_dev, y_dev = dev[FEATURES], dev[TARGET].to_numpy()
    x_test, y_test = test[FEATURES], test[TARGET].to_numpy()

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=0.8,
            class_weight="balanced",
            random_state=seed,
        ),
    }

    results: dict[str, Metrics] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        dev_probability = model.predict_proba(x_dev)[:, 1]
        threshold = _select_threshold(y_dev, dev_probability)
        test_probability = model.predict_proba(x_test)[:, 1]
        results[name] = _metrics(y_test, test_probability, threshold)

    baseline = results["logistic_regression"]
    candidate = results["hist_gradient_boosting"]
    promote = (
        candidate.business_cost <= baseline.business_cost
        and candidate.pr_auc >= baseline.pr_auc
        and candidate.recall >= 0.60
    )

    return {
        "dataset": {
            "kind": "deterministic synthetic telemetry",
            "seed": seed,
            "rows": len(frame),
            "machines": int(frame["machine_id"].nunique()),
            "positive_rate": float(frame[TARGET].mean()),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "test_rows": len(test),
        },
        "features": FEATURES,
        "target": TARGET,
        "cost_model": {
            "false_negative": FALSE_NEGATIVE_COST,
            "false_positive": FALSE_POSITIVE_COST,
        },
        "models": {name: asdict(metrics) for name, metrics in results.items()},
        "promotion": {
            "candidate": "hist_gradient_boosting",
            "baseline": "logistic_regression",
            "decision": "PROMOTE" if promote else "REJECT",
            "rule": "candidate cost <= baseline cost, PR-AUC >= baseline, recall >= 0.60",
        },
    }


def write_report(path: str | Path, seed: int = 42) -> dict[str, object]:
    report = evaluate(seed=seed)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
