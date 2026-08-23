from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .scania import TARGET, UCI_DATASET_ID, UCI_DATASET_URL, UCI_DOI, load_scania_dataset, sha256_file

FALSE_POSITIVE_COST = 10
FALSE_NEGATIVE_COST = 500


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
    challenge_cost: int
    cost_per_1000: float


class SigmoidCalibrator:
    """One-dimensional Platt scaling fitted on a dedicated calibration split."""

    def __init__(self) -> None:
        self._model = LogisticRegression(solver="lbfgs", max_iter=1000)

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "SigmoidCalibrator":
        self._model.fit(self._logit(probabilities), y_true)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._logit(probabilities))[:, 1]


def _binary_target(series: pd.Series) -> np.ndarray:
    mapped = series.map({"neg": 0, "pos": 1})
    if mapped.isna().any():
        raise ValueError("Scania target contains labels other than 'neg' and 'pos'")
    return mapped.to_numpy(dtype=np.int8)


def _challenge_cost(y_true: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int]:
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))
    cost = FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn
    return fp, fn, int(cost)


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="stable")
    y_sorted = y_true[order]
    p_sorted = probabilities[order]
    cumulative_tp = np.cumsum(y_sorted == 1)
    cumulative_fp = np.cumsum(y_sorted == 0)
    total_positive = int(np.sum(y_true == 1))

    group_ends = np.r_[np.flatnonzero(p_sorted[:-1] != p_sorted[1:]), len(p_sorted) - 1]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    fn = total_positive - tp
    costs = FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn
    recalls = tp / max(total_positive, 1)

    all_negative_cost = FALSE_NEGATIVE_COST * total_positive
    best_cost = int(costs.min())
    if all_negative_cost < best_cost:
        return 1.0

    candidates = np.flatnonzero(costs == best_cost)
    best_position = candidates[np.argmax(recalls[candidates])]
    return float(p_sorted[group_ends[best_position]])


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Metrics:
    predictions = (probabilities >= threshold).astype(np.int8)
    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp, fn, cost = _challenge_cost(y_true, predictions)
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
        challenge_cost=cost,
        cost_per_1000=float(cost / len(y_true) * 1000),
    )


def _split_training(
    x: pd.DataFrame, y: np.ndarray, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x_fit, x_rest, y_fit, y_rest = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    x_cal, x_dev, y_cal, y_dev = train_test_split(
        x_rest,
        y_rest,
        test_size=0.50,
        random_state=seed,
        stratify=y_rest,
    )
    return x_fit, x_cal, x_dev, y_fit, y_cal, y_dev


def _models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.2,
                solver="liblinear",
                random_state=seed,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=140,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            random_state=seed,
        ),
    }


def evaluate_scania(
    data_dir: str | Path,
    *,
    download: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    dataset = load_scania_dataset(data_dir, download=download)
    train = dataset.train
    test = dataset.test
    feature_names = [column for column in train.columns if column != TARGET]
    x_all = train[feature_names]
    y_all = _binary_target(train[TARGET])
    x_test = test[feature_names]
    y_test = _binary_target(test[TARGET])
    x_fit, x_cal, x_dev, y_fit, y_cal, y_dev = _split_training(x_all, y_all, seed)

    results: dict[str, Metrics] = {}
    for name, model in _models(seed).items():
        model.fit(x_fit, y_fit)
        raw_cal = model.predict_proba(x_cal)[:, 1]
        calibrator = SigmoidCalibrator().fit(raw_cal, y_cal)
        dev_probability = calibrator.predict(model.predict_proba(x_dev)[:, 1])
        threshold = _select_threshold(y_dev, dev_probability)
        test_probability = calibrator.predict(model.predict_proba(x_test)[:, 1])
        results[name] = _metrics(y_test, test_probability, threshold)

    baseline = results["logistic_regression"]
    candidate = results["hist_gradient_boosting"]
    promote = (
        candidate.challenge_cost <= 0.95 * baseline.challenge_cost
        and candidate.pr_auc >= baseline.pr_auc
        and candidate.recall >= baseline.recall
    )

    return {
        "release": "v0.2",
        "dataset": {
            "name": "APS Failure at Scania Trucks",
            "uci_dataset_id": UCI_DATASET_ID,
            "doi": UCI_DOI,
            "source_url": UCI_DATASET_URL,
            "kind": "operational data collected from heavy Scania trucks in everyday usage",
            "train_rows": len(train),
            "official_test_rows": len(test),
            "features": len(feature_names),
            "train_positive_rate": float(y_all.mean()),
            "test_positive_rate": float(y_test.mean()),
            "missing_cells_train": int(train[feature_names].isna().sum().sum()),
            "train_sha256": sha256_file(dataset.train_path),
            "test_sha256": sha256_file(dataset.test_path),
        },
        "split": {
            "strategy": "official test held out; official training split stratified 70/15/15 for fit/calibration/threshold tuning",
            "seed": seed,
            "fit_rows": len(x_fit),
            "calibration_rows": len(x_cal),
            "development_rows": len(x_dev),
            "test_rows": len(x_test),
        },
        "cost_model": {
            "false_positive": FALSE_POSITIVE_COST,
            "false_negative": FALSE_NEGATIVE_COST,
            "source": "original IDA 2016 / Scania challenge metric",
        },
        "calibration": "sigmoid/Platt scaling on dedicated calibration split",
        "models": {name: asdict(metrics) for name, metrics in results.items()},
        "promotion": {
            "baseline": "logistic_regression",
            "candidate": "hist_gradient_boosting",
            "decision": "PROMOTE" if promote else "REJECT",
            "rule": "candidate cost <= 95% of baseline cost, PR-AUC >= baseline, recall >= baseline",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def write_scania_report(
    path: str | Path,
    data_dir: str | Path,
    *,
    download: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    report = evaluate_scania(data_dir, download=download, seed=seed)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
