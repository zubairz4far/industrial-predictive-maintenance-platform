from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predictive_maintenance.scania import EXPECTED_COLUMNS, load_scania_csv
from predictive_maintenance.scania_evaluation import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    SigmoidCalibrator,
    _metrics,
    _select_threshold,
)


def _write_tiny_scania(path: Path) -> None:
    features = [f"f_{idx:03d}" for idx in range(EXPECTED_COLUMNS - 1)]
    frame = pd.DataFrame(
        [
            ["neg", *range(EXPECTED_COLUMNS - 1)],
            ["pos", *["na" if idx == 4 else idx + 1 for idx in range(EXPECTED_COLUMNS - 1)]],
        ],
        columns=["class", *features],
    )
    metadata = "\n".join(f"metadata {idx}" for idx in range(20)) + "\n"
    path.write_text(metadata + frame.to_csv(index=False), encoding="utf-8")


def test_scania_parser_contract(tmp_path):
    path = tmp_path / "aps.csv"
    _write_tiny_scania(path)
    frame = load_scania_csv(path)
    assert frame.shape == (2, EXPECTED_COLUMNS)
    assert frame.loc[1, "f_004"] != frame.loc[1, "f_004"]
    assert set(frame["class"]) == {"neg", "pos"}


def test_threshold_uses_original_asymmetric_cost():
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.10, 0.40, 0.30, 0.80])
    threshold = _select_threshold(y_true, probabilities)
    metrics = _metrics(y_true, probabilities, threshold)
    assert threshold == 0.30
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.challenge_cost == FALSE_POSITIVE_COST
    assert FALSE_NEGATIVE_COST > FALSE_POSITIVE_COST


def test_sigmoid_calibrator_returns_probabilities():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    raw = np.array([0.02, 0.10, 0.30, 0.45, 0.70, 0.95])
    calibrated = SigmoidCalibrator().fit(raw, y_true).predict(raw)
    assert calibrated.shape == raw.shape
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) > 0)
