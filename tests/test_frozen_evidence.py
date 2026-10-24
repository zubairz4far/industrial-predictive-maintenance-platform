"""Regression coverage for comparison of frozen Scania evidence."""

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_frozen_evidence.py"
spec = importlib.util.spec_from_file_location("verify_frozen_evidence", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
differences = module.differences


def test_tolerates_observed_ci_float_drift():
    expected = {"models": {"baseline": {
        "pr_auc": 0.7937751114982717,
        "threshold": 0.03559391618621555,
        "false_positives": 267,
    }}}
    observed = {"models": {"baseline": {
        "pr_auc": 0.7937891435182487,
        "threshold": 0.035588263002883314,
        "false_positives": 267,
    }}}
    assert differences(expected, observed) == []


def test_rejects_meaningful_metric_regression():
    assert "float drift" in differences({"pr_auc": 0.80}, {"pr_auc": 0.79})[0]


def test_counts_hashes_and_decisions_are_exact():
    expected = {"sha256": "abc", "false_negatives": 28, "decision": "PROMOTE"}
    for changed in (
        {"sha256": "xyz", "false_negatives": 28, "decision": "PROMOTE"},
        {"sha256": "abc", "false_negatives": 29, "decision": "PROMOTE"},
        {"sha256": "abc", "false_negatives": 28, "decision": "REJECT"},
    ):
        assert differences(expected, changed)


def test_structural_changes_and_boolean_types_are_rejected():
    assert differences({"rows": [1, 2]}, {"rows": [1]})
    assert differences({"rows": 1}, {"rows": True})
    assert differences({"rows": [1]}, {"rows": [1], "extra": 0})


def test_nonfinite_float_is_rejected():
    assert differences({"value": 0.25}, {"value": float("nan")})
