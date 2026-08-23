from predictive_maintenance.evaluation import evaluate


def test_evaluation_is_reproducible_and_complete():
    first = evaluate(seed=42)
    second = evaluate(seed=42)
    assert first == second
    assert first["dataset"]["rows"] == 7200
    assert 0 < first["dataset"]["positive_rate"] < 0.5
    assert set(first["models"]) == {"logistic_regression", "hist_gradient_boosting"}
    for metrics in first["models"].values():
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert metrics["business_cost"] >= 0


def test_promotion_decision_is_explicit():
    report = evaluate(seed=42)
    assert report["promotion"]["decision"] in {"PROMOTE", "REJECT"}
    assert "rule" in report["promotion"]
