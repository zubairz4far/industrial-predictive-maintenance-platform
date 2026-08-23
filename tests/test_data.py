from predictive_maintenance.data import (
    FEATURES,
    TARGET,
    chronological_split,
    generate_synthetic_telemetry,
)


def test_fixture_is_deterministic_and_has_expected_contract():
    first = generate_synthetic_telemetry(n_machines=4, steps_per_machine=30, seed=7)
    second = generate_synthetic_telemetry(n_machines=4, steps_per_machine=30, seed=7)
    assert first.equals(second)
    assert set(FEATURES).issubset(first.columns)
    assert TARGET in first.columns
    assert TARGET not in FEATURES


def test_chronological_split_has_no_time_overlap():
    frame = generate_synthetic_telemetry(n_machines=5, steps_per_machine=40, seed=9)
    train, dev, test = chronological_split(frame)
    assert train["timestamp"].max() < dev["timestamp"].min()
    assert dev["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(dev) + len(test) == len(frame)
