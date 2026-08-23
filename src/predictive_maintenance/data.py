from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "load_pct",
    "lubricant_quality",
    "hours_since_maintenance",
    "machine_age_days",
]
TARGET = "failure_next_24h"


def generate_synthetic_telemetry(
    n_machines: int = 40,
    steps_per_machine: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic telemetry with a forward-looking failure label.

    The target is sampled from current/latent condition only. No future sensor value is
    included in the feature set, so the fixture is suitable for leakage checks and CI.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")

    for machine_idx in range(n_machines):
        machine_id = f"machine-{machine_idx:03d}"
        base_temp = rng.normal(66.0, 2.5)
        base_vibration = rng.normal(2.1, 0.2)
        base_pressure = rng.normal(5.2, 0.25)
        base_rpm = rng.normal(1450.0, 70.0)
        machine_age_days = float(rng.integers(60, 1800))
        hours_since_maintenance = float(rng.integers(0, 96))

        for step in range(steps_per_machine):
            timestamp = start + pd.Timedelta(hours=6 * step)
            load_pct = float(np.clip(rng.normal(63.0, 16.0), 15.0, 100.0))
            hours_since_maintenance += 6.0

            # Deterministic maintenance events reset accumulated wear without peeking ahead.
            if hours_since_maintenance > 420 and rng.random() < 0.16:
                hours_since_maintenance = float(rng.integers(0, 24))

            wear = (
                0.0028 * hours_since_maintenance
                + 0.008 * max(load_pct - 60.0, 0.0)
                + 0.00022 * machine_age_days
            )
            temperature = base_temp + 2.2 * wear + rng.normal(0.0, 1.2)
            vibration = base_vibration + 0.75 * wear + rng.normal(0.0, 0.18)
            pressure = base_pressure - 0.28 * wear + rng.normal(0.0, 0.12)
            rpm = base_rpm + 2.5 * (load_pct - 60.0) + rng.normal(0.0, 35.0)
            lubricant_quality = float(np.clip(1.0 - 0.16 * wear + rng.normal(0, 0.035), 0, 1))

            interaction = float(temperature > 70.0 and vibration > 2.8)
            logit = (
                -5.0
                + 1.35 * wear
                + 0.10 * max(temperature - 68.0, 0.0)
                + 1.10 * max(vibration - 2.5, 0.0)
                + 0.025 * max(load_pct - 65.0, 0.0)
                + 2.0 * max(0.75 - lubricant_quality, 0.0)
                + 1.2 * interaction
            )
            failure_probability = 1.0 / (1.0 + np.exp(-logit))
            failure_next_24h = int(rng.random() < failure_probability)

            rows.append(
                {
                    "timestamp": timestamp,
                    "machine_id": machine_id,
                    "temperature_c": temperature,
                    "vibration_mm_s": vibration,
                    "pressure_bar": pressure,
                    "rpm": rpm,
                    "load_pct": load_pct,
                    "lubricant_quality": lubricant_quality,
                    "hours_since_maintenance": hours_since_maintenance,
                    "machine_age_days": machine_age_days,
                    TARGET: failure_next_24h,
                }
            )

    return pd.DataFrame(rows).sort_values(["timestamp", "machine_id"]).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    dev_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by timestamp so evaluation always occurs after training in time."""
    timestamps = np.array(sorted(frame["timestamp"].unique()))
    train_cut = max(1, int(len(timestamps) * train_fraction))
    dev_cut = max(train_cut + 1, int(len(timestamps) * (train_fraction + dev_fraction)))
    train_times = set(timestamps[:train_cut])
    dev_times = set(timestamps[train_cut:dev_cut])

    train = frame[frame["timestamp"].isin(train_times)].copy()
    dev = frame[frame["timestamp"].isin(dev_times)].copy()
    test = frame[~frame["timestamp"].isin(train_times | dev_times)].copy()
    return train, dev, test
