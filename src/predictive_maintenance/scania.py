from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UCI_DATASET_ID = 421
UCI_DOI = "10.24432/C51S51"
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks"
TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00421/"
    "aps_failure_training_set.csv"
)
TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00421/"
    "aps_failure_test_set.csv"
)
TRAIN_FILENAME = "aps_failure_training_set.csv"
TEST_FILENAME = "aps_failure_test_set.csv"
TARGET = "class"
EXPECTED_COLUMNS = 171
EXPECTED_TRAIN_ROWS = 60_000
EXPECTED_TEST_ROWS = 16_000


@dataclass(frozen=True)
class ScaniaDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    train_path: Path
    test_path: Path


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "predictive-maintenance-v0.2"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(destination)


def ensure_scania_files(data_dir: str | Path, download: bool = False) -> tuple[Path, Path]:
    root = Path(data_dir)
    train_path = root / TRAIN_FILENAME
    test_path = root / TEST_FILENAME
    missing = [path for path in (train_path, test_path) if not path.exists()]
    if missing and not download:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing Scania APS files: {names}. Re-run with --download or place the official "
            f"UCI files in {root}."
        )
    if not train_path.exists():
        _download(TRAIN_URL, train_path)
    if not test_path.exists():
        _download(TEST_URL, test_path)
    return train_path, test_path


def load_scania_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=20, na_values="na", low_memory=False)
    if frame.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(f"Expected {EXPECTED_COLUMNS} columns, found {frame.shape[1]} in {path}")
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {path}")
    labels = set(frame[TARGET].dropna().unique())
    if not labels.issubset({"neg", "pos"}):
        raise ValueError(f"Unexpected class labels in {path}: {sorted(labels)}")
    numeric_columns = [column for column in frame.columns if column != TARGET]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame


def load_scania_dataset(
    data_dir: str | Path,
    *,
    download: bool = False,
    validate_sizes: bool = True,
) -> ScaniaDataset:
    train_path, test_path = ensure_scania_files(data_dir, download=download)
    train = load_scania_csv(train_path)
    test = load_scania_csv(test_path)
    if validate_sizes:
        if len(train) != EXPECTED_TRAIN_ROWS:
            raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} training rows, found {len(train)}")
        if len(test) != EXPECTED_TEST_ROWS:
            raise ValueError(f"Expected {EXPECTED_TEST_ROWS} test rows, found {len(test)}")
    return ScaniaDataset(train=train, test=test, train_path=train_path, test_path=test_path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
