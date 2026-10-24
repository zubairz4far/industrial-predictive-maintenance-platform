"""Compare frozen benchmark evidence without masking meaningful regressions.

Dataset hashes, integer counts, decisions, keys, and text must remain identical.
Only floating-point values receive a small absolute/relative tolerance to account
for insignificant numerical variation across BLAS/CPU/scikit-learn builds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_ATOL = 2e-5
DEFAULT_RTOL = 1e-8


def differences(
    expected: Any,
    actual: Any,
    *,
    path: str = "$",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> list[str]:
    """Return human-readable differences between expected and actual JSON."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if type(expected) is type(actual) and expected == actual else [f"{path}: boolean changed"]
    if isinstance(expected, int) or isinstance(actual, int):
        return [] if type(expected) is type(actual) and expected == actual else [f"{path}: integer/type changed: {expected!r} != {actual!r}"]
    if isinstance(expected, float) or isinstance(actual, float):
        if not (isinstance(expected, float) and isinstance(actual, float)):
            return [f"{path}: numeric type changed: {type(expected).__name__} != {type(actual).__name__}"]
        if not (math.isfinite(expected) and math.isfinite(actual)):
            return [f"{path}: non-finite numeric value is not valid evidence"]
        if math.isclose(expected, actual, abs_tol=atol, rel_tol=rtol):
            return []
        return [f"{path}: float drift {expected!r} -> {actual!r} exceeds atol={atol}, rtol={rtol}"]
    if type(expected) is not type(actual):
        return [f"{path}: type changed: {type(expected).__name__} != {type(actual).__name__}"]
    if isinstance(expected, dict):
        problems: list[str] = []
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        if missing:
            problems.append(f"{path}: missing keys: {missing!r}")
        if extra:
            problems.append(f"{path}: unexpected keys: {extra!r}")
        for key in sorted(expected.keys() & actual.keys()):
            problems.extend(differences(expected[key], actual[key], path=f"{path}.{key}", atol=atol, rtol=rtol))
        return problems
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: list length changed: {len(expected)} != {len(actual)}"]
        problems = []
        for index, (lhs, rhs) in enumerate(zip(expected, actual)):
            problems.extend(differences(lhs, rhs, path=f"{path}[{index}]", atol=atol, rtol=rtol))
        return problems
    return [] if expected == actual else [f"{path}: value changed: {expected!r} != {actual!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()
    if not (0 <= args.atol <= DEFAULT_ATOL and 0 <= args.rtol <= DEFAULT_RTOL):
        parser.error("tolerances cannot exceed the reviewed benchmark contract")
    expected = json.loads(args.frozen.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    observed = json.loads(args.observed.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    problems = differences(expected, observed, atol=args.atol, rtol=args.rtol)
    if problems:
        print("Frozen evidence verification FAILED:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("Frozen evidence verification PASSED: exact structure/decisions/counts/hashes; bounded float tolerance")
    return 0


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON numeric constant rejected: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
