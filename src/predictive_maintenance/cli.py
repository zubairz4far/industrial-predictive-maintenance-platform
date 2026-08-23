from __future__ import annotations

import argparse
import json

from .evaluation import write_report
from .scania_evaluation import write_scania_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial predictive maintenance evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic_parser = subparsers.add_parser(
        "evaluate-synthetic", help="run the deterministic v0.1 synthetic evaluation"
    )
    synthetic_parser.add_argument("--output", default="evals/results/v0.1_synthetic.json")
    synthetic_parser.add_argument("--seed", type=int, default=42)

    scania_parser = subparsers.add_parser(
        "evaluate-scania", help="run the v0.2 Scania APS benchmark"
    )
    scania_parser.add_argument("--data-dir", default=".cache/scania")
    scania_parser.add_argument("--output", default="evals/results/v0.2_scania_aps.json")
    scania_parser.add_argument("--seed", type=int, default=42)
    scania_parser.add_argument(
        "--download",
        action="store_true",
        help="download the official UCI train/test files when missing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate-synthetic":
        report = write_report(args.output, seed=args.seed)
    else:
        report = write_scania_report(
            args.output,
            args.data_dir,
            download=args.download,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
