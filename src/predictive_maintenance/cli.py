from __future__ import annotations

import argparse
import json

from .evaluation import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial predictive maintenance evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="run the deterministic v0.1 evaluation"
    )
    evaluate_parser.add_argument("--output", default="evals/results/v0.1_synthetic.json")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        report = write_report(args.output, seed=args.seed)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
