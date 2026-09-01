"""Command-line interface for running the main alignment pipeline
with custom parameters without modifying pipeline.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# lets this script import from src/, same reasoning as tests/test_evaluation.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Drug Safety alignment pipeline.")
    parser.add_argument("--articles-csv", default="data/raw/articles.csv")
    parser.add_argument("--aims-scope-txt", default="data/raw/aims_scope.txt")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--min-year", type=int, default=2015)
    parser.add_argument("--max-year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_pipeline(
        articles_csv=args.articles_csv,
        aims_scope_txt=args.aims_scope_txt,
        z_threshold=args.z_threshold,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    print("\n--- Summary ---")
    print(results["summary"])

    n_outliers = results["outliers"].sum()
    print(f"Outliers found (z-score): {n_outliers} out of {len(results['scores'])}")


if __name__ == "__main__":
    main()

# EXAMPLE USAGE: 
# python3 scripts/run_pipeline.py --z-threshold 3.0 --min-year 2018
