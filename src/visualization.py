"""Visualizes how alignment scores change over time (temporal drift).

Uses the articles and scores already computed by pipeline.py - this
module only produces plots, it doesn't recompute anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_score_by_year(articles: list, scores: np.ndarray, output_path: str) -> None:
    """Line chart: average alignment score per year, with error bars
    showing the standard deviation within each year.
    """
    years = sorted(set(a.year for a in articles))

    means = []
    stds = []
    for year in years:
        year_scores = [scores[i] for i, a in enumerate(articles) if a.year == year]
        means.append(np.mean(year_scores))
        stds.append(np.std(year_scores))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.errorbar(years, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("Year")
    plt.ylabel("Alignment score (mean +/- std)")
    plt.title("Alignment score trend over time (Drug Safety)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_boxplot_by_year(articles: list, scores: np.ndarray, output_path: str) -> None:
    """Boxplot: distribution of alignment scores per year, to see
    spread and outliers within each year, not just the average.

    NB: the circles matplotlib draws beyond the whiskers follow its own
    built-in convention (IQR, computed separately per year) - these are
    NOT the project's official outlier list. The official outliers
    (used everywhere else: evaluation.py, pipeline.py, the CSV exports)
    are computed with z-score on the whole dataset, see evaluation.py.
    """
    years = sorted(set(a.year for a in articles))

    data_per_year = [
        [scores[i] for i, a in enumerate(articles) if a.year == year]
        for year in years
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data_per_year, tick_labels=[str(y) for y in years])
    plt.xlabel("Year")
    plt.ylabel("Alignment score")
    plt.title("Alignment score distribution per year (Drug Safety)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # Quick manual check with the real pipeline results
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import run_pipeline

    results = run_pipeline(
        articles_csv="data/raw/articles.csv",
        aims_scope_txt="data/raw/aims_scope.txt",
    )

    plot_score_by_year(
        results["articles"],
        results["scores"],
        "data/results/figures/score_trend_by_year.png",
    )
    print("Saved data/results/figures/score_trend_by_year.png")

    plot_boxplot_by_year(
        results["articles"],
        results["scores"],
        "data/results/figures/score_boxplot_by_year.png",
    )
    print("Saved data/results/figures/score_boxplot_by_year.png")