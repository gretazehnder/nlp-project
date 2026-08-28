"""Runs the full analysis end to end: load data, clean text, compute
embeddings, compute alignment scores, and summarize the results.
"""

from __future__ import annotations

import numpy as np

from data_loader import (
    load_articles,
    load_aims_scope,
    filter_research_articles,
    remove_duplicate_titles,
    filter_by_year_range,
)
from preprocessing import clean_text
from model_interface import EmbeddingModel
from evaluation import compute_alignment_scores, detect_outliers, detect_outliers_iqr, summarize_scores


def run_pipeline(
    articles_csv: str,
    aims_scope_txt: str,
    z_threshold: float = 2.0,
    min_year: int = 2015,
    max_year: int = 2025,
) -> dict:
    """Runs every step and returns a dict with all the results together."""

    print("Loading data...")
    articles = load_articles(articles_csv)
    articles = filter_research_articles(articles)  # remove correction notices, found by inspecting the first pipeline run's results

    # NB: remove_duplicate_titles() is NOT called here on purpose. EDA
    # found only 1 duplicated title, and inspecting the actual abstracts
    # showed it was two distinct correction notices about the same
    # original article (not a true database duplicate) - both already
    # removed by filter_research_articles() above. The function stays
    # in data_loader.py as a safety net for future re-fetches, where a
    # real duplicate unrelated to corrections could appear.
    # articles = remove_duplicate_titles(articles)

    articles = filter_by_year_range(articles, min_year=min_year, max_year=max_year)  # excludes incomplete 2026
    aims_scope_raw = load_aims_scope(aims_scope_txt)

    print("Cleaning text...")
    aims_scope = clean_text(aims_scope_raw)
    abstracts = [clean_text(a.abstract) for a in articles]

    print(f"Computing embeddings for {len(abstracts)} abstracts...")
    model = EmbeddingModel()
    abstract_embeddings = model.encode(abstracts)
    aims_scope_embedding = model.encode_one(aims_scope)

    print("Computing alignment scores...")
    scores = compute_alignment_scores(abstract_embeddings, aims_scope_embedding)
    outliers = detect_outliers(scores, z_threshold=z_threshold)
    summary = summarize_scores(scores)

    return {
        "articles": articles,
        "scores": scores,
        "outliers": outliers,
        "summary": summary,
    }


import csv


def save_results_to_csv(results: dict, output_dir: str = "data/results/tables") -> None:
    """Saves the key results as CSV tables, one per aspect of the analysis.

    Mirrors (partially) the output tables used in the colleague's repo,
    adapted to what we actually compute - we skip the bertopic_* files
    since topic_analysis.py (BERTopic) hasn't been implemented yet.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    articles = results["articles"]
    scores = results["scores"]
    outliers_z = results["outliers"]

    _save_summary_statistics(results["summary"], output_path / "summary_statistics.csv")
    _save_yearly_alignment(articles, scores, output_path / "yearly_alignment.csv")
    _save_top_or_bottom_articles(articles, scores, n=3, top=True, path=output_path / "top_aligned_articles.csv")
    _save_top_or_bottom_articles(articles, scores, n=3, top=False, path=output_path / "least_aligned_articles.csv")
    _save_outlier_articles(articles, scores, outliers_z, output_path / "outlier_articles.csv")

    print(f"Saved result tables to {output_dir}/")


def _save_summary_statistics(summary: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def _save_yearly_alignment(articles: list, scores, path: Path) -> None:
    years = sorted(set(a.year for a in articles))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "count", "mean_score", "std_score"])
        for year in years:
            year_scores = [scores[i] for i, a in enumerate(articles) if a.year == year]
            writer.writerow([year, len(year_scores), float(np.mean(year_scores)), float(np.std(year_scores))])


def _save_top_or_bottom_articles(articles: list, scores, n: int, top: bool, path: Path) -> None:
    order = np.argsort(scores)
    indices = order[-n:][::-1] if top else order[:n]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "year", "pmid", "title"])
        for i in indices:
            writer.writerow([f"{scores[i]:.4f}", articles[i].year, articles[i].pmid, articles[i].title])


def _save_outlier_articles(articles: list, scores, outliers_z, path: Path) -> None:
    from evaluation import detect_outliers_iqr

    outliers_iqr = detect_outliers_iqr(scores)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "year", "pmid", "title", "outlier_zscore", "outlier_iqr"])
        for i in range(len(articles)):
            if outliers_z[i] or outliers_iqr[i]:
                writer.writerow([
                    f"{scores[i]:.4f}", articles[i].year, articles[i].pmid, articles[i].title,
                    outliers_z[i], outliers_iqr[i],
                ])


if __name__ == "__main__":
    results = run_pipeline(
        articles_csv="data/raw/articles.csv",
        aims_scope_txt="data/raw/aims_scope.txt",
    )

    print("\n--- Summary ---")
    print(results["summary"])

    # Check whether the scores look roughly normally distributed, since
    # detect_outliers() uses the z-score method which assumes normality.
    # If this fails, IQR-based outlier detection might be more appropriate.
    from scipy import stats

    scores = results["scores"]
    shapiro_stat, shapiro_p = stats.shapiro(scores)
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)

    print(f"\n--- Normality check ---")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Skewness: {skewness:.4f} (0 = symmetric)")
    print(f"Kurtosis: {kurtosis:.4f} (0 = normal-like tails)")
    # NB: Shapiro-Wilk is very sensitive with large samples (n>1000) and
    # tends to reject normality even for minor deviations - so we also
    # look at skewness/kurtosis and the histogram, not just the p-value.
    if shapiro_p < 0.05:
        print("-> Shapiro-Wilk suggests NOT normal (p < 0.05). Check skewness/histogram before trusting z-score outliers.")
    else:
        print("-> Shapiro-Wilk does not reject normality.")

    n_outliers = results["outliers"].sum()
    print(f"\nOutliers found (z-score): {n_outliers} out of {len(results['scores'])}")

    # Cross-check with the IQR method, which doesn't assume normality -
    # since Shapiro-Wilk formally rejected normality above, this tells
    # us whether that rejection actually matters in practice
    outliers_iqr = detect_outliers_iqr(scores)
    n_outliers_iqr = outliers_iqr.sum()
    print(f"Outliers found (IQR): {n_outliers_iqr} out of {len(scores)}")

    # NB: comparing the two boolean arrays position-by-position (e.g.
    # z_outliers == iqr_outliers) would be misleading here, since most
    # articles are "not an outlier" under BOTH methods - that trivial
    # agreement would dominate the count. What we actually want is the
    # overlap between the two SETS of flagged outliers.
    z_outlier_set = set(np.where(results["outliers"])[0])
    iqr_outlier_set = set(np.where(outliers_iqr)[0])
    overlap = z_outlier_set & iqr_outlier_set
    union = z_outlier_set | iqr_outlier_set

    print(f"Of the {len(iqr_outlier_set)} IQR outliers, {len(overlap)} are also flagged by z-score")
    print(f"Jaccard similarity (intersection/union) between the two outlier sets: {len(overlap) / len(union):.3f}")

    # Show the 3 highest and 3 lowest scoring articles, to sanity-check
    # the results by reading the actual titles
    articles = results["articles"]

    top_indices = np.argsort(scores)[-3:][::-1]
    bottom_indices = np.argsort(scores)[:3]

    print("\n--- Top 3 most aligned articles ---")
    for i in top_indices:
        print(f"[{scores[i]:.3f}] {articles[i].title}")

    print("\n--- Bottom 3 least aligned articles ---")
    for i in bottom_indices:
        print(f"[{scores[i]:.3f}] {articles[i].title}")

    # Which 2025 articles are outliers, using our single official method
    # (z-score, computed on the whole 2015-2025 dataset - see evaluation.py).
    # NB: the boxplot in visualization.py draws its own outlier circles
    # using matplotlib's built-in convention (IQR, computed per year) -
    # that's just how boxplots are drawn, not a second official outlier
    # list. Our one real list of outliers is this one, z-score based.
    print("\n--- Outliers in 2025 (z-score, our official method) ---")
    for i in range(len(articles)):
        if articles[i].year == 2025 and results["outliers"][i]:
            print(f"[{scores[i]:.3f}] {articles[i].title}")

    # Quick visual check of the score distribution's shape, to decide
    # if the z-score method's normality assumption is reasonable here
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path("data/results/figures").mkdir(parents=True, exist_ok=True)  # matplotlib doesn't create folders on its own, unlike our own CSV-saving code

    plt.hist(scores, bins=30, edgecolor="black")
    plt.xlabel("Alignment score")
    plt.ylabel("Number of articles")
    plt.title("Distribution of alignment scores (Drug Safety)")
    plt.savefig("data/results/figures/score_distribution_check.png")
    print("\nSaved histogram to data/results/figures/score_distribution_check.png")

    save_results_to_csv(results)