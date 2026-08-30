"""Generates data/results/tables/chunk_count_report.csv, showing how many chunks model_interface.py's chunking split it
into.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_articles, filter_research_articles, filter_by_year_range
from preprocessing import clean_text
from model_interface import EmbeddingModel


if __name__ == "__main__":
    print("Loading data...")
    articles = load_articles("data/raw/articles.csv")
    articles = filter_research_articles(articles)
    articles = filter_by_year_range(articles, min_year=2015, max_year=2025)
    abstracts = [clean_text(a.abstract) for a in articles]

    print(f"Computing embeddings for {len(abstracts)} abstracts (to get chunk counts)...")
    model = EmbeddingModel()
    model.encode(abstracts)  # we only need the side effect: model.last_chunk_counts

    chunk_counts = model.last_chunk_counts

    output_path = Path("data/results/tables/chunk_count_report.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pmid", "year", "abstract_length_chars", "n_chunks", "title"])
        for article, abstract, n_chunks in zip(articles, abstracts, chunk_counts):
            writer.writerow([article.pmid, article.year, len(abstract), n_chunks, article.title])

    print(f"Saved {len(articles)} rows to {output_path}")

    # Quick summary printed to terminal too
    multi_chunk = sum(1 for c in chunk_counts if c > 1)
    print(f"\n{multi_chunk} out of {len(articles)} abstracts ({100 * multi_chunk / len(articles):.1f}%) needed more than 1 chunk")
    print(f"Max chunks for a single abstract: {max(chunk_counts)}")