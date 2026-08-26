"""Exploratory Data Analysis (EDA) on the raw articles dataset.

Checks the data itself (not embeddings or scores): abstract length
stats, articles per year, and duplicate titles.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_articles


def check_abstract_lengths(articles) -> dict:
    """Returns basic statistics on abstract length (in characters)."""
    import numpy as np

    lengths = np.array([len(a.abstract) for a in articles])
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
    }


def check_years(articles) -> dict:
    """Counts how many articles fall in each year."""
    from collections import Counter

    return dict(sorted(Counter(a.year for a in articles).items()))


def check_duplicates(articles) -> int:
    """Counts how many articles share the exact same title (possible duplicates)."""
    titles = [a.title.strip().lower() for a in articles]
    return len(titles) - len(set(titles))


def show_duplicate_titles(articles) -> None:
    """Prints the duplicated title(s) with their PMIDs and abstracts,
    so we can check by eye if it's really the same article indexed
    twice (same abstract text --> true duplicate, safe to remove) or
    two different correction notices about the same original article
    (different abstract text --> not a database duplicate).
    """
    from collections import defaultdict

    by_title = defaultdict(list)
    for a in articles:
        key = a.title.strip().lower()
        by_title[key].append(a)

    for key, group in by_title.items():
        if len(group) > 1:
            print(f"\nTitle: {group[0].title}")
            for a in group:
                print(f"  PMID: {a.pmid}, year: {a.year}, journal: {a.journal}")
                print(f"  Abstract: {a.abstract}")


def check_token_truncation(articles, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: int = 256) -> dict:
    """Checks how many abstracts exceed the model's token limit and
    would be silently truncated. Uses the model's own tokenizer, since
    character count is only a rough proxy for token count.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    exceeding = 0
    token_counts = []
    for a in articles:
        n_tokens = len(tokenizer.encode(a.abstract))
        token_counts.append(n_tokens)
        if n_tokens > max_tokens:
            exceeding += 1

    import numpy as np

    token_counts = np.array(token_counts)
    return {
        "max_tokens_limit": max_tokens,
        "abstracts_exceeding_limit": exceeding,
        "percent_exceeding": 100 * exceeding / len(articles),
        "token_count_mean": float(token_counts.mean()),
        "token_count_max": int(token_counts.max()),
    }


if __name__ == "__main__":
    articles = load_articles("data/raw/articles.csv")
    print(f"Total articles loaded: {len(articles)}")

    print("\n--- Abstract length check (characters) ---")
    print(check_abstract_lengths(articles))

    print("\n--- Articles per year ---")
    for year, count in check_years(articles).items():
        print(f"{year}: {count}")

    print("\n--- Duplicate check ---")
    n_duplicates = check_duplicates(articles)
    print(f"Articles with a duplicated title: {n_duplicates}")
    if n_duplicates:
        show_duplicate_titles(articles)

    print("\n--- Token truncation check (MiniLM, 256 token limit) ---")
    print(check_token_truncation(articles, max_tokens=256))

    print("\n--- Token truncation check (mpnet, 384 token limit) ---")
    print(check_token_truncation(
        articles,
        model_name="sentence-transformers/all-mpnet-base-v2",
        max_tokens=384,
    ))