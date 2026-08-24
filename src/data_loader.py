"""Reads and checks the article data used by the rest of the project.

NB: this module does not talk to PubMed directly, it just reads the CSV
already produced by scripts/fetch_articles.py, checks that it looks
correct, and returns clean data for the next steps (preprocessing,
embeddings, evaluation).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# Columns that MUST be present and non-empty in every row of the CSV
REQUIRED_COLUMNS = ["title", "abstract", "year"]


@dataclass
class LoadedArticle:
    """One article row from the CSV, after validation."""
    pmid: str
    title: str
    abstract: str
    year: int
    journal: str
    doi: str


class DataValidationError(Exception):
    """Raised when the input csv is missing something it needs."""


def load_articles(csv_path: str) -> list[LoadedArticle]:
    """Reads the articles CSV and returns only valid rows."""

    # check the file exists before trying to open it
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Articles file not found: {csv_path}. "
            "Run scripts/fetch_articles.py first to generate it."
        )

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # check needed columns of csv
        _check_columns(reader.fieldnames)

        articles = []
        dropped = 0
        for row in reader:
            # some rows may be missing title/abstract/year (for example an incomplete record) --> drop, don't crash
            article = _row_to_article(row)
            if article is None:
                dropped += 1
                continue
            articles.append(article)

    # if nothing was valid something is wrong with the file
    if not articles:
        raise DataValidationError("No valid articles found in the CSV.")

    if dropped:
        print(f"Dropped {dropped} rows with missing data")

    return articles


def load_aims_scope(txt_path: str) -> str:
    """Reads the Aims & Scope text file and returns it as a single string."""
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Aims & Scope file not found: {txt_path}")

    text = path.read_text(encoding="utf-8").strip()

    # empty file means something went wrong when it was copied/saved
    if not text:
        raise DataValidationError(f"Aims & Scope file is empty: {txt_path}")

    return text


def _check_columns(fieldnames) -> None:
    """Makes sure the CSV has all the columns we depend on."""
    if fieldnames is None:
        raise DataValidationError("CSV file has no header row.")

    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        raise DataValidationError(f"CSV is missing required columns: {missing}")


def _row_to_article(row: dict) -> LoadedArticle | None:
    """Converts one CSV row into a LoadedArticle, or None if invalid."""
    title = (row.get("title") or "").strip()  # row.get returns None if the column is missing entirely, "or ''" turns it into an empty string so .strip() works
    abstract = (row.get("abstract") or "").strip()
    year_raw = (row.get("year") or "").strip()

    if not title or not abstract or not year_raw:
        return None

    try:
        year = int(year_raw)
    except ValueError:
        return None

    return LoadedArticle(
        pmid=(row.get("pmid") or "").strip(),
        title=title,
        abstract=abstract,
        year=year,
        journal=(row.get("journal") or "").strip(),
        doi=(row.get("doi") or "").strip(),
    )


if __name__ == "__main__":
    # for a quick manual check
    articles = load_articles("data/raw/articles.csv")
    scope = load_aims_scope("data/raw/aims_scope.txt")

    print(f"Loaded {len(articles)} valid articles")
    print(f"Aims & Scope length: {len(scope)} characters")
    print("\nExample article:")
    print(articles[0].title)
    print(f"Year: {articles[0].year}")

    # RESULT: Loaded 1129 valid articles
    # Aims & Scope length: 905 characters
    # Example article:
    # Pharmacovigilance Due Diligence in Drug Development: A Practical Playbook for Risk Identification, Compliance Assessment, and Strategic Decision Making.
    # Year: 2026