from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# PubMed web addresses: one for searching, one for downloading details
ESEARCH_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Pause between requests, so we don't overload PubMed's server
DELAY = 0.4


@dataclass
class Article:
    """One article downloaded from PubMed, with its main data."""
    pmid: str          # PubMed's identifying number
    title: str
    abstract: str
    year: int | None
    journal: str
    doi: str = ""

    def is_usable(self, min_abstract_chars: int = 50) -> bool:
        """Checks if the article has enough information to be usable."""
        return (
            bool(self.title.strip())
            and bool(self.abstract.strip())
            and len(self.abstract) >= min_abstract_chars
            and self.year is not None
        )


def find_pmids(journal_name: str, start_year: int, end_year: int, max_results: int = 2000) -> list[str]:
    """Search PubMed for articles from a journal within a date range.
    Returns only their ID numbers (PMIDs), not the actual content yet.
    """
    params = {
        "db": "pubmed",
        "term": f'"{journal_name}"[Journal] AND {start_year}:{end_year}[PDAT]',
        "retmax": str(max_results),
        "retmode": "xml",
    }
    xml_bytes = _call_eutils(ESEARCH_ENDPOINT, params)
    root = ET.fromstring(xml_bytes)
    return [node.text for node in root.findall(".//IdList/Id") if node.text]


def fetch_articles(pmids: list[str], batch_size: int = 100) -> list[Article]:
    """Given the PMIDs found earlier, download the real details of each
    article (title, abstract, year...). Done in batches of 100 at a time.
    """
    articles = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        xml_bytes = _call_eutils(EFETCH_ENDPOINT, params)
        root = ET.fromstring(xml_bytes)
        for article_node in root.findall(".//PubmedArticle"):
            articles.append(_parse_article(article_node))
    return articles


def _call_eutils(endpoint: str, params: dict) -> bytes:
    """Makes a web request to PubMed and returns the raw response."""
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    time.sleep(DELAY)
    return data


def _parse_article(node: ET.Element) -> Article:
    """Extracts title, abstract, year, etc. from PubMed's XML format."""
    article_el = node.find(".//Article")
    title = _text(article_el.find("ArticleTitle")) if article_el is not None else ""

    abstract_parts = []
    if article_el is not None:
        for chunk in article_el.findall(".//Abstract/AbstractText"):
            text = _text(chunk)
            if text:
                abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    pmid_el = node.find(".//PMID")
    pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""

    journal = _text(node.find(".//Journal/Title"))
    year = _find_year(node)

    doi = ""
    for id_el in node.findall(".//ArticleIdList/ArticleId"):
        if id_el.attrib.get("IdType") == "doi":
            doi = _text(id_el)

    return Article(pmid=pmid, title=title, abstract=abstract, year=year, journal=journal, doi=doi)


def _text(element) -> str:
    """Cleans up extracted text (double spaces, line breaks, etc.)."""
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def save_to_csv(articles: list[Article], output_path: str) -> None:
    """Saves a list of articles to a CSV file, one row per article."""
    import csv
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pmid", "title", "abstract", "year", "journal", "doi"])
        for a in articles:
            writer.writerow([a.pmid, a.title, a.abstract, a.year, a.journal, a.doi])


def _find_year(node: ET.Element) -> int | None:
    """Looks for the publication year in a few possible places."""
    for path in (".//Journal/JournalIssue/PubDate/Year", ".//ArticleDate/Year"):
        year_el = node.find(path)
        if year_el is not None and year_el.text and year_el.text.isdigit():
            return int(year_el.text)
    medline_date_el = node.find(".//Journal/JournalIssue/PubDate/MedlineDate")
    if medline_date_el is not None and medline_date_el.text:
        match = re.search(r"(19|20)\d{2}", medline_date_el.text)
        if match:
            return int(match.group(0))
    return None


# This part only runs if you launch the file directly from the terminal
# (used for a quick test, without saving anything to disk yet)
if __name__ == "__main__":
    import sys

    journal = sys.argv[1] if len(sys.argv) > 1 else "Drug Safety"
    y_start = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
    y_end = int(sys.argv[3]) if len(sys.argv) > 3 else 2025
    max_results = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    output_path = sys.argv[5] if len(sys.argv) > 5 else None

    # Step 1: find the article IDs
    ids = find_pmids(journal, y_start, y_end, max_results=max_results)
    print(f"Found {len(ids)} articles for '{journal}' ({y_start}-{y_end})")

    # Step 2: download the details of those articles
    articles = fetch_articles(ids)
    usable = [a for a in articles if a.is_usable()]
    print(f"Downloaded {len(articles)} articles, {len(usable)} usable")

    # Step 3: either save to CSV, or just show one example
    if output_path:
        save_to_csv(usable, output_path)
        print(f"Saved {len(usable)} usable articles to {output_path}")
    elif usable:
        first = usable[0]
        print(f"\nExample - PMID {first.pmid} ({first.year}):")
        print(first.title)
        print(first.abstract[:300])