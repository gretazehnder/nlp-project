"""Cleans text before it goes into the embedding model.

NB: The text extraction in fetch_articles.py already strips XML/HTML tags,
so the only cleanup needed here is normalizing whitespace
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Cleans a single piece of text: fixes irregular whitespace."""
    if not text:
        return ""

    return _normalize_whitespace(text).strip()


def _normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces, tabs, and line breaks into a single space."""
    return re.sub(r"\s+", " ", text)


if __name__ == "__main__":
    sample = "This drug affects  the P450\nenzyme   system.\n\n" 
    print(repr(sample))
    print(repr(clean_text(sample))) #RESULT: 'This drug affects the P450 enzyme system.'