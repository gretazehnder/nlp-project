"""Automated tests for preprocessing.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import clean_text


def test_multiple_whitespace_collapsed():
    """Multiple spaces, tabs, and line breaks should collapse to a single space."""
    messy = "This drug  affects\tthe   P450\nenzyme system."
    result = clean_text(messy)

    assert result == "This drug affects the P450 enzyme system."

def test_wording_and_casing_preserved():
    """clean_text should only touch whitespace, not the actual words or casing."""
    text = "Patients receiving Metformin showed NO adverse events."
    result = clean_text(text)

    assert "Metformin" in result  # casing preserved
    assert "NO" in result
    assert "patients receiving metformin" not in result  # not lowercased