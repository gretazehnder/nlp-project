"""Automated tests for evaluation.py.

These check the same behaviors we verified by hand earlier but in a repeatable way that
runs automatically every time.
"""

import sys
from pathlib import Path

import numpy as np

# lets this test file import from src/, since tests/ and src/ are separate folders at the same level in the project
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import compute_alignment_scores, detect_outliers, detect_outliers_iqr, summarize_scores


def test_identical_vectors_have_similarity_one():
    """A vector compared with itself must score exactly 1.0."""
    vector = np.array([[0.5, 0.3, 0.1, 0.9]])  # numbers don't matter here, only that it's the same vector twice
    scores = compute_alignment_scores(vector, vector[0])

    assert scores.shape == (1,)
    assert np.isclose(scores[0], 1.0)  # np.isclose instead of == because of floating point rounding


def test_orthogonal_vectors_have_similarity_zero():
    """Two vectors at a 90-degree angle must score 0 (unrelated)."""
    abstracts = np.array([[1.0, 0.0]])  # points along the x-axis
    aims_scope = np.array([0.0, 1.0])   # points along the y-axis, perpendicular to the one above

    scores = compute_alignment_scores(abstracts, aims_scope)

    assert np.isclose(scores[0], 0.0)


def test_compute_alignment_scores_returns_one_score_per_abstract():
    """With 3 abstracts, we must get exactly 3 scores back."""
    abstracts = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.7],
    ])
    aims_scope = np.array([1.0, 0.0])

    scores = compute_alignment_scores(abstracts, aims_scope)

    assert scores.shape == (3,)  # one score per row/abstract, not one big combined number


def test_detect_outliers_flags_extreme_values():
    """A value far from the rest should be flagged as an outlier."""
    scores = np.array([0.5, 0.52, 0.48, 0.51, 0.02, 0.49])

    outliers = detect_outliers(scores, z_threshold=1.5)  # lower threshold than the 2.0 default, so this small test set still triggers a flag

    # 0.02 is at index 4 and is clearly far from the rest
    assert outliers[4] == True
    # a "normal" value like 0.5 (index 0) should not be flagged
    assert outliers[0] == False


def test_detect_outliers_handles_identical_scores():
    """If every score is the same, nothing should be flagged."""
    scores = np.array([0.5, 0.5, 0.5, 0.5])  # std = 0 here --> edge case handled in evaluation.py

    outliers = detect_outliers(scores)

    assert not outliers.any()


def test_detect_outliers_iqr_flags_extreme_values():
    """Same extreme-value case as the z-score test, but for the IQR method --> added when detect_outliers_iqr was introduced as a cross-check"""
    scores = np.array([0.5, 0.52, 0.48, 0.51, 0.02, 0.49])

    outliers = detect_outliers_iqr(scores)

    assert outliers[4] == True   # 0.02 is clearly far from the rest
    assert outliers[0] == False  # a "normal" value like 0.5 should not be flagged


def test_summarize_scores_returns_expected_keys():
    """The summary dict must contain all the statistics we rely on."""
    scores = np.array([0.1, 0.2, 0.3])

    summary = summarize_scores(scores)

    assert summary["count"] == 3
    assert np.isclose(summary["mean"], 0.2)
    assert summary["min"] == 0.1
    assert summary["max"] == 0.3
