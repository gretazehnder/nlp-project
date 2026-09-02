"""Computes how well each article's abstract aligns with the journal's
Aims & Scope, using embedding similarity.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_alignment_scores(
    abstract_embeddings: np.ndarray,
    aims_scope_embedding: np.ndarray,
) -> np.ndarray:
    """Computes one alignment score per abstract, against the Aims & Scope.

    Each score is a cosine similarity between -1 and 1 (in practice,
    for real text, almost always between 0 and 1). Higher means the
    abstract is more semantically close to the journal's stated scope.
    """
    if abstract_embeddings.ndim != 2:
        raise ValueError("abstract_embeddings must be a 2D array (one row per abstract).")

    # cosine_similarity expects two 2D arrays and compares every row of the first against every row of the second 
    # we only have one reference vector --> reshape it into a 1-row matrix
    aims_scope_matrix = aims_scope_embedding.reshape(1, -1)

    similarities = cosine_similarity(abstract_embeddings, aims_scope_matrix)

    # result has shape (n_abstracts, 1) one column, since we only compared against one reference --> flatten it to a simple 1D array
    return similarities.flatten()


def detect_outliers(scores: np.ndarray, z_threshold: float = 2.0) -> np.ndarray: # checked 3.0 too on real data --> too few outliers to inspect qualitatively --> kept 2.0
    """Flags scores that are unusually far from the average.

    Uses the z-score method: how many standard deviations a value is from the mean. 
    A common threshold is 2.0 (about the top/bottom 5% for a roughly normal distribution).

    Returns a boolean array: True where the article is an outlier.
    """
    mean = scores.mean()
    std = scores.std()

    if std == 0:
        # all scores are identical --> nothing can be an outlier
        return np.zeros(len(scores), dtype=bool)

    z_scores = (scores - mean) / std #NB: interpretation assumes not too skewed data --> checked on real scores in pipeline.py (skewness -0.13, close to 0 --> acceptable)
    return np.abs(z_scores) > z_threshold


def detect_outliers_iqr(scores: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Flags outliers using IQR instead of z-score --> doesn't assume normality, used as cross-check since Shapiro-Wilk rejected normality on real scores.

    k=1.5 is the standard threshold (Tukey's rule).
    """
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    return (scores < lower_bound) | (scores > upper_bound)


def summarize_scores(scores: np.ndarray) -> dict:
    """Returns basic descriptive statistics for a set of alignment scores."""
    return {
        "count": len(scores),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


# if __name__ == "__main__":
    # Quick manual check with made-up numbers, not real embeddings (to confirm the math behaves as expected)
#    fake_scores = np.array([0.5, 0.52, 0.48, 0.51, 0.02, 0.49, 0.90])

#    print("Summary:", summarize_scores(fake_scores))

#    outliers = detect_outliers(fake_scores, z_threshold=1.5)
#    print("Outlier flags:", outliers)
#    print("Outlier scores:", fake_scores[outliers])

# RESULT:
# Summary: {'count': 7, 'mean': 0.4885714285714285, 'median': 0.5, 'std': 0.2361856758344751, 'min': 0.02, 'max': 0.9}
# Outlier flags: [False False False False  True False  True]
# Outlier scores: [0.02 0.9 ]