"""Compares naive BERT (mean-pooling) against SentenceTransformer for the
same alignment-scoring task.


Sample size: 250 articles
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from transformers import AutoModel, AutoTokenizer

BERT_MODEL_NAME = "bert-base-uncased"
SAMPLE_SIZE = 250


class NaiveBertEncoder:
    """Encodes text with raw BERT + naive mean-pooling over token vectors.

    Used as a simple baseline for comparison with the Sentence-BERT approach
adopted in the main analysis.
    """

    def __init__(self, model_name: str = BERT_MODEL_NAME) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # inference only, no dropout/training behavior

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encodes a list of texts with mean-pooling over BERT's token
        outputs. Long texts are truncated at BERT's own limit (512
        tokens) --> no chunking here, since the point is to reproduce the
        naive approach as-is, not to fix its limitations.
        """
        embeddings = []

        with torch.no_grad():  # no gradients needed, we're not training
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                outputs = self.model(**inputs)

                # outputs.last_hidden_state has shape (1, n_tokens, 768) 
                # naive mean-pooling over all returned token vectors, including special tokens ([CLS] and [SEP])
                token_vectors = outputs.last_hidden_state[0]
                mean_vector = token_vectors.mean(dim=0).numpy()
                embeddings.append(mean_vector)

        return np.array(embeddings)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def sample_articles(articles: list, n: int = SAMPLE_SIZE, seed: int = 42) -> list:
    """Randomly samples n articles, with a fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(articles), size=min(n, len(articles)), replace=False)
    return [articles[i] for i in indices]


def compare_methods(
    sbert_scores: np.ndarray,
    bert_scores: np.ndarray,
) -> dict:
    """Computes the three comparison measures discussed in the project log:
    correlation, spread (variance) comparison, and basic descriptive stats
    for both score sets.
    """
    correlation, correlation_p = stats.pearsonr(sbert_scores, bert_scores)

    return {
        "correlation": float(correlation),
        "correlation_p_value": float(correlation_p),
        "sbert_std": float(sbert_scores.std()),
        "bert_std": float(bert_scores.std()),
        "sbert_mean": float(sbert_scores.mean()),
        "bert_mean": float(bert_scores.mean()),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data_loader import (
        load_articles,
        load_aims_scope,
        filter_research_articles,
        filter_by_year_range,
    )
    from preprocessing import clean_text
    from model_interface import EmbeddingModel
    from evaluation import compute_alignment_scores

    print("Loading data...")
    articles = load_articles("data/raw/articles.csv")
    articles = filter_research_articles(articles)
    articles = filter_by_year_range(articles, min_year=2015, max_year=2025)
    aims_scope_raw = load_aims_scope("data/raw/aims_scope.txt")
    aims_scope = clean_text(aims_scope_raw)

    print(f"Sampling {SAMPLE_SIZE} articles (seed=42, reproducible)...")
    sample = sample_articles(articles, n=SAMPLE_SIZE)
    abstracts = [clean_text(a.abstract) for a in sample]

    print("Computing SentenceTransformer embeddings (our official method)...")
    sbert_model = EmbeddingModel()
    sbert_embeddings = sbert_model.encode(abstracts)
    sbert_aims_scope_embedding = sbert_model.encode_one(aims_scope)
    sbert_scores = compute_alignment_scores(sbert_embeddings, sbert_aims_scope_embedding)

    print("Computing naive BERT embeddings (comparison baseline)...")
    bert_model = NaiveBertEncoder()
    bert_embeddings = bert_model.encode(abstracts)
    bert_aims_scope_embedding = bert_model.encode_one(aims_scope)
    bert_scores = compute_alignment_scores(bert_embeddings, bert_aims_scope_embedding)

    print("\n--- Comparison ---")
    comparison = compare_methods(sbert_scores, bert_scores)
    for key, value in comparison.items():
        print(f"{key}: {value}")