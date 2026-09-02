"""Turns text into numeric vectors (embeddings) using SentenceTransformer.
NB: This is the only file that talks directly to the embedding model. Every
other module just calls the functions here, so if we ever want to change
the model, we only need to touch this file.

"""

from __future__ import annotations

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Kept as a separate constant (instead of writing it inside the class) so it's easy to change the model if needed
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """A loaded embedding model, ready to turn text into vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)  # important for performance
        self._max_tokens = self._model.max_seq_length  # the model's own token limit (256 for MiniLM)
        self.last_chunk_counts: list[int] = []  # how many chunks each text needed, set by encode()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Turns a list of texts into a matrix of embeddings.

        Any text longer than the model's token limit is automatically
        split into sentence-based chunks and averaged.

        Returns an array of shape (number_of_texts, embedding_size).
        """
        if not texts:
            raise ValueError("texts must not be empty.")

        # Collect every chunk from every text into one flat list, so we can encode them all in a single batched model call 
        all_chunks: list[str] = []
        chunk_owner: list[int] = []  
        chunk_weights: list[int] = []  # token count of each chunk, used for the weighted average

        for i, text in enumerate(texts):
            for chunk in self._split_into_chunks(text):
                all_chunks.append(chunk)
                chunk_owner.append(i)
                chunk_weights.append(len(self._model.tokenizer.encode(chunk)))

        # record how many chunks each text produced --> exported by pipeline.py in chunk_count_report.csv
        self.last_chunk_counts = [chunk_owner.count(i) for i in range(len(texts))]

        chunk_embeddings = self._model.encode(all_chunks, show_progress_bar=True)  # progress bar useful for large batches 

        # normalize each chunk embedding to unit length before averaging, so the weighted average is driven only by our explicit weights (token count),
        # not by incidental differences in vector magnitude between chunks (SentenceTransformer's raw output is not guaranteed to be unit-norm)
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        chunk_embeddings = chunk_embeddings / norms

        # recombine the chunks belonging to the same original text into a single embedding per text (through weighted average)
        embedding_size = chunk_embeddings.shape[1]
        result = np.zeros((len(texts), embedding_size))

        for i in range(len(texts)):
            indices = [j for j, owner in enumerate(chunk_owner) if owner == i]
            embeddings_for_text = chunk_embeddings[indices]
            weights = np.array([chunk_weights[j] for j in indices], dtype=float)
            weights = weights / weights.sum()  # normalize so weights sum to 1
            result[i] = np.average(embeddings_for_text, axis=0, weights=weights)

        return result

    def encode_one(self, text: str) -> np.ndarray:
        """Turns a single text into one embedding vector."""
        if not text.strip():
            raise ValueError("text must not be empty.")

        return self.encode([text])[0]  # wraps text in a list because encode() expects a list, then takes the first (only) result

    def _split_into_chunks(self, text: str) -> list[str]:
        """Splits text into chunks that fit the model's token limit (never cutting a sentence in half).
        Sentence tokens are counted without special tokens, reserving 2 for the chunk's own [CLS]/[SEP] added later at encoding time. 
        A lone sentence over the limit becomes its own chunk and gets truncated (rare case).
        """
        sentences = self._split_into_sentences(text)
        usable_tokens = self._max_tokens - 2  # headroom for the chunk's own [CLS]/[SEP]

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = len(self._model.tokenizer.encode(sentence, add_special_tokens=False))

            if current_sentences and current_token_count + sentence_tokens > usable_tokens:
                # adding this sentence would overflow the current chunk --> close the current chunk and start a new one
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_token_count = 0

            current_sentences.append(sentence)
            current_token_count += sentence_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Splits text into sentences using NLTK's sent_tokenize.
        """
        sentences = sent_tokenize(text.strip())
        return [s for s in sentences if s]


if __name__ == "__main__":
    # Quick manual check: two similar sentences should end up close together, an unrelated one should end up further away
    model = EmbeddingModel()

    texts = [
        "This drug is used to treat type 2 diabetes.",
        "The medication helps control blood sugar levels.",
        "The stock market fell sharply today.",
    ]

    vectors = model.encode(texts)
    print(f"Shape: {vectors.shape}")

    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(vectors)
    print("\nSimilarity matrix:")
    print(similarities)

    # Quick check that chunking kicks in for long text: repeat a sentence enough times to clearly exceed the model's token limit
    long_text = "Patients receiving this treatment were monitored for adverse events. " * 60
    long_vector = model.encode_one(long_text)
    print(f"\nLong text encoded successfully, shape: {long_vector.shape}")

    # verify periods survive intact and sentence splitting works on a real abstract from our actual dataset (not made-up text)
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import load_articles

    real_articles = load_articles("data/raw/articles.csv")
    sample_abstract = real_articles[0].abstract

    print(f"\n--- Real abstract check ---")
    print(f"First 200 chars: {sample_abstract[:200]}")
    print(f"Contains periods: {'.' in sample_abstract}")

    sentences = model._split_into_sentences(sample_abstract)
    print(f"Split into {len(sentences)} sentences")
    print(f"First sentence: {sentences[0]}")