"""Automated tests for model_interface.py's chunking logic.

NB: these tests load the real SentenceTransformer model (via
EmbeddingModel), since chunking depends on the model's actual tokenizer
and token limit --> not a pure unit test in the strictest sense, but the
only way to test this logic against real behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_interface import EmbeddingModel

model = EmbeddingModel()  # loaded once, reused across tests below


def test_short_text_produces_one_chunk():
    """A short text well under the token limit should stay as 1 chunk."""
    short_text = "This drug is used to treat type 2 diabetes."
    chunks = model._split_into_chunks(short_text)

    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    """A text clearly exceeding the token limit should be split into more than 1 chunk."""
    long_text = "Patients receiving this treatment were monitored for adverse events. " * 60
    chunks = model._split_into_chunks(long_text)

    assert len(chunks) > 1


def test_chunks_do_not_exceed_model_limit():
    """No individual chunk should exceed the model's own token limit."""
    long_text = "Patients receiving this treatment were monitored for adverse events. " * 60
    chunks = model._split_into_chunks(long_text)

    for chunk in chunks:
        n_tokens = len(model._model.tokenizer.encode(chunk))  # with special tokens, matches real encoding
        assert n_tokens <= model._max_tokens


def test_sentence_boundaries_are_preserved():
    """Sentences should remain intact when text is split across chunks."""
    sentence = "Patients receiving this treatment were monitored for adverse events."
    text = " ".join([sentence] * 60)

    sentences = model._split_into_sentences(text)
    chunks = model._split_into_chunks(text)

    assert len(chunks) > 1

    for sentence in sentences:
        assert any(sentence in chunk for chunk in chunks)