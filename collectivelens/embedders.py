"""Turn text into vectors so that texts about the same thing end up close together.

Two embedders share one interface:

* SemanticEmbedder uses a multilingual model (downloaded once). It matches by meaning, also across
  languages: a question in Italian can find a document in English.
* LiteEmbedder needs no download. It matches by shared words and word parts, so it only finds
  documents that use similar words. It keeps the app usable offline and in tests.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .textutils import content_words

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    semantic: bool

    def fit(self, texts: list[str]) -> "Embedder": ...
    def encode_passages(self, texts: list[str]): ...
    def encode_queries(self, texts: list[str]): ...


def _prepare(text: str) -> str:
    return " ".join(content_words(text))


class LiteEmbedder:
    semantic = False
    name = "lite (keyword matching, no model download)"

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: list[str]) -> "LiteEmbedder":
        # Word parts that occur in nearly every passage say little, so they are dropped (max_df). In a very
        # small or very uniform library that can leave nothing, so the filter is then relaxed.
        for max_df in (0.85, 1.0):
            try:
                self._vectorizer = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_df=max_df,
                    preprocessor=_prepare, dtype=np.float32,
                ).fit(texts)
                return self
            except ValueError:
                continue
        raise ValueError("The documents contain no usable text")

    def _transform(self, texts: list[str]):
        if self._vectorizer is None:
            raise RuntimeError("LiteEmbedder must be fitted first")
        return self._vectorizer.transform(texts)

    def encode_passages(self, texts: list[str]):
        return self._transform(texts)

    def encode_queries(self, texts: list[str]):
        return self._transform(texts)


class SemanticEmbedder:
    semantic = True

    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = "",
                 max_seq_length: int = 512) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed

        self.model_name = model_name
        self.name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = SentenceTransformer(model_name)
        # Passages are about 250 tokens; a limit of 512 keeps CPU use low.
        self._model.max_seq_length = max_seq_length

    def fit(self, texts: list[str]) -> "SemanticEmbedder":
        return self  # nothing to learn from the library

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vectors = self._model.encode(
            [prefix + t for t in texts],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)


def get_embedder(mode: str = "auto", model_name: str = "BAAI/bge-m3", query_prefix: str = "",
                 passage_prefix: str = "") -> tuple[Embedder, str | None]:
    """Return an embedder and, when the semantic model could not be used, a warning to show."""
    mode = (mode or "auto").lower()
    if mode == "lite":
        return LiteEmbedder(), None
    try:
        return SemanticEmbedder(model_name, query_prefix, passage_prefix), None
    except Exception as exc:  # missing package, no network, out of memory...
        if mode == "semantic":
            raise
        reason = f"{type(exc).__name__}: {exc}".strip()
        log.warning("Semantic model unavailable (%s). Using keyword matching.", reason)
        return LiteEmbedder(), (
            "The semantic model could not be loaded, so search uses keyword matching only "
            f"(reason: {reason[:160]})."
        )


# ---- vector helpers that work for dense arrays and sparse matrices ----

def as_dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def cosine_matrix(a, b) -> np.ndarray:
    """Cosine similarity between every row of a and every row of b. Rows must be L2-normalised."""
    if sparse.issparse(a) or sparse.issparse(b):
        a_s = a if sparse.issparse(a) else sparse.csr_matrix(a)
        b_s = b if sparse.issparse(b) else sparse.csr_matrix(b)
        return (a_s @ b_s.T).toarray()
    return np.asarray(a) @ np.asarray(b).T


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def mean_vectors(vectors, groups: list[list[int]]) -> np.ndarray:
    """One normalised vector per group of rows (here: the passages of one document)."""
    rows = []
    for idx in groups:
        part = vectors[idx]
        mean = np.asarray(part.mean(axis=0)).reshape(-1)
        rows.append(mean)
    return normalize_rows(np.vstack(rows).astype(np.float32))
