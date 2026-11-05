"""The library: documents, their passages, topics and clusters.

On disk a library is a folder with `documents.json` (metadata, summaries and full text) and
`meta.json`. Everything derived from it (passages, vectors, topics, clusters) is computed when the
library is opened, so it always matches the embedder in use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from . import __version__
from .config import Category
from .embedders import Embedder, as_dense, cosine_matrix, mean_vectors
from .ingest import RawDocument
from .summarise import Summariser
from .textutils import STOPWORDS, chunk_text, sha

log = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    filename: str
    title: str
    org: str
    year: int | None
    language: str
    text: str
    summary: str = ""
    summary_source: str = "none"
    source_url: str = ""
    gold_topic: str = ""


@dataclass
class Chunk:
    doc_index: int
    index: int
    text: str


@dataclass
class TopicAssignment:
    name: str
    score: float      # cosine similarity with the category description
    second: str
    margin: float     # gap to the second category: small means the choice is a toss-up


class Library:
    def __init__(self, docs: list[Document], embedder: Embedder, categories: list[Category], *,
                 name: str = "library", is_sample: bool = False, chunk_words: int = 180,
                 chunk_overlap: int = 30, cache_dir: Path | None = None, clusters: int | str = "auto") -> None:
        if not docs:
            raise ValueError("The library has no documents")
        self.docs = docs
        self.embedder = embedder
        self.categories = categories
        self.name = name
        self.is_sample = is_sample
        self.cache_dir = cache_dir
        self._chunk_words, self._chunk_overlap, self._k = chunk_words, chunk_overlap, clusters
        self._compute()

    # ------------------------------------------------------------ derived data
    def _compute(self) -> None:
        self.chunks: list[Chunk] = [
            Chunk(i, j, t)
            for i, d in enumerate(self.docs)
            for j, t in enumerate(chunk_text(d.text, self._chunk_words, self._chunk_overlap))
        ]
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.chunk_vecs = self._embed_passages(texts)
        groups: list[list[int]] = [[] for _ in self.docs]
        for n, c in enumerate(self.chunks):
            groups[c.doc_index].append(n)
        self.doc_chunks = groups
        self.doc_vecs = mean_vectors(self.chunk_vecs, groups)

        cat_vecs = self.embedder.encode_queries([c.description for c in self.categories])
        sims = cosine_matrix(self.doc_vecs, cat_vecs)
        self.topics: list[TopicAssignment] = []
        for row in sims:
            order = np.argsort(-row)
            first, second = int(order[0]), int(order[1])
            self.topics.append(TopicAssignment(
                self.categories[first].name, float(row[first]),
                self.categories[second].name, float(row[first] - row[second])))

        self.cluster_ids, self.cluster_terms = self._cluster()

    def _embed_passages(self, texts: list[str]):
        if not self.embedder.semantic or self.cache_dir is None:
            return self.embedder.encode_passages(texts)
        key = sha(self.embedder.name, *texts)[:20]
        path = self.cache_dir / f"embeddings-{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.embedder.encode_passages(texts)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        except OSError:
            log.info("Could not save the embedding cache (read-only folder?)")
        return vectors

    def _cluster(self) -> tuple[np.ndarray, dict[int, list[str]]]:
        vectors = as_dense(self.doc_vecs)
        n = len(self.docs)
        if n < 4:
            return np.zeros(n, dtype=int), {0: self._terms(np.zeros(n, dtype=int)).get(0, [])}
        if self._k == "auto":
            best_k, best = 2, -1.0
            for k in range(2, min(8, n - 1) + 1):
                labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(vectors, labels, metric="cosine")
                if score > best:
                    best_k, best = k, score
            k = best_k
        else:
            k = max(1, min(int(self._k), n))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
        return labels, self._terms(labels)

    def _terms(self, labels: np.ndarray) -> dict[int, list[str]]:
        """Keywords that set each cluster apart from the others (cluster-level TF-IDF)."""
        ids = sorted(set(int(x) for x in labels))
        joined = [" ".join(d.text for d, lab in zip(self.docs, labels) if lab == i) for i in ids]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=list(STOPWORDS),
                                  token_pattern=r"(?u)\b[^\W\d_]{4,}\b",
                                  max_df=0.6 if len(ids) >= 3 else 1.0)
            matrix = vec.fit_transform(joined)
        except ValueError:  # nothing left after filtering
            return {i: [] for i in ids}
        words = np.array(vec.get_feature_names_out())
        out = {}
        for row, i in enumerate(ids):
            scores = matrix[row].toarray().ravel()
            top = scores.argsort()[::-1][:8]
            out[i] = [str(words[t]) for t in top if scores[t] > 0]
        return out

    # ------------------------------------------------------------ views
    @property
    def topic_names(self) -> list[str]:
        return [c.name for c in self.categories]

    def orgs(self) -> list[str]:
        return sorted({d.org for d in self.docs})

    def years(self) -> list[int]:
        return sorted({d.year for d in self.docs if d.year})

    def languages(self) -> list[str]:
        return sorted({d.language for d in self.docs})

    @property
    def semantic(self) -> bool:
        return bool(self.embedder.semantic)

    # ------------------------------------------------------------ saving and opening
    def save(self, out_dir: Path | str) -> None:
        save_documents(self.docs, out_dir, is_sample=self.is_sample, name=self.name)


def save_documents(docs: list[Document], out_dir: Path | str, *, is_sample: bool = False,
                   name: str = "library") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "documents.json").write_text(
        json.dumps([asdict(d) for d in docs], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "name": name, "is_sample": is_sample, "collectivelens_version": __version__,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": len(docs),
    }, indent=1), encoding="utf-8")


def load_documents(library_dir: Path | str) -> tuple[list[Document], dict]:
    folder = Path(library_dir)
    docs_file = folder / "documents.json"
    if not docs_file.exists():
        raise FileNotFoundError(f"No library found in {folder} (documents.json is missing). "
                                "Build one with: python -m collectivelens build --help")
    docs = [Document(**d) for d in json.loads(docs_file.read_text(encoding="utf-8"))]
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    return docs, meta


def open_library(library_dir: Path | str, embedder: Embedder, categories: list[Category], *,
                 chunk_words: int = 180, chunk_overlap: int = 30) -> Library:
    docs, meta = load_documents(library_dir)
    return Library(docs, embedder, categories, name=meta.get("name", Path(library_dir).name),
                   is_sample=bool(meta.get("is_sample", False)), chunk_words=chunk_words,
                   chunk_overlap=chunk_overlap, cache_dir=Path(library_dir) / "cache")


def build_documents(raw: list[RawDocument], summariser: Summariser | None) -> list[Document]:
    """Turn ingested files into library documents, adding a summary to each."""
    docs = []
    for r in raw:
        summary, source = "", "none"
        if summariser is not None:
            result = summariser.summarise(r.filename, r.title, r.text)
            summary, source = result.text, result.source
        docs.append(Document(id=r.id, filename=r.filename, title=r.title, org=r.org, year=r.year,
                             language=r.language, text=r.text, summary=summary, summary_source=source,
                             source_url=r.source_url, gold_topic=r.gold_topic))
    return docs
