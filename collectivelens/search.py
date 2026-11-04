"""Search the library by meaning (or by words, in lite mode)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape

import numpy as np

from .embedders import cosine_matrix
from .library import Document, Library
from .textutils import content_words


@dataclass
class Passage:
    doc_index: int
    chunk_index: int
    text: str
    score: float


@dataclass
class Hit:
    doc: Document
    doc_index: int
    score: float
    relative: float                       # score divided by the best score: 1.0 is the top hit
    passages: list[Passage] = field(default_factory=list)
    topic: str = ""
    cluster: int = 0


def _allowed(lib: Library, topics, orgs, year_from, year_to, languages) -> np.ndarray:
    keep = np.ones(len(lib.docs), dtype=bool)
    for i, d in enumerate(lib.docs):
        if topics and lib.topics[i].name not in topics:
            keep[i] = False
        if orgs and d.org not in orgs:
            keep[i] = False
        if languages and d.language not in languages:
            keep[i] = False
        if year_from and (d.year is None or d.year < year_from):
            keep[i] = False
        if year_to and (d.year is None or d.year > year_to):
            keep[i] = False
    return keep


def passage_scores(lib: Library, query: str) -> np.ndarray:
    qv = lib.embedder.encode_queries([query])
    return cosine_matrix(qv, lib.chunk_vecs)[0]


def search(lib: Library, query: str = "", *, topics: list[str] | None = None, orgs: list[str] | None = None,
           year_from: int | None = None, year_to: int | None = None, languages: list[str] | None = None,
           sort: str = "relevance", limit: int = 10, passages_per_doc: int = 2,
           min_relative: float | None = None) -> list[Hit]:
    """Documents that match the query and the filters. With an empty query, the filtered documents
    are listed (newest first), which works as browsing.

    `min_relative` hides documents whose score is below that share of the best score. By default it is
    0 for semantic search and 0.2 for keyword search, where weak matches are noise (one shared word part)."""
    keep = _allowed(lib, topics, orgs, year_from, year_to, languages)
    query = (query or "").strip()
    hits: list[Hit] = []
    if not query:
        order = sorted(np.flatnonzero(keep), key=lambda i: (-(lib.docs[i].year or 0), lib.docs[i].title))
        for i in order[:limit]:
            hits.append(Hit(lib.docs[i], int(i), 0.0, 0.0, [], lib.topics[i].name, int(lib.cluster_ids[i])))
        return hits

    scores = passage_scores(lib, query)
    best_by_doc: dict[int, list[Passage]] = {}
    for n, c in enumerate(lib.chunks):
        if not keep[c.doc_index]:
            continue
        best_by_doc.setdefault(c.doc_index, []).append(Passage(c.doc_index, c.index, c.text, float(scores[n])))
    for i, ps in best_by_doc.items():
        ps.sort(key=lambda p: -p.score)
        hits.append(Hit(lib.docs[i], i, ps[0].score, 0.0, ps[:passages_per_doc], lib.topics[i].name,
                        int(lib.cluster_ids[i])))
    hits = [h for h in hits if h.score > 0]
    top = max((h.score for h in hits), default=0.0)
    for h in hits:
        h.relative = h.score / top if top else 0.0
    floor = (0.0 if lib.semantic else 0.2) if min_relative is None else min_relative
    hits = [h for h in hits if h.relative >= floor]
    if sort == "newest":
        hits.sort(key=lambda h: (-(h.doc.year or 0), -h.score))
    else:
        hits.sort(key=lambda h: -h.score)
    return hits[:limit]


def highlight(text: str, query: str, limit: int = 420) -> str:
    """HTML-safe excerpt with the query words in bold."""
    # "amp" and "quot" would match inside HTML entities such as &amp; once the text is escaped.
    words = {w for w in content_words(query) if len(w) > 2} - {"amp", "quot"}
    excerpt = " ".join(text.split())
    if len(excerpt) > limit:
        first = min((m.start() for w in words for m in re.finditer(re.escape(w), excerpt, re.I)), default=0)
        start = max(0, first - limit // 4)
        excerpt = ("…" if start else "") + excerpt[start : start + limit].rsplit(" ", 1)[0] + "…"
    safe = escape(excerpt)
    if words:
        pattern = re.compile("(" + "|".join(re.escape(escape(w)) for w in sorted(words, key=len, reverse=True)) + ")", re.I)
        safe = pattern.sub(r"<mark>\1</mark>", safe)
    return safe
