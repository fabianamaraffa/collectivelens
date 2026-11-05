"""Small text helpers: names, language detection, chunking."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from langdetect import DetectorFactory, LangDetectException, detect

# Version 1 got a different language for the same document from one run to the next.
# A fixed seed makes detection repeatable.
DetectorFactory.seed = 0

_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# Short stop word lists (English, Italian, French) used by the keyword matcher and cluster labels.
STOPWORDS = frozenset(
    """
    the and for are but not you all any can had her was one our out has have been from that this with
    they will would there their what about which when your said each she how its than them these those does should could make many much where who whom while being were doing might must shall may
    into more other some such only over also most very just between through during before after under
    il lo la le gli un uno una dei del della delle degli nel nella nelle per con che non sono come alla
    alle dal dalla dalle sul sulla sulle anche piu più tra fra essere stato stati sua suo loro questo
    questa questi queste quello quella ogni molto tutti tutte nostro nostra
    les des une est dans pour que qui sur pas par plus avec sont ont ces cette aux du de en au ne se
    ses leur leurs nous vous ils elles mais comme tout tous toutes entre sans sous aussi etre être fait
    """.split()
)


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "doc"


def normalize_key(name: object) -> str:
    """Match a PDF file name to a metadata row: drop folders and the .pdf/.txt/.md extension."""
    if name is None:
        return ""
    text = str(name).strip().replace("\\", "/").split("/")[-1]
    return re.sub(r"\.(pdf|txt|md)$", "", text, flags=re.IGNORECASE)


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "unknown"
    try:
        return detect(text[:5000])
    except LangDetectException:
        return "unknown"


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS]


def chunk_text(text: str, size: int = 180, overlap: int = 30, min_tail: int = 40) -> list[str]:
    """Split a document into passages of about `size` words, overlapping by `overlap` words.

    Version 1 embedded each whole document, but its model only reads the first ~128 tokens, so most
    of every document was ignored. Passages let every part of a document be found.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    chunks: list[list[str]] = []
    start = 0
    while start < len(words):
        chunks.append(words[start : start + size])
        if start + size >= len(words):
            break
        start += step
    if len(chunks) > 1 and len(chunks[-1]) < min_tail:
        # A very short last passage adds little: fold its new words into the previous one.
        # The first `overlap` words of the tail are already in the previous passage.
        tail = chunks.pop()
        chunks[-1].extend(tail[overlap:])
    return [" ".join(c) for c in chunks]


def sha(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def clamp(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
