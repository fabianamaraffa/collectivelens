"""Settings. Every value can be changed with an environment variable, so no code edit is needed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
DEFAULT_CATEGORIES_FILE = PACKAGE_DIR / "categories.yaml"
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_LIBRARY_DIR = SAMPLE_DIR / "library"
USER_LIBRARY_DIR = ROOT / "library"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    # Claude model used for summaries and answers. Haiku is the small, cheap, fast tier.
    claude_model: str = "claude-haiku-5-5"
    # "auto" tries the semantic model and falls back to keyword matching; "semantic" fails if the
    # model cannot load; "lite" never downloads a model.
    embedder: str = "auto"
    embedding_model: str = "BAAI/bge-m3"
    query_prefix: str = ""
    passage_prefix: str = ""
    chunk_words: int = 180
    chunk_overlap: int = 30
    summary_max_chars: int = 30000
    summary_max_tokens: int = 400
    answer_max_tokens: int = 900
    passages_for_answer: int = 6
    categories_file: Path = DEFAULT_CATEGORIES_FILE
    library_dir: Path | None = None
    share: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        library = os.getenv("COLLECTIVELENS_LIBRARY")
        return cls(
            claude_model=_env("COLLECTIVELENS_CLAUDE_MODEL", cls.claude_model),
            embedder=_env("COLLECTIVELENS_EMBEDDER", cls.embedder).lower(),
            embedding_model=_env("COLLECTIVELENS_EMBEDDING_MODEL", cls.embedding_model),
            query_prefix=os.getenv("COLLECTIVELENS_QUERY_PREFIX", ""),
            passage_prefix=os.getenv("COLLECTIVELENS_PASSAGE_PREFIX", ""),
            categories_file=Path(_env("COLLECTIVELENS_CATEGORIES", str(DEFAULT_CATEGORIES_FILE))),
            library_dir=Path(library) if library else None,
            share=_env("COLLECTIVELENS_SHARE", "0").lower() in {"1", "true", "yes"},
        )

    def resolve_library_dir(self) -> Path:
        """Your own library if it exists, otherwise the bundled sample library."""
        if self.library_dir:
            return self.library_dir
        if (USER_LIBRARY_DIR / "documents.json").exists():
            return USER_LIBRARY_DIR
        return SAMPLE_LIBRARY_DIR


@dataclass(frozen=True)
class Category:
    name: str
    description: str


def load_categories(path: Path | str | None = None) -> list[Category]:
    path = Path(path) if path else DEFAULT_CATEGORIES_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("categories") or []
    categories = [Category(str(i["name"]).strip(), str(i.get("description") or i["name"]).strip()) for i in items]
    if len(categories) < 2:
        raise ValueError(f"{path} needs at least two categories")
    names = [c.name for c in categories]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} has duplicate category names")
    return categories
