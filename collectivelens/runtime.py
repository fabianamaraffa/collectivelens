"""Put the pieces together: settings, embedder, library and Claude client."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Category, Settings, load_categories
from .embedders import get_embedder
from .library import Library, open_library
from .llm import make_client


@dataclass
class Runtime:
    settings: Settings
    library: Library
    categories: list[Category]
    client: object | None
    warning: str | None  # shown in the app when the semantic model could not be used

    @property
    def has_claude(self) -> bool:
        return self.client is not None


def load_runtime(settings: Settings | None = None, *, api_key: str | None = None) -> Runtime:
    settings = settings or Settings.from_env()
    categories = load_categories(settings.categories_file)
    embedder, warning = get_embedder(settings.embedder, settings.embedding_model,
                                     settings.query_prefix, settings.passage_prefix, settings.embedding_device)
    library = open_library(settings.resolve_library_dir(), embedder, categories,
                           chunk_words=settings.chunk_words, chunk_overlap=settings.chunk_overlap)
    return Runtime(settings, library, categories, make_client(api_key), warning)
