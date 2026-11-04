import sys
import types

import numpy as np
import pytest

from collectivelens import embedders
from collectivelens.embedders import LiteEmbedder, SemanticEmbedder, cosine_matrix, get_embedder, mean_vectors


def test_lite_matches_shared_words():
    lite = LiteEmbedder().fit(["youth participation in policy", "climate change and energy", "student mobility abroad"])
    docs = lite.encode_passages(["youth participation in policy", "climate change and energy", "student mobility abroad"])
    query = lite.encode_queries(["participation of young people"])
    assert int(np.argmax(cosine_matrix(query, docs))) == 0


def test_semantic_embedder_uses_prefixes_and_normalises(monkeypatch):
    seen = {}

    class FakeModel:
        def __init__(self, name):
            seen["name"] = name

        def encode(self, texts, **kwargs):
            seen["texts"], seen["kwargs"] = list(texts), kwargs
            return np.array([[3.0, 4.0]] * len(texts))

    module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    emb = SemanticEmbedder("fake/model", query_prefix="query: ", passage_prefix="passage: ")
    emb.encode_queries(["hello"])
    assert seen["texts"] == ["query: hello"] and seen["kwargs"]["normalize_embeddings"] is True
    emb.encode_passages(["world"])
    assert seen["texts"] == ["passage: world"]
    assert emb.semantic is True and seen["name"] == "fake/model"


def test_auto_mode_falls_back_with_a_warning(monkeypatch):
    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(embedders, "SemanticEmbedder", boom)
    emb, warning = get_embedder("auto")
    assert isinstance(emb, LiteEmbedder) and "no network" in warning


def test_semantic_mode_does_not_hide_the_failure(monkeypatch):
    monkeypatch.setattr(embedders, "SemanticEmbedder", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(OSError):
        get_embedder("semantic")


def test_lite_mode_never_loads_a_model():
    emb, warning = get_embedder("lite")
    assert isinstance(emb, LiteEmbedder) and warning is None


def test_mean_vectors_are_normalised():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    out = mean_vectors(vecs, [[0, 1], [2]])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
