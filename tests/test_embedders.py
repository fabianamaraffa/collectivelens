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
        def __init__(self, name, device=None):
            seen["name"], seen["device"] = name, device

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
    assert emb.semantic is True and seen["name"] == "fake/model" and seen["device"] == "cpu"


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


def test_a_torch_crash_in_the_probe_falls_back_instead_of_closing_the_app(monkeypatch):
    monkeypatch.setattr(embedders, "probe_torch", lambda *a, **k: "PyTorch crashed on a test calculation (signal 6)")
    monkeypatch.setattr(embedders, "SemanticEmbedder", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load")))
    emb, warning = get_embedder("auto")
    assert isinstance(emb, LiteEmbedder) and "signal 6" in warning and "Python 3.12" in warning
    with pytest.raises(RuntimeError, match="signal 6"):
        get_embedder("semantic")


def test_probe_reports_a_crashing_child_process(monkeypatch):
    class Result:
        returncode, stderr = -6, "terminate called\nabort"

    monkeypatch.setattr(embedders.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(embedders.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.delenv("COLLECTIVELENS_SKIP_PROBE", raising=False)
    assert "signal 6" in embedders.probe_torch()


def test_probe_is_silent_without_torch_or_when_skipped(monkeypatch):
    monkeypatch.setattr(embedders.importlib.util, "find_spec", lambda name: None)
    assert embedders.probe_torch() is None
    monkeypatch.setenv("COLLECTIVELENS_SKIP_PROBE", "1")
    assert embedders.probe_torch() is None


def _fake_torch(root, body):
    pkg = root / "torch"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    return root


def test_probe_with_a_real_child_process(tmp_path, monkeypatch):
    """Runs the real subprocess against stand-in `torch` packages: one that works, one that aborts."""
    import importlib

    monkeypatch.delenv("COLLECTIVELENS_SKIP_PROBE", raising=False)
    works = _fake_torch(tmp_path / "ok",
                        "import types\nnn = types.SimpleNamespace(functional=types.SimpleNamespace(linear=lambda a, b: 1))\n"
                        "ones = lambda *shape: 0\n")
    aborts = _fake_torch(tmp_path / "bad", "import os\nos.abort()\n")
    for folder, should_work in ((works, True), (aborts, False)):
        monkeypatch.syspath_prepend(str(folder))
        monkeypatch.setenv("PYTHONPATH", str(folder))
        sys.modules.pop("torch", None)
        importlib.invalidate_caches()
        result = embedders.probe_torch(timeout=60)
        assert (result is None) is should_work, result
        if not should_work:
            assert "signal 6" in result
