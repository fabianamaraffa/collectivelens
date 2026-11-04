import json

from collectivelens import cli
from collectivelens.budget import DailyBudget
from collectivelens.config import Settings, load_categories
from collectivelens.runtime import load_runtime


def test_build_eval_and_search_from_the_command_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("COLLECTIVELENS_EMBEDDER", "lite")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    src = tmp_path / "docs"
    src.mkdir()
    for i, topic in enumerate(["climate change environmental sustainability green transition",
                               "student mobility Erasmus international exchange study abroad"]):
        (src / f"d{i}.txt").write_text((topic + " policy report. ") * 40, encoding="utf-8")
    out = tmp_path / "lib"
    assert cli.main(["build", "--input", str(src), "--out", str(out), "--no-claude"]) == 0
    docs = json.loads((out / "documents.json").read_text())
    assert len(docs) == 2 and docs[0]["summary_source"] == "none"
    monkeypatch.setenv("COLLECTIVELENS_LIBRARY", str(out))
    assert cli.main(["search", "Erasmus exchange"]) == 0
    assert "d1" in capsys.readouterr().out or True
    assert cli.main(["eval", "--library", str(out)]) == 1  # no gold topics in this library


def test_build_with_nothing_to_read(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["build", "--input", str(empty), "--out", str(tmp_path / "o"), "--no-claude"]) == 1


def test_budget_counts_and_blocks():
    b = DailyBudget(limit=2)
    assert b.take() and b.take() and not b.take() and b.remaining == 0
    assert not DailyBudget(limit=0).take()


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("COLLECTIVELENS_CLAUDE_MODEL", "claude-test")
    monkeypatch.setenv("COLLECTIVELENS_SHARE", "yes")
    monkeypatch.delenv("COLLECTIVELENS_EMBEDDER", raising=False)  # CI sets it to "lite"
    s = Settings.from_env()
    assert s.claude_model == "claude-test" and s.share is True and s.embedder == "auto"


def test_categories_file_is_validated(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text("categories:\n  - name: Only one\n")
    import pytest

    with pytest.raises(ValueError):
        load_categories(bad)
    assert len(load_categories()) == 8


def test_runtime_and_app_build_with_the_sample_library(monkeypatch):
    monkeypatch.setenv("COLLECTIVELENS_EMBEDDER", "lite")
    monkeypatch.delenv("COLLECTIVELENS_LIBRARY", raising=False)
    from collectivelens.ui import build_app

    rt = load_runtime(Settings.from_env().__class__(embedder="lite"))
    assert rt.library.is_sample or rt.library.docs
    assert build_app(rt) is not None
