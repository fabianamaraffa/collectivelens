import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("make_space_bundle", ROOT / "scripts" / "make_space_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_has_what_a_space_needs_and_nothing_private(tmp_path):
    (ROOT / "library").mkdir(exist_ok=True)  # simulates a private library next to the code
    marker = ROOT / "library" / "documents.json"
    created = not marker.exists()
    if created:
        marker.write_text("[]")
    try:
        out = load().main(str(tmp_path / "bundle"))
    finally:
        if created:
            marker.unlink()
            (ROOT / "library").rmdir()
    assert (out / "app.py").exists() and (out / "requirements.txt").exists()
    assert (out / "data" / "sample" / "library" / "documents.json").exists()
    assert not (out / "library").exists() and not (out / ".env").exists()
    assert (out / "README.md").read_text(encoding="utf-8").startswith("---\ntitle: CollectiveLens")


def test_bundle_refuses_to_overwrite_the_project(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        load().main(str(ROOT))
