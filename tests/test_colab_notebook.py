import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "colab" / "CollectiveLens.ipynb"


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_colab_notebook_is_clean_and_its_code_compiles():
    nb = load()
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 4  # get the code, install, key, launch
    for cell in code:
        compile("".join(cell["source"]), "<cell>", "exec")
        assert cell["outputs"] == [] and cell["execution_count"] is None  # no saved outputs, no leaked key
    assert "sk-ant" not in NOTEBOOK.read_text(encoding="utf-8")


def test_colab_notebook_starts_the_app_in_its_own_process(monkeypatch):
    from collectivelens.config import Settings
    from collectivelens.ui import launch_kwargs

    cells = [c for c in load()["cells"] if c["cell_type"] == "code"]
    launch_cell = "".join(cells[-1]["source"])
    get_cell = "".join(cells[0]["source"])
    # Colab only reaches the app through a share link; the notebook asks for it with an environment variable
    assert "COLLECTIVELENS_SHARE" in launch_cell and '"app.py"' in launch_cell and "subprocess.Popen" in launch_cell
    monkeypatch.setenv("COLLECTIVELENS_SHARE", "1")
    assert Settings.from_env().share is True and launch_kwargs()["share"] is True
    # an earlier upload without the demo library must be replaced, not reused
    assert "data/sample/library/documents.json" in get_cell
    assert (ROOT / "data" / "sample" / "library" / "documents.json").exists()
