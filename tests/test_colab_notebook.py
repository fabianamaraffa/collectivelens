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


def test_colab_notebook_uses_names_that_exist_in_the_package():
    from collectivelens import ui

    launch_cell = "".join([c for c in load()["cells"] if c["cell_type"] == "code"][-1]["source"])
    assert "from collectivelens.ui import build_app, launch_kwargs" in launch_cell
    assert hasattr(ui, "build_app") and hasattr(ui, "launch_kwargs")
    # Colab only reaches the app through a share link, and launch_kwargs() sets share=False unless told otherwise
    assert "launch_kwargs(share=True)" in launch_cell
