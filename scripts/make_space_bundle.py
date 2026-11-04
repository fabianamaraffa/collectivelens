"""Prepare a folder with exactly the files a Hugging Face Space needs.

    python scripts/make_space_bundle.py            # creates ./space-bundle
    python scripts/make_space_bundle.py my-folder

Upload or `git push` the content of that folder to your Space. The Space README has a special
header (see docs/huggingface-space-README.md), which is why the GitHub README is not reused.
Your own library (./library) is never copied: a public demo should only hold the fictional sample.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "summary-cache.json")


def main(target: str = "space-bundle") -> Path:
    out = Path(target).resolve()
    protected = {ROOT, *ROOT.parents} | {ROOT / name for name in ("collectivelens", "data", "docs", "tests", "legacy")}
    if out in protected:
        raise SystemExit(f"Refusing to overwrite {out}: choose another folder.")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(ROOT / "collectivelens", out / "collectivelens", ignore=IGNORE)
    shutil.copytree(ROOT / "data" / "sample", out / "data" / "sample", ignore=IGNORE)
    for name in ("app.py", "requirements.txt", "LICENSE"):
        shutil.copy2(ROOT / name, out / name)
    shutil.copy2(ROOT / "docs" / "huggingface-space-README.md", out / "README.md")
    print(f"Space bundle ready in {out}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:2])
