from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for script in (
    "build_colab_notebook.py",
    "finalize_colab_notebooks.py",
    "finalize_colab_source.py",
):
    runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
