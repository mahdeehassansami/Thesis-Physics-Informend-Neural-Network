from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "build_colab_notebook.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "finalize_colab_notebooks.py"), run_name="__main__")
