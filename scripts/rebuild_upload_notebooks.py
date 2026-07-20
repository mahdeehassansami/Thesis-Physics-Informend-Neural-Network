from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "rebuild_notebooks.py"), run_name="__main__")
runpy.run_path(
    str(ROOT / "scripts" / "finalize_upload_workspace.py"),
    run_name="__main__",
)
