from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_testing_msp_import_cell_runs_from_notebooks_directory() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "testing_msp.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    import_cell = notebook["cells"][1]
    source = "".join(import_cell["source"])

    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=repo_root / "notebooks",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
