from pathlib import Path

import pytest

from jaxley_refactored.reporting.runs import RunDirectory


def test_missing_run_directory_has_actionable_error(tmp_path: Path):
    output = RunDirectory(tmp_path, "active-run")
    (output.path / "checkpoints").rmdir()
    (output.path / ".claim").unlink()
    output.path.rmdir()

    with pytest.raises(RuntimeError, match="older background fit"):
        output.append_metrics({"epoch": 0})
