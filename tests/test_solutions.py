"""Every reference solution must pass its own notebook checker."""

import contextlib
import io
import runpy
from pathlib import Path

import pytest

SOLUTIONS = sorted((Path(__file__).parent.parent / "solutions").glob("*.py"))


@pytest.mark.parametrize("path", SOLUTIONS, ids=lambda p: p.stem)
def test_solution_passes(path):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(path), run_name="__main__")
    assert "All checks passed!" in out.getvalue(), out.getvalue()
