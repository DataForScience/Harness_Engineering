"""Helpers for building the harness package from the notebooks.

Notebooks 02-06 each write one module into ``harness/`` with ``%%writefile``.
Finished copies live in ``reference/``. If you skipped a notebook, or your copy
of a module is broken, ``ensure`` puts the reference version in place.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from types import ModuleType

PACKAGE = Path(__file__).resolve().parent
REFERENCE = PACKAGE.parent / "reference"

# Module -> the notebook that builds it
BUILT_IN = {"permissions": "02", "context": "03", "guardrails": "04", "verify": "05",
            "supervisor": "06", "analyst": "06"}


def load(name: str) -> ModuleType:
    """Import ``harness.<name>``, re-reading the file if it was imported before."""
    importlib.invalidate_caches()
    full = f"harness.{name}"
    if full in sys.modules:
        return importlib.reload(sys.modules[full])
    return importlib.import_module(full)


def restore(name: str) -> ModuleType:
    """Replace ``harness/<name>.py`` with the reference version and import it."""
    target = PACKAGE / f"{name}.py"
    if target.exists():
        backup = target.with_suffix(".py.broken")
        shutil.copy(target, backup)
        print(f"Your harness/{name}.py was saved as {backup.name}")
    shutil.copy(REFERENCE / f"{name}.py", target)
    print(f"Restored harness/{name}.py from reference/ (built in notebook {BUILT_IN[name]})")
    return load(name)


def ensure(*names: str) -> None:
    """Make sure each module exists and imports; restore it from reference/ if not."""
    for name in names:
        if not (PACKAGE / f"{name}.py").exists():
            print(f"harness/{name}.py is missing (built in notebook {BUILT_IN[name]}).")
            restore(name)
            continue
        try:
            load(name)
        except Exception as e:  # noqa: BLE001
            print(f"harness/{name}.py does not import: {type(e).__name__}: {e}")
            restore(name)
