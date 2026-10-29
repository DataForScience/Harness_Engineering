"""The %%writefile cells in the notebooks must rebuild reference/ exactly.

If you edit a module in reference/, update the matching notebook cells (or the
other way around); this test tells you when the two have drifted apart.
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MAGIC = re.compile(r"^%%writefile( -a)? harness/(\w+)\.py\n")


def built_modules() -> dict[str, str]:
    """Replay every %%writefile cell in notebook order, as IPython would."""
    files = defaultdict(str)
    for nb in sorted(ROOT.glob("[0-9][0-9] - *.ipynb")):
        for cell in json.loads(nb.read_text())["cells"]:
            src = "".join(cell["source"])
            m = MAGIC.match(src) if cell["cell_type"] == "code" else None
            if m:
                body = src[m.end():] + "\n"
                files[m.group(2)] = files[m.group(2)] + body if m.group(1) else body
    return files


BUILT = built_modules()


@pytest.mark.parametrize("name", sorted(p.stem for p in (ROOT / "reference").glob("*.py")))
def test_notebooks_rebuild_reference(name):
    assert name in BUILT, f"no notebook writes harness/{name}.py"
    reference = (ROOT / "reference" / f"{name}.py").read_text()
    assert ast.dump(ast.parse(BUILT[name])) == ast.dump(ast.parse(reference))
