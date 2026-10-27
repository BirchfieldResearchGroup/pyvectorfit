import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("page", ["README.md", "docs/index.md"])
def test_python_blocks_run_as_written(page):
    path = ROOT / page
    if not path.exists():
        pytest.skip(f"{page} is not in the sdist")
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
    assert blocks
    for code in blocks:
        exec(code, {})
