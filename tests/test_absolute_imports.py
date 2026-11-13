"""Modules under `src/moveitmoveit` and `scripts` name `moveitmoveit` imports absolutely. No simulator."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_relative_imports_in_package_or_scripts():
    sources = sorted((ROOT / "src" / "moveitmoveit").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    assert sources
    relative = [
        f"{path.relative_to(ROOT)}:{node.lineno}"
        for path in sources
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert not relative, f"relative imports remain: {relative}"
