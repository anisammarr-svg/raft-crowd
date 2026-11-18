"""Validate a generated notebook's schema, Python cells and embedded source pins."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import nbformat

from build_raftcrowd_notebook import ROOT, source_files


def check(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"notebook-code-cell-{index}", "exec")
        if cell.outputs or cell.execution_count is not None:
            raise ValueError("Distributed notebook must contain no execution outputs")
    tree = ast.parse(code_cells[0].source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "FILES" for t in n.targets))
    bundle = ast.literal_eval(assignment.value)
    expected = source_files()
    if set(bundle) != set(expected):
        raise ValueError("Embedded file set differs from canonical sources")
    for name, text in expected.items():
        digest = hashlib.sha256(text.encode()).hexdigest()
        if bundle[name]["text"] != text or bundle[name]["sha256"] != digest:
            raise ValueError(f"Embedded source mismatch: {name}")
        if notebook.metadata["raftcrowd_source_sha256"][name] != digest:
            raise ValueError(f"Notebook metadata pin mismatch: {name}")
    print(f"Valid schema; {len(code_cells)} code cells compile; {len(bundle)} embedded files match source pins; outputs empty")
    return notebook


if __name__ == "__main__":
    check(ROOT / "output/jupyter-notebook/RAFTCrowdFlow_from_scratch.ipynb")
