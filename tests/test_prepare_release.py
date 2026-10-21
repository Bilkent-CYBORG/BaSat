"""A shareable project never includes private campaigns or personal paths."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_release", ROOT / "scripts/prepare_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_release_allowlist_has_only_shareable_project_files():
    paths = [path.relative_to(ROOT) for path in release.selected_files(ROOT)]
    assert {path.parts[0] for path in paths} <= {
        "basat", "docs", "data", "tests", "scripts", ".gitignore",
        "LICENSE", "README.md", "pyproject.toml", "uv.lock",
    }
    assert Path("docs/basat_user_guide.ipynb") in paths
    assert Path("basat/algorithms/Stochastic/_agvs_source_LICENSE.txt") in paths
    assert Path("basat/py.typed") in paths
    assert not any("__pycache__" in path.parts or ".git" in path.parts for path in paths)
    assert not any(path.name in release.DEVELOPER_TESTS for path in paths)


def test_clean_copy_preserves_figures_and_source_and_redacts_paths(tmp_path):
    original = ROOT / "docs/basat_user_guide.ipynb"
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    target = tmp_path / "clean-release"
    report = release.prepare(target, root=ROOT, archive=True)
    assert report["files"] > 100 and report["bytes"] < 15_000_000
    assert Path(report["archive"]).is_file()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert not (target / ".git").exists() and not (target / "results").exists()
    raw = (target / "docs/basat_user_guide.ipynb").read_text()
    assert "/Users/" not in raw
    assert "tmp/advers_native_compat_runtime/bin/python" not in raw
    notebook = nbformat.reads(raw, as_version=4)
    nbformat.validate(notebook)
    source = nbformat.read(original, as_version=4)
    assert [cell.get("attachments") for cell in notebook.cells] == [
        cell.get("attachments") for cell in source.cells]
    assert notebook.cells[-1].id == "cgp-mhc-maximum"
    for cell in notebook.cells:
        if cell.cell_type == "code" and "source_sha256" in cell.metadata:
            assert cell.metadata.source_sha256 == hashlib.sha256(cell.source.encode()).hexdigest()
    assert (target / "basat/algorithms/Stochastic/AG_VS.py").read_bytes() == (
        ROOT / "basat/algorithms/Stochastic/AG_VS.py").read_bytes()


def test_existing_destinations_are_never_overwritten(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("Keep my files.")
    with pytest.raises(FileExistsError):
        release.prepare(existing, root=ROOT)
    assert marker.read_text() == "Keep my files."
    with pytest.raises(ValueError):
        release.prepare(ROOT / "nested-release", root=ROOT)


def test_portable_notebook_keeps_scientific_numbers_and_image_data():
    original = {"source": "/Users/example/Desktop/BOlibrary/basat/experiment.py",
                "arrays": [1.0, 2, None], "image/png": "abc123base64"}
    clean = release.portable_notebook(original, ROOT)
    assert clean["source"] == "basat/experiment.py"
    assert clean["arrays"] == original["arrays"]
    assert clean["image/png"] == original["image/png"]
    assert json.dumps(original) != json.dumps(clean)
