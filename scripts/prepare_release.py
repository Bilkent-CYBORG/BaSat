"""Create a clean, shareable project copy without deleting local research files.

Run from any directory with the destination outside the source project:
    python scripts/prepare_release.py --destination /path/to/Basat-ready --archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVELOPER_TESTS = {
    "test_guide_completed_cache.py", "test_guide_presentation.py",
    "test_mhc_confidence_sensitivity.py", "test_cgp_sensor_notebook.py",
    "test_mhc_notebook.py", "test_mhc_seed_search.py", "test_mhc_reference.py",
}


def selected_files(root):
    paths = [root / name for name in (
        ".gitignore", "LICENSE", "README.md", "pyproject.toml", "uv.lock",
        "docs/basat_user_guide.ipynb", "docs/reproduction.md",
        "data/README.md", "data/manifest.json", "data/native_manifest.json",
        "scripts/prepare_release.py",
    )]
    paths.extend(path for path in (root / "basat").rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts
                 and (path.suffix in {".py", ".txt"} or path.name == "py.typed"))
    for family in ("stochastic", "contextual", "distributional", "adversarial"):
        paths.extend(path for path in (root / "data" / family).iterdir()
                     if path.is_file() and path.name != ".DS_Store")
    paths.extend(path for path in (root / "tests").glob("test_*.py")
                 if path.name not in DEVELOPER_TESTS)
    paths.extend((root / "tests/fixtures").glob("*.json"))
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked release input: {path.relative_to(root)}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("A release input escapes the source project.")
    return sorted(set(paths))


def portable_notebook(value, root):
    """Preserve figures and measurements while redacting machine-specific paths."""
    if isinstance(value, dict):
        return {key: portable_notebook(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_notebook(item, root) for item in value]
    if isinstance(value, str):
        value = value.replace(str(root) + "/", "")
        value = re.sub(r"/Users/[^/\s\"']+/Desktop/BOlibrary/", "", value)
        value = re.sub(r"/Users/[^/\s\"']+/", "<user-home>/", value)
        value = value.replace(
            "Use tmp/advers_native_compat_runtime/bin/python with the released NumPy 1.26.4 / SciPy 1.12.0 / scikit-learn 1.7.2 stack.",
            "Use a separate Python 3.11 environment installed with .[advers-source]; see docs/reproduction.md.",
        )
        return value
    return value


def prepare(destination, *, root=ROOT, archive=False):
    root, destination = root.resolve(), Path(destination).expanduser().resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("The release must be outside the original project.")
    if destination.exists():
        raise FileExistsError("Destination already exists; choose a fresh directory.")
    archive_path = destination.with_suffix(".zip")
    if archive and archive_path.exists():
        raise FileExistsError("Release ZIP already exists; choose a fresh destination.")
    paths = selected_files(root)
    destination.mkdir(parents=True)
    manifest = {}
    for path in paths:
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "docs/basat_user_guide.ipynb":
            notebook = portable_notebook(json.loads(path.read_text()), root)
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code" and "source_sha256" in cell.get("metadata", {}):
                    source = cell.get("source", "")
                    source = "".join(source) if isinstance(source, list) else source
                    cell["metadata"]["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        else:
            shutil.copy2(path, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    if archive:
        shutil.make_archive(str(archive_path.with_suffix("")), "zip",
                            root_dir=destination.parent, base_dir=destination.name)
    return {"directory": str(destination), "files": len(manifest),
            "bytes": sum((destination / path).stat().st_size for path in manifest),
            "archive": str(archive_path) if archive else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, archive=args.archive)))


if __name__ == "__main__":
    main()
