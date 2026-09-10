"""Launch the bundled notebook runtime with ``python -m bayopy notebook``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch BayOPy's notebook command to its installed Jupyter runtime."""
    parser = argparse.ArgumentParser(prog="bayopy")
    parser.add_argument("command", choices=("notebook",), help="Open the notebook interface")
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Notebook path and optional notebook-server arguments",
    )
    args = parser.parse_args(argv)
    return subprocess.run(
        [sys.executable, "-m", "jupyter", "lab", *args.arguments],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
