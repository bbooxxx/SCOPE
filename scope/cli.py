"""Command-line entry point kept separate from the modeling modules."""

from __future__ import annotations

from typing import Optional, Sequence

from .core import main as _core_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _core_main(argv)
