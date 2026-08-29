#!/usr/bin/env python3
"""Compatibility entry point for ``python3 scope.py ...``."""

from scope.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
