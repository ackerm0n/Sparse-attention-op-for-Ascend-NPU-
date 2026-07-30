#!/usr/bin/env python3
"""Unified entry point for installed-wheel TriangleMix release validation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "installed-correctness",
            "installed-crossover",
            "model-smoke",
            "ttft-abba",
            "parse-counters",
        ),
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ("-h", "--help"):
        parser.parse_args(arguments)
    # Parse only the command token here.  Remaining options belong to the
    # selected command, so ``installed-crossover --help`` must reach that
    # command's parser instead of being consumed as top-level help.
    args = parser.parse_args(arguments[:1])
    remaining = arguments[1:]
    commands: dict[str, Callable[[list[str] | None], int]]
    if args.command == "installed-correctness":
        from .installed_wheel_correctness import main as command_main
    elif args.command == "installed-crossover":
        from .installed_wheel_crossover import main as command_main
    elif args.command == "model-smoke":
        from .model_smoke import main as command_main
    elif args.command == "ttft-abba":
        from .ttft_abba import main as command_main
    else:
        from .counters import main as command_main
    return command_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
