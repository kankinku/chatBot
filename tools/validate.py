#!/usr/bin/env python3
"""Stable validation bridge across legacy and canonical repository layouts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT / "Chatbot_v6"


def is_legacy_layout() -> bool:
    return LEGACY_ROOT.is_dir()


def run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def compile_sources() -> None:
    if is_legacy_layout():
        run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "Server/backend",
                "tests/unit",
                "modules",
                "config",
                "scripts",
            ],
            cwd=LEGACY_ROOT,
        )
        return

    run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "services/gateway",
            "services/inference",
            "src/chatbot",
            "tests/unit",
            "config",
            "scripts",
        ]
    )


def run_unit_tests() -> None:
    cwd = LEGACY_ROOT if is_legacy_layout() else REPO_ROOT
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=cwd,
    )


def install_ci_dependencies() -> None:
    lock = (
        LEGACY_ROOT / "requirements-ci.lock"
        if is_legacy_layout()
        else REPO_ROOT / "requirements-ci.lock"
    )
    if not lock.is_file():
        raise SystemExit(f"CI dependency lock not found: {lock}")

    run([sys.executable, "-m", "pip", "install", "pip==26.2.1"])
    run([sys.executable, "-m", "pip", "install", "-r", str(lock)])
    run([sys.executable, "-m", "pip", "check"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("compile", "test", "install-ci"))
    args = parser.parse_args()

    if args.action == "compile":
        compile_sources()
    elif args.action == "test":
        run_unit_tests()
    else:
        install_ci_dependencies()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
