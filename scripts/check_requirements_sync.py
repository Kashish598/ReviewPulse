"""Fail when requirements-serve.txt drifts from requirements.txt.

requirements-serve.txt is the slim list the Docker image installs. Every line in it must be an
exact pin that also appears, with the same version, in requirements.txt.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger("reviewpulse.requirements")

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;#]+)")


def parse_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = PIN.match(line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def unpinned_lines(text: str) -> list[str]:
    bad = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")) and not PIN.match(line):
            bad.append(line)
    return bad


def find_problems(serve_text: str, full_text: str) -> list[str]:
    problems = [f"not an exact pin: {line}" for line in unpinned_lines(serve_text)]
    full = parse_pins(full_text)
    for name, version in parse_pins(serve_text).items():
        if name not in full:
            problems.append(f"{name}=={version} is not in requirements.txt")
        elif full[name] != version:
            problems.append(f"{name}: serve has {version}, requirements.txt has {full[name]}")
    return problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    serve = (ROOT / "requirements-serve.txt").read_text(encoding="utf-8")
    full = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    problems = find_problems(serve, full)
    for problem in problems:
        logger.error(problem)
    if not problems:
        logger.info("requirements-serve.txt matches requirements.txt")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
