# scripts/check_requirements_sync.py. CI fails if any pin in the serve file differs from requirements.txt. 
# It protects against "works in tests, breaks in the image". 
# The test is tests/test_requirements_sync.py.

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_requirements_sync", ROOT / "scripts" / "check_requirements_sync.py"
)
sync = importlib.util.module_from_spec(_spec)
sys.modules["check_requirements_sync"] = sync
_spec.loader.exec_module(sync)


def test_parse_pins_ignores_comments_and_normalises_names():
    text = "# comment\nScikit_Learn==1.5.2\n\nfastapi==0.115.6  # api\n"

    assert sync.parse_pins(text) == {"scikit-learn": "1.5.2", "fastapi": "0.115.6"}


def test_matching_files_have_no_problems():
    assert sync.find_problems("fastapi==1.0\n", "fastapi==1.0\nruff==0.8.4\n") == []


def test_version_mismatch_and_missing_package_are_reported():
    problems = sync.find_problems("fastapi==1.0\nuvicorn==2.0\n", "fastapi==1.1\n")

    assert len(problems) == 2
    assert any("fastapi" in p and "1.1" in p for p in problems)
    assert any("uvicorn" in p for p in problems)


def test_unpinned_line_is_reported():
    assert sync.find_problems("fastapi>=1.0\n", "fastapi==1.0\n") == [
        "not an exact pin: fastapi>=1.0"
    ]


def test_the_real_files_are_in_sync():
    serve = (ROOT / "requirements-serve.txt").read_text(encoding="utf-8")
    full = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert sync.find_problems(serve, full) == []
