from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".html", ".json", ".md", ".py", ".sh", ".tex", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_SUBSTRINGS = tuple(
    "".join(characters)
    for characters in (
        ("s", "i", "d", "a"),
        ("l", "i", "u", "s", "i", "d", "a"),
        ("f", "e", "i", "j", "i", "a", "n", "g"),
        ("h", "a", "n", "f", "e", "i", "j", "i", "a", "n", "g"),
    )
)
FORBIDDEN_WORDS = ("".join(("c", "h", "a", "s", "e")),)
EXCLUDED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv"}


def test_repository_contains_no_author_identifiers() -> None:
    matches: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(value in text for value in FORBIDDEN_SUBSTRINGS):
            matches.append(str(path.relative_to(ROOT)))
            continue
        if any(re.search(rf"\b{re.escape(value)}\b", text) for value in FORBIDDEN_WORDS):
            matches.append(str(path.relative_to(ROOT)))

    assert not matches, f"author identifiers found in: {matches}"
