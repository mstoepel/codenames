"""Fuzzy-match an OCR'd token against the Codenames word list."""

from __future__ import annotations

import difflib
from functools import lru_cache
from pathlib import Path

_WORD_LIST_PATH = Path(__file__).resolve().parent.parent / "full_word_list.txt"


@lru_cache(maxsize=1)
def load_word_list(path: str | None = None) -> tuple[str, ...]:
    p = Path(path) if path else _WORD_LIST_PATH
    words = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip().upper()
        if line:
            words.append(line)
    return tuple(words)


def closest_word(raw: str, cutoff: float = 0.6) -> str | None:
    """Return the best fuzzy match for `raw` from the word list, or None."""
    if not raw:
        return None
    normalized = "".join(ch for ch in raw.upper() if ch.isalpha() or ch == "-")
    if not normalized:
        return None
    matches = difflib.get_close_matches(normalized, load_word_list(), n=1, cutoff=cutoff)
    return matches[0] if matches else None
